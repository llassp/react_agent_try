"""DAG 编排的内置模板和节点工厂。

这里提供两个开箱即用的模板：

- ``classic``：等价于老 Orchestrator 的 ``decompose → gather → merge``。保证升级不
  破坏旧行为，也给用户一个"不要额外动脑"的默认值。
- ``critic_loop``：``planner → executor → critic → (merge | 回到 executor)``，最多
  重试 ``max_retries`` 次。演示真正的 DAG 能力——图里有环，critic 说不合格就把
  整个 executor 重跑一遍。

所有节点都以"工厂函数"的形式暴露：它们吃一个 ``TemplateDeps``（装着 llm、tools、
mq、db 等运行时依赖），返回一个 ``NodeFn``。这样图结构和实现就彻底解耦——以后想把
``CriticNode`` 替换成别的评估器，只要改工厂就行。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from core.agent import AgentResult, ReactAgent
from core.context import SharedContext
from core.graph import (
    Edge,
    EdgeCondition,
    Graph,
    GraphState,
    Node,
    NodeFn,
    always,
    extra_flag,
)
from core.llm import DeepSeekClient, TokenUsage
from core.message_queue import MessageQueue
from core.tools.base import BaseTool
from storage.db import Database


# 和老 Orchestrator 一致的"简单问题"判定，放在这里给 planner 直接复用：
SIMPLE_QUERY_PATTERNS = [
    re.compile(r"^[\s\d\+\-\*\/\.\(\)%\^]+$"),
    re.compile(r"(几次方|次方|平方根|开方|对数|百分之|百分比)"),
    re.compile(r"(今天|现在|当前).{0,6}(几号|几点|时间|日期|星期)"),
    re.compile(r"(天气|气温|下雨|下雪).{0,8}(怎么样|如何|吗)?$"),
]
SIMPLE_QUERY_MAX_LEN = 40


def _is_simple_query(query: str) -> bool:
    q = (query or "").strip()
    if not q:
        return False
    if len(q) <= SIMPLE_QUERY_MAX_LEN:
        for pat in SIMPLE_QUERY_PATTERNS:
            if pat.search(q):
                return True
        if SIMPLE_QUERY_PATTERNS[0].match(q):
            return True
    return False


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    m = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


@dataclass
class TemplateDeps:
    """一张图在执行时需要的所有外部依赖的捆绑。"""
    llm: DeepSeekClient
    context: SharedContext
    mq: MessageQueue
    db: Database
    tools: List[BaseTool]
    num_agents: int = 3
    max_iterations: int = 10
    use_thinking: bool = False
    # critic_loop 专用
    max_retries: int = 1


# ---------- 节点工厂 ----------

def planner_node(deps: TemplateDeps) -> Node:
    """把 ``state.query`` 拆成 ``state.tasks``。

    与老 Orchestrator 的 decompose 行为一致：
    - ``num_agents <= 1`` 或简单问题 → 不调 LLM，直接 ``tasks=[query]``；
    - 否则让 LLM 返回 JSON 数组，截断到 ``num_agents`` 上限；
    - LLM 坏掉或 JSON 解析失败 → 降级成 ``[query]``。
    """

    async def _fn(state: GraphState) -> None:
        if deps.num_agents <= 1 or _is_simple_query(state.query):
            state.tasks = [state.query]
            logger.info(f"[planner] fast-path, single task for session {state.session_id}")
            return

        prompt = f"""请将以下用户查询拆分为若干个可并行执行的子任务，用于并行处理。

用户查询: {state.query}

要求:
1. 如果用户查询本身就是单一、简单的问题，直接返回只含原问题的单元素数组，**不要**强行拆分。
2. 否则最多拆成 {deps.num_agents} 个子任务，每个子任务独立、覆盖不同方面。
3. 返回纯 JSON 数组，不要用 markdown 代码块包裹。

示例：
- 简单问题 "2的10次方是多少"  → ["2的10次方是多少"]
- 复杂问题 "对比北京和上海的天气与美食" → ["查询北京天气", "查询上海天气", "对比两地美食"]"""

        try:
            response = await deps.llm.call(
                messages=[{"role": "user", "content": prompt}],
                model="deepseek-chat",
            )
            content = _strip_code_fence(response.content)
            tasks = json.loads(content)
            if isinstance(tasks, list) and tasks:
                cleaned = [t.strip() for t in tasks if isinstance(t, str) and t.strip()]
                if cleaned:
                    if len(cleaned) > deps.num_agents:
                        cleaned = cleaned[: deps.num_agents]
                    state.tasks = cleaned
                    return
        except json.JSONDecodeError:
            logger.warning(f"[planner] JSON parse failed, falling back to single task")
        except Exception as e:
            logger.error(f"[planner] decompose failed: {e}")

        state.tasks = [state.query]

    return Node(
        name="planner",
        fn=_fn,
        description="拆解 query 成子任务（简单问题走 fast-path）",
    )


def executor_node(deps: TemplateDeps) -> Node:
    """对 ``state.tasks`` 起并行 Agent，结果写回 ``state.agent_results``。

    注意：被多次踩到时（critic 要求重跑），老的 agent_results 会被清空，原地重跑，
    Dashboard 上对应的 agent card 会被新事件刷新。数据库里的 agent_execution 也会
    再插一条新纪录，不会覆盖上一轮——方便事后审计"critic 踢了我几次"。
    """

    async def _fn(state: GraphState) -> None:
        if not state.tasks:
            # planner 挂掉或返回空，兜一个
            state.tasks = [state.query]

        # 每次 executor 运行都是一轮。critic 会让它跑第 2、3 轮，agent_id 要带上轮数后缀
        # 才不会被前端事件流当成"同一个 agent 发了两次 agent_start"。
        round_idx = int(state.extra.get("executor_round", 0))
        state.extra["executor_round"] = round_idx + 1
        state.agent_results = []

        async def run_one(i: int, task: str) -> AgentResult:
            agent_id = f"agent-{i}" if round_idx == 0 else f"agent-{i}-r{round_idx}"
            agent = ReactAgent(
                agent_id=agent_id,
                tools=deps.tools,
                context=deps.context,
                llm=deps.llm,
                message_queue=deps.mq,
                max_iterations=deps.max_iterations,
                use_thinking=deps.use_thinking,
                session_id=state.session_id,
            )
            await deps.db.create_agent_execution(state.session_id, agent_id, task)
            try:
                result = await agent.run(task)
            except Exception as e:
                logger.exception(f"[executor] agent {agent_id} crashed: {e}")
                return AgentResult(
                    agent_id=agent_id,
                    task=task,
                    final_answer=f"执行失败: {e}",
                    trajectory=[],
                    total_tokens=TokenUsage(),
                    status="error",
                )
            await deps.db.update_agent_execution(
                session_id=state.session_id,
                agent_id=agent_id,
                final_answer=result.final_answer,
                trajectory=result.trajectory,
                status="completed",
            )
            return result

        results = await asyncio.gather(
            *(run_one(i, t) for i, t in enumerate(state.tasks)),
            return_exceptions=False,
        )
        state.agent_results = list(results)

    return Node(
        name="executor",
        fn=_fn,
        description="并行跑 ReAct agent（可被 critic 触发重跑）",
    )


def merger_node(deps: TemplateDeps) -> Node:
    """把多个 AgentResult 合成最终答案。单结果直接透传，省一次 LLM 调用。"""

    async def _fn(state: GraphState) -> None:
        if not state.agent_results:
            state.final_answer = "(执行失败：没有任何 agent 结果)"
            return

        if len(state.agent_results) == 1:
            state.final_answer = state.agent_results[0].final_answer
            return

        summary = "\n---\n".join(
            f"[{r.agent_id}]\n任务: {r.task}\n结果: {r.final_answer}\n"
            for r in state.agent_results
        )
        prompt = f"""基于以下多个子任务的分析结果，请综合回答用户的原始问题。

用户原始问题: {state.query}

各子任务分析结果:
{summary}

请:
1. 综合分析各子任务的结果
2. 整合成一个完整、连贯的答案
3. 如果子任务结果有冲突，请说明并给出最合理的结论
4. 直接给出最终答案，不要提及子任务分配"""

        try:
            response = await deps.llm.call(
                messages=[{"role": "user", "content": prompt}],
                model="deepseek-chat",
            )
            state.final_answer = (response.content or "").strip()
        except Exception as e:
            logger.error(f"[merger] failed: {e}, degrading to concat")
            state.final_answer = "\n\n".join(
                f"[{r.agent_id}]\n{r.final_answer}" for r in state.agent_results
            )

    return Node(
        name="merger",
        fn=_fn,
        description="把多 agent 结果合成最终答案（单结果直接透传）",
    )


def critic_node(deps: TemplateDeps) -> Node:
    """让 LLM 评判 merger 合出来的答案够不够好。

    写入 ``state.extra``：
    - ``critic_passed`` ∈ {True, False}
    - ``critic_feedback``：文字反馈，用于下一轮 executor 参考
    - ``critic_retries``：本 session 内 critic 已经要求重跑的次数
    """

    async def _fn(state: GraphState) -> None:
        retries = int(state.extra.get("critic_retries", 0))

        # 先合一个"当前候选答案"出来给 critic 看：如果 merger 还没跑过，就用 agent_results 拼
        candidate = state.final_answer
        if not candidate:
            if state.agent_results:
                candidate = "\n---\n".join(
                    f"[{r.agent_id}] {r.final_answer}" for r in state.agent_results
                )
            else:
                candidate = "(空)"

        prompt = f"""你是一个答案质量评审。请判断下面这个候选答案是否足够回答用户的原始问题。

用户原始问题: {state.query}

候选答案:
{candidate}

请返回严格的 JSON，格式:
{{"passed": true 或 false, "reason": "一句话说明"}}
- passed=true: 答案完整、直接、没有明显错误；
- passed=false: 答案缺失、跑题、或含明显事实错误。此时 reason 要指出缺什么，executor 会据此重跑。
不要输出 markdown 代码块。"""

        passed = True
        feedback = ""
        try:
            response = await deps.llm.call(
                messages=[{"role": "user", "content": prompt}],
                model="deepseek-chat",
            )
            content = _strip_code_fence(response.content)
            verdict = json.loads(content)
            passed = bool(verdict.get("passed", True))
            feedback = str(verdict.get("reason", "")).strip()
        except Exception as e:
            # critic 本身挂了就放行——不能让评审机制自己把 session 卡死
            logger.warning(f"[critic] self-error, treating as passed: {e}")
            passed = True
            feedback = f"critic error: {e}"

        # 限流：不允许无限循环
        if not passed and retries >= deps.max_retries:
            logger.info(
                f"[critic] max_retries={deps.max_retries} reached, forcing pass "
                f"to avoid infinite loop"
            )
            passed = True
            feedback = (feedback + " (retry limit reached)").strip()

        state.extra["critic_passed"] = passed
        state.extra["critic_feedback"] = feedback
        if not passed:
            state.extra["critic_retries"] = retries + 1
            # 下一轮 executor 会看到这个反馈（通过 task 拼接）
            state.tasks = [
                f"{t}\n\n[上一轮评审反馈] {feedback}" for t in state.tasks
            ]
            # 清掉旧答案，避免 merger 再次跑时拿到过期候选
            state.final_answer = None

    return Node(
        name="critic",
        fn=_fn,
        description="LLM 评审候选答案，不合格则指示 executor 重跑",
    )


# ---------- 模板工厂 ----------

def build_classic_graph(deps: TemplateDeps) -> Graph:
    """老行为：planner → executor → merger。无环，和升级前对齐。"""
    g = Graph(entry="planner")
    g.add_node(planner_node(deps))
    g.add_node(executor_node(deps))
    g.add_node(merger_node(deps))
    g.add_edge("planner", "executor")
    g.add_edge("executor", "merger")
    g.add_edge("merger", None)
    return g


def build_critic_loop_graph(deps: TemplateDeps) -> Graph:
    """planner → executor → merger → critic → {结束 | 回 executor}。

    critic 通过（``critic_passed=True``）→ 结束；不通过 → 回到 executor 带反馈重跑。
    ``deps.max_retries`` 控制最多回到 executor 的次数。
    """
    g = Graph(entry="planner")
    g.add_node(planner_node(deps))
    g.add_node(executor_node(deps))
    g.add_node(merger_node(deps))
    g.add_node(critic_node(deps))

    g.add_edge("planner", "executor")
    g.add_edge("executor", "merger")
    g.add_edge("merger", "critic")
    # 先判断通过：通过就结束；否则回到 executor
    g.add_edge("critic", None, condition=extra_flag("critic_passed", True), label="critic_passed")
    g.add_edge("critic", "executor", label="critic_retry")
    return g


TEMPLATES = {
    "classic": build_classic_graph,
    "critic_loop": build_critic_loop_graph,
}


def build_graph(template: str, deps: TemplateDeps) -> Graph:
    """按名字拿一张已编译的图。未知模板 → 降级 ``classic``。"""
    factory = TEMPLATES.get(template)
    if factory is None:
        logger.warning(f"Unknown template {template!r}, falling back to 'classic'")
        factory = TEMPLATES["classic"]
    return factory(deps)
