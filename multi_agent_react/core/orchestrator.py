import os
import re
import json
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from core.llm import DeepSeekClient, TokenUsage
from core.agent import ReactAgent, AgentResult
from core.context import SharedContext
from core.message_queue import MessageQueue, EventType
from core.tools.base import BaseTool
from storage.db import Database


@dataclass
class SessionResult:
    session_id: str
    query: str
    final_answer: str
    agent_results: List[AgentResult]
    total_tokens: TokenUsage


# 命中任意一条即判定为"简单问题"，走单 agent fast-path：
#   - 纯算术表达式（例如 "2**10"、"3+4*5"）
#   - 含典型数学关键词（几次方/平方根/对数/百分比）
#   - 含一次性事实查询关键词（今天几号/现在几点/天气怎么样）
# 匹配策略故意宽松：出错的代价（退化为多 agent）比漏判代价（token 爆炸）小。
SIMPLE_QUERY_PATTERNS = [
    re.compile(r"^[\s\d\+\-\*\/\.\(\)%\^]+$"),
    re.compile(r"(几次方|次方|平方根|开方|对数|百分之|百分比)"),
    re.compile(r"(今天|现在|当前).{0,6}(几号|几点|时间|日期|星期)"),
    re.compile(r"(天气|气温|下雨|下雪).{0,8}(怎么样|如何|吗)?$"),
]
SIMPLE_QUERY_MAX_LEN = 40


class Orchestrator:
    """多 Agent 调度器，负责任务拆分、并行执行和结果合并。

    ``num_agents`` 在新版本里的语义是**上限**而不是硬性数量：
      - LLM decompose 返回几个子任务就跑几个 agent；
      - 超过 ``num_agents`` 会被截断；
      - 少于 ``num_agents`` 不再用 ``补充分析`` 强行凑数；
      - 如果最终只剩一个任务，会跳过 ``_merge_results`` 直接返回单 agent 的答案。
    对"2 的 10 次方"这类简单问题还会走 fast-path 直接跳过 decompose。
    """
    
    def __init__(
        self,
        llm: DeepSeekClient,
        context: SharedContext,
        message_queue: MessageQueue,
        database: Database,
        tools: List[BaseTool],
        num_agents: int = 3,
        max_iterations: int = 10
    ):
        self.llm = llm
        self.context = context
        self.mq = message_queue
        self.db = database
        self.tools = tools
        self.num_agents = max(1, num_agents)
        self.max_iterations = max_iterations
        
        self.use_thinking = os.environ.get("USE_THINKING_MODE", "false").lower() == "true"
    
    async def run(self, query: str, session_id: Optional[str] = None) -> SessionResult:
        """执行完整的多 Agent 工作流。

        ``session_id`` 可由调用方预先生成并传入，便于 API 层先返回 session_id 给
        客户端建立 SSE 订阅，再异步触发本方法，避免事件在 SSE 未连接时全部发完。
        调用方传入的 session_id 必须保证在 ``MessageQueue`` 里已 ``ensure_session``，
        并且 ``Database`` 里尚未创建对应记录（本方法会负责写入）。
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        logger.info(f"Starting session {session_id} with query: {query}")
        
        # 创建会话记录
        await self.db.create_session(session_id, query)
        
        # 步骤 1: 任务拆分（简单问题走 fast-path，不调 LLM）
        if self.num_agents <= 1 or self._is_simple_query(query):
            tasks = [query]
            logger.info(f"Fast-path: skipping decompose for session {session_id}")
        else:
            tasks = await self._decompose_task(query)
        logger.info(f"Task decomposed into {len(tasks)} subtasks")
        
        # 步骤 2: 并行执行（单任务就单 agent 跑，不必 gather）
        agent_results = await self._execute_parallel(session_id, tasks)
        logger.info(f"All agents completed for session {session_id}")
        
        # 步骤 3: 结果合并（单 agent 直接用它的答案，省一次 LLM 调用）
        if len(agent_results) == 1:
            final_answer = agent_results[0].final_answer
            logger.info(f"Single-agent path: skipping merge for session {session_id}")
        else:
            final_answer = await self._merge_results(query, agent_results)
            logger.info(f"Results merged for session {session_id}")
        
        # 更新会话状态
        await self.db.update_session(session_id, final_answer, "completed")
        
        # 计算总 token
        total_tokens = TokenUsage()
        for result in agent_results:
            total_tokens.prompt_tokens += result.total_tokens.prompt_tokens
            total_tokens.completion_tokens += result.total_tokens.completion_tokens
            total_tokens.total_tokens += result.total_tokens.total_tokens
        
        # 发布会话完成事件
        await self.mq.publish(EventType.SESSION_DONE, None, {
            "session_id": session_id,
            "final_answer": final_answer,
            "total_tokens": {
                "prompt": total_tokens.prompt_tokens,
                "completion": total_tokens.completion_tokens,
                "total": total_tokens.total_tokens
            }
        })
        
        return SessionResult(
            session_id=session_id,
            query=query,
            final_answer=final_answer,
            agent_results=agent_results,
            total_tokens=total_tokens
        )
    
    def _is_simple_query(self, query: str) -> bool:
        """轻量启发式：判断 query 是否是一个"不值得多 agent 并行"的简单问题。"""
        q = (query or "").strip()
        if not q:
            return False
        if len(q) <= SIMPLE_QUERY_MAX_LEN:
            for pat in SIMPLE_QUERY_PATTERNS:
                if pat.search(q):
                    return True
            # 纯算术：命中
            if SIMPLE_QUERY_PATTERNS[0].match(q):
                return True
        return False

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """把 ```json ... ``` / ``` ... ``` 这类 markdown 围栏去掉，只留中间内容。"""
        s = (text or "").strip()
        m = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", s, re.DOTALL)
        if m:
            return m.group(1).strip()
        return s

    async def _decompose_task(self, query: str) -> List[str]:
        """将任务拆分为 1~num_agents 个子任务。

        不再强行把结果补到 ``num_agents`` 个——简单问题让 LLM 回 1 个就好，
        否则会出现"补充分析: xxx"这种没信息量的占位任务，徒增 token 和耗时。
        """
        prompt = f"""请将以下用户查询拆分为若干个可并行执行的子任务，用于并行处理。

用户查询: {query}

要求:
1. 如果用户查询本身就是单一、简单的问题，直接返回只含原问题的单元素数组，**不要**强行拆分。
2. 否则最多拆成 {self.num_agents} 个子任务，每个子任务独立、覆盖不同方面。
3. 返回纯 JSON 数组，不要用 markdown 代码块包裹。

示例：
- 简单问题 "2的10次方是多少"  → ["2的10次方是多少"]
- 复杂问题 "对比北京和上海的天气与美食" → ["查询北京天气", "查询上海天气", "对比两地美食"]"""

        try:
            response = await self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                model="deepseek-chat"
            )

            content = self._strip_code_fence(response.content)

            try:
                tasks = json.loads(content)
                if isinstance(tasks, list) and tasks:
                    # 去掉非字符串、空字符串
                    cleaned = [t.strip() for t in tasks if isinstance(t, str) and t.strip()]
                    if cleaned:
                        if len(cleaned) > self.num_agents:
                            cleaned = cleaned[: self.num_agents]
                        return cleaned
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse task decomposition JSON: {content}")

        except Exception as e:
            logger.error(f"Task decomposition error: {e}")

        # 降级：就当作一个任务，交给单 agent 处理
        return [query]
    
    async def _execute_parallel(self, session_id: str, tasks: List[str]) -> List[AgentResult]:
        """并行执行多个 Agent"""
        
        async def run_agent(agent_id: str, task: str) -> AgentResult:
            agent = ReactAgent(
                agent_id=agent_id,
                tools=self.tools,
                context=self.context,
                llm=self.llm,
                message_queue=self.mq,
                max_iterations=self.max_iterations,
                use_thinking=self.use_thinking,
                session_id=session_id
            )
            
            # 创建执行记录
            await self.db.create_agent_execution(session_id, agent_id, task)
            
            result = await agent.run(task)
            
            # 更新执行记录
            await self.db.update_agent_execution(
                session_id=session_id,
                agent_id=agent_id,
                final_answer=result.final_answer,
                trajectory=result.trajectory,
                status="completed"
            )
            
            return result
        
        # 创建任务
        coroutines = []
        for i, task in enumerate(tasks):
            agent_id = f"agent-{i}"
            coroutines.append(run_agent(agent_id, task))
        
        # 并行执行
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 处理结果
        agent_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} failed: {result}")
                # 创建失败结果
                agent_results.append(AgentResult(
                    agent_id=f"agent-{i}",
                    task=tasks[i],
                    final_answer=f"执行失败: {str(result)}",
                    trajectory=[],
                    total_tokens=TokenUsage(),
                    status="error"
                ))
            else:
                agent_results.append(result)
        
        return agent_results
    
    async def _merge_results(self, query: str, agent_results: List[AgentResult]) -> str:
        """合并多个 Agent 的结果"""
        
        # 构建结果汇总
        results_summary = []
        for result in agent_results:
            results_summary.append(f"""[{result.agent_id}]
任务: {result.task}
结果: {result.final_answer}
""")
        
        results_text = "\n---\n".join(results_summary)
        
        prompt = f"""基于以下多个子任务的分析结果，请综合回答用户的原始问题。

用户原始问题: {query}

各子任务分析结果:
{results_text}

请:
1. 综合分析各子任务的结果
2. 整合成一个完整、连贯的答案
3. 如果子任务结果有冲突，请说明并给出最合理的结论
4. 直接给出最终答案，不要提及子任务分配"""
        
        try:
            response = await self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                model="deepseek-chat"
            )
            
            return response.content.strip()
        
        except Exception as e:
            logger.error(f"Result merging error: {e}")
            # 降级：简单拼接
            return "\n\n".join([f"[{r.agent_id}]\n{r.final_answer}" for r in agent_results])
