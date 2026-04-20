"""会话级调度器。

原本这里是手写的 ``decompose → gather → merge`` 三段式。把真正的编排搬到 ``graph.py``
和 ``templates.py`` 之后，本模块只剩下两件事：

1. 把一次 HTTP 请求翻译成一张已编译的 ``Graph``（默认 ``classic`` 模板，用户可指定
   ``critic_loop`` 等）；
2. 驱动 ``GraphRunner`` 跑完，再负责 session 级的"写库 / 发 ``session_done``"收尾。

这样升级 PR-A 中改出来的简单问题 fast-path 等优化全被搬进 ``planner_node`` 里，
行为对齐之前；而"critic 反思重跑"这种新能力只需要新增一个 graph template，不用动
``Orchestrator``。
"""

import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from core.agent import AgentResult
from core.context import SharedContext
from core.graph import GraphRunner, GraphState
from core.llm import DeepSeekClient, TokenUsage
from core.message_queue import EventType, MessageQueue
from core.templates import TEMPLATES, TemplateDeps, build_graph
from core.tools.base import BaseTool
from storage.db import Database


DEFAULT_TEMPLATE = os.environ.get("ORCHESTRATOR_TEMPLATE", "classic")


@dataclass
class SessionResult:
    session_id: str
    query: str
    final_answer: str
    agent_results: List[AgentResult]
    total_tokens: TokenUsage


class Orchestrator:
    """按模板组一张图、跑完、收尾。

    兼容 PR-A 的构造签名——外部 ``api/routes.py`` 不需要改就能继续跑 ``classic``
    默认行为。新增一个可选 ``template`` 参数，允许选 ``critic_loop`` 等。
    """

    def __init__(
        self,
        llm: DeepSeekClient,
        context: SharedContext,
        message_queue: MessageQueue,
        database: Database,
        tools: List[BaseTool],
        num_agents: int = 3,
        max_iterations: int = 10,
        template: Optional[str] = None,
        max_retries: int = 1,
    ):
        self.llm = llm
        self.context = context
        self.mq = message_queue
        self.db = database
        self.tools = tools
        self.num_agents = max(1, num_agents)
        self.max_iterations = max_iterations
        self.template = template or DEFAULT_TEMPLATE
        self.max_retries = max(0, max_retries)
        self.use_thinking = os.environ.get("USE_THINKING_MODE", "false").lower() == "true"

    async def run(self, query: str, session_id: Optional[str] = None) -> SessionResult:
        """按 ``self.template`` 构图并驱动执行，最终发布 ``session_done``。

        ``session_id`` 可由调用方预先生成并传入，便于 API 层先返回 session_id 给
        客户端建 SSE 订阅，再异步触发本方法，避免事件在 SSE 未连接时全部发完。
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        logger.info(
            f"Starting session {session_id} template={self.template} query={query!r}"
        )

        await self.db.create_session(session_id, query)

        deps = TemplateDeps(
            llm=self.llm,
            context=self.context,
            mq=self.mq,
            db=self.db,
            tools=self.tools,
            num_agents=self.num_agents,
            max_iterations=self.max_iterations,
            use_thinking=self.use_thinking,
            max_retries=self.max_retries,
        )
        graph = build_graph(self.template, deps)
        runner = GraphRunner(graph=graph, mq=self.mq)

        state = GraphState(session_id=session_id, query=query)
        await runner.run(state)

        final_answer = state.final_answer or "(no final answer)"
        agent_results = state.agent_results

        await self.db.update_session(session_id, final_answer, "completed")

        total_tokens = TokenUsage()
        for result in agent_results:
            total_tokens.prompt_tokens += result.total_tokens.prompt_tokens
            total_tokens.completion_tokens += result.total_tokens.completion_tokens
            total_tokens.total_tokens += result.total_tokens.total_tokens
        # critic_loop 下 executor 可能被踩多轮，老轮次结果被 executor_node 搬到
        # state.extra["tokens_carry"] 里，这里补回来——不然 session_done 的 token
        # 统计会少算前面重试的部分。
        carry = state.extra.get("tokens_carry") if isinstance(state.extra, dict) else None
        if carry:
            total_tokens.prompt_tokens += int(carry.get("prompt", 0))
            total_tokens.completion_tokens += int(carry.get("completion", 0))
            total_tokens.total_tokens += int(carry.get("total", 0))

        await self.mq.publish(EventType.SESSION_DONE, None, {
            "session_id": session_id,
            "final_answer": final_answer,
            "total_tokens": {
                "prompt": total_tokens.prompt_tokens,
                "completion": total_tokens.completion_tokens,
                "total": total_tokens.total_tokens,
            },
            "template": self.template,
        })

        return SessionResult(
            session_id=session_id,
            query=query,
            final_answer=final_answer,
            agent_results=agent_results,
            total_tokens=total_tokens,
        )

    @staticmethod
    def list_templates() -> List[str]:
        """列出可用模板名，供前端下拉菜单或 ``/api/templates`` 使用。"""
        return sorted(TEMPLATES.keys())
