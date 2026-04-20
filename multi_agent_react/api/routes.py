import asyncio
import uuid
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from loguru import logger

from core.llm import DeepSeekClient
from core.context import SharedContext
from core.message_queue import mq, EventType
from core.orchestrator import Orchestrator
from core.tools.calculator import CalculatorTool, CalculatorToolSimple
from core.tools.search import SearchTool, WeatherTool, DateTimeTool
from storage.db import db


router = APIRouter()


# 后台任务强引用集合：asyncio 只对 Task 持弱引用，如果不保留强引用，
# GC 可能在任务跑完之前把它回收，导致会话永远卡在 running 且 SSE 永远
# 收不到 session_done。参考 Python 文档 asyncio.create_task 的警告。
_background_tasks: "set[asyncio.Task]" = set()


# 请求/响应模型
class QueryRequest(BaseModel):
    query: str
    num_agents: Optional[int] = 3
    max_iterations: Optional[int] = 10


class QueryAcceptedResponse(BaseModel):
    """/api/query 异步受理响应：仅返回 session_id，实际结果通过 SSE 推送。"""
    session_id: str
    status: str = "accepted"


class SessionResponse(BaseModel):
    id: str
    query: str
    final_answer: Optional[str]
    status: str
    created_at: str
    completed_at: Optional[str]


class TokenSummaryResponse(BaseModel):
    session_id: str
    agent_breakdown: dict
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int


class ThinkingResponse(BaseModel):
    agent_id: str
    thinking_steps: list


def _build_default_tools():
    """为一次查询构造默认工具集合。

    故意放在函数里而不是模块级：部分工具将来如果要注入 API key 等运行时配置，
    这里就是扩展点。
    """
    return [
        CalculatorTool(),
        CalculatorToolSimple(),
        SearchTool(),
        WeatherTool(),
        DateTimeTool(),
    ]


async def _run_query_background(
    session_id: str,
    query: str,
    num_agents: int,
    max_iterations: int,
) -> None:
    """在后台任务里跑 Orchestrator，将所有事件通过共享 mq 推给 SSE。"""
    try:
        llm = DeepSeekClient()
        context = SharedContext()
        orchestrator = Orchestrator(
            llm=llm,
            context=context,
            message_queue=mq,
            database=db,
            tools=_build_default_tools(),
            num_agents=num_agents,
            max_iterations=max_iterations,
        )
        await orchestrator.run(query, session_id=session_id)
    except Exception as e:
        # 保证任何内部异常都会以 error + session_done 的形式被 SSE 客户端看到，
        # 避免 Dashboard 永远停在 "处理中…" 状态。
        logger.exception(f"Background query {session_id} failed: {e}")
        await mq.publish(EventType.ERROR, None, {
            "session_id": session_id,
            "error": str(e),
        })
        try:
            await db.update_session(session_id, f"执行出错: {e}", status="error")
        except Exception:
            logger.exception("Failed to mark session as error")
        await mq.publish(EventType.SESSION_DONE, None, {
            "session_id": session_id,
            "final_answer": f"执行出错: {e}",
            "status": "error",
            "total_tokens": {"prompt": 0, "completion": 0, "total": 0},
        })


@router.post("/query", response_model=QueryAcceptedResponse, status_code=202)
async def create_query(request: QueryRequest) -> QueryAcceptedResponse:
    """异步受理查询：立即返回 session_id，客户端再用它订阅 SSE。

    之前的实现是同步阻塞——POST 会一直等到所有 Agent 跑完才返回 ``final_answer``，
    但此时会话已经结束，再去 ``GET /sse/stream/{session_id}`` 已经没有事件可订阅，
    "实时 Dashboard" 实际上拿不到任何中间事件。现在改成：

    1. 立即生成 session_id；
    2. 在 MessageQueue 里为该 session_id 预创建重放缓冲；
    3. 把 Orchestrator.run 丢进后台任务；
    4. 立即返回 ``{session_id, status="accepted"}``。

    客户端收到后去 ``/sse/stream/{session_id}`` 订阅即可；订阅建立时会自动
    把订阅窗口之前已经发生的事件先复播一遍，避免丢 agent_start。
    """
    try:
        session_id = str(uuid.uuid4())
        # 关键：在后台任务还没来得及 publish 任何事件之前就把缓冲建起来
        mq.ensure_session(session_id)
        
        task = asyncio.create_task(
            _run_query_background(
                session_id=session_id,
                query=request.query,
                num_agents=request.num_agents or 3,
                max_iterations=request.max_iterations or 10,
            ),
            name=f"query:{session_id}",
        )
        # 保留强引用直到任务结束，防止被 GC 悄悄回收
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        
        return QueryAcceptedResponse(session_id=session_id, status="accepted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """获取会话信息"""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        id=session["id"],
        query=session["query"],
        final_answer=session["final_answer"],
        status=session["status"],
        created_at=session["created_at"],
        completed_at=session["completed_at"]
    )


@router.get("/tokens/{session_id}", response_model=TokenSummaryResponse)
async def get_token_summary(session_id: str):
    """获取会话的 Token 使用统计"""
    summary = await db.get_session_token_summary(session_id)
    
    return TokenSummaryResponse(
        session_id=summary["session_id"],
        agent_breakdown=summary["agent_breakdown"],
        total_prompt_tokens=summary["total_prompt_tokens"],
        total_completion_tokens=summary["total_completion_tokens"],
        total_tokens=summary["total_tokens"]
    )


@router.get("/events/{session_id}")
async def get_session_events(session_id: str):
    """获取会话的所有事件"""
    events = await db.get_session_events(session_id)
    return {"session_id": session_id, "events": events}


@router.get("/thinking/{session_id}/{agent_id}")
async def get_agent_thinking(session_id: str, agent_id: str):
    """获取 Agent 的思考过程"""
    events = await db.get_session_events(session_id)
    
    thinking_steps = []
    for event in events:
        if event["agent_id"] == agent_id and event["event_type"] == "thinking":
            thinking_steps.append({
                "timestamp": event["created_at"],
                "content": event["event_data"].get("text", "")
            })
    
    return ThinkingResponse(
        agent_id=agent_id,
        thinking_steps=thinking_steps
    )
