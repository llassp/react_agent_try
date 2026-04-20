from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends

from core.llm import DeepSeekClient
from core.context import SharedContext
from core.message_queue import MessageQueue
from core.orchestrator import Orchestrator
from core.tools.calculator import CalculatorTool, CalculatorToolSimple
from core.tools.search import SearchTool, WeatherTool, DateTimeTool
from storage.db import Database


router = APIRouter()


# 请求/响应模型
class QueryRequest(BaseModel):
    query: str
    num_agents: Optional[int] = 3
    max_iterations: Optional[int] = 10


class QueryResponse(BaseModel):
    session_id: str
    query: str
    final_answer: str
    total_tokens: dict


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


# 依赖注入
async def get_llm():
    return DeepSeekClient()


async def get_db():
    db = Database()
    await db.init_tables()
    return db


def get_mq():
    from core.message_queue import mq
    return mq


def get_context():
    return SharedContext()


@router.post("/query", response_model=QueryResponse)
async def create_query(
    request: QueryRequest,
    llm: DeepSeekClient = Depends(get_llm),
    db: Database = Depends(get_db),
    mq = Depends(get_mq),
    context: SharedContext = Depends(get_context)
):
    """创建新查询，启动多 Agent 处理"""
    try:
        # 准备工具
        tools = [
            CalculatorTool(),
            CalculatorToolSimple(),
            SearchTool(),
            WeatherTool(),
            DateTimeTool()
        ]
        
        # 创建调度器
        orchestrator = Orchestrator(
            llm=llm,
            context=context,
            message_queue=mq,
            database=db,
            tools=tools,
            num_agents=request.num_agents,
            max_iterations=request.max_iterations
        )
        
        # 执行
        result = await orchestrator.run(request.query)
        
        return QueryResponse(
            session_id=result.session_id,
            query=result.query,
            final_answer=result.final_answer,
            total_tokens={
                "prompt": result.total_tokens.prompt_tokens,
                "completion": result.total_tokens.completion_tokens,
                "total": result.total_tokens.total_tokens
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, db: Database = Depends(get_db)):
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
async def get_token_summary(session_id: str, db: Database = Depends(get_db)):
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
async def get_session_events(session_id: str, db: Database = Depends(get_db)):
    """获取会话的所有事件"""
    events = await db.get_session_events(session_id)
    return {"session_id": session_id, "events": events}


@router.get("/thinking/{session_id}/{agent_id}")
async def get_agent_thinking(session_id: str, agent_id: str, db: Database = Depends(get_db)):
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
