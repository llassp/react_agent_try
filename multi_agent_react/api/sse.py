import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from core.message_queue import mq


router = APIRouter()


# session_done 事件出现后延迟多久关闭流，给客户端一点时间读到最后一条消息
SESSION_DONE_DRAIN_SECONDS = 0.5


@router.get("/stream/{session_id}")
async def event_stream(request: Request, session_id: str):
    """SSE 流式推送会话事件。

    - 订阅建立时由 MessageQueue 负责把重放缓冲里已有的事件先复播一遍
      （见 ``MessageQueue.subscribe_session``），因此即便客户端在事件开始
      产生之后才连上来，也不会丢 ``agent_start`` / ``thinking`` 等早期事件。
    - 客户端断开或收到 ``session_done`` 后自动结束，避免无限 hang。
    """
    
    async def event_generator():
        queue = mq.subscribe_session(session_id)
        
        try:
            while True:
                # 客户端断开立刻退出
                if await request.is_disconnected():
                    break
                
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 发送心跳保持连接（原实现用 `if 'message' in dir()` 
                    # 在首次 timeout 时一定走 else 分支；这里用明确的 ISO 时间戳）
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"timestamp": datetime.now().isoformat()})
                    }
                    continue
                
                event_data = {
                    "event_type": message.event_type,
                    "agent_id": message.agent_id,
                    "data": message.data,
                    "timestamp": message.timestamp,
                }
                yield {
                    "event": message.event_type,
                    "data": json.dumps(event_data, ensure_ascii=False),
                }
                
                # session_done 是终态事件：再给一点点时间让客户端读出，就关闭流
                if message.event_type == "session_done":
                    await asyncio.sleep(SESSION_DONE_DRAIN_SECONDS)
                    break
        finally:
            mq.unsubscribe_session(session_id, queue)
    
    return EventSourceResponse(event_generator())


@router.get("/stream/all")
async def all_events_stream(request: Request):
    """SSE 流式推送所有事件（用于 Dashboard 全局监控）。"""
    
    async def event_generator():
        event_types = [
            "agent_start", "thinking", "thinking_delta", "content_delta",
            "action", "observation", "token_update", "context_compressed",
            "agent_done", "session_done", "error",
        ]
        
        queues = {et: mq.subscribe(et) for et in event_types}
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                drained = False
                for event_type, queue in queues.items():
                    try:
                        message = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        continue
                    
                    drained = True
                    event_data = {
                        "event_type": message.event_type,
                        "agent_id": message.agent_id,
                        "data": message.data,
                        "timestamp": message.timestamp,
                    }
                    yield {
                        "event": message.event_type,
                        "data": json.dumps(event_data, ensure_ascii=False),
                    }
                
                # 只有当所有队列都空时才让出控制权，避免空转
                if not drained:
                    await asyncio.sleep(0.05)
        finally:
            for event_type, queue in queues.items():
                mq.unsubscribe(event_type, queue)
    
    return EventSourceResponse(event_generator())
