import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from core.message_queue import MessageQueue, mq


router = APIRouter()


@router.get("/stream/{session_id}")
async def event_stream(request: Request, session_id: str):
    """SSE 流式推送会话事件"""
    
    async def event_generator():
        # 订阅会话事件
        queue = mq.subscribe_session(session_id)
        
        try:
            while True:
                # 检查客户端是否断开连接
                if await request.is_disconnected():
                    break
                
                try:
                    # 等待消息，设置超时以便检查连接状态
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # 构建 SSE 事件
                    event_data = {
                        "event_type": message.event_type,
                        "agent_id": message.agent_id,
                        "data": message.data,
                        "timestamp": message.timestamp
                    }
                    
                    yield {
                        "event": message.event_type,
                        "data": json.dumps(event_data, ensure_ascii=False)
                    }
                
                except asyncio.TimeoutError:
                    # 发送心跳保持连接
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"timestamp": message.timestamp if 'message' in dir() else ""})
                    }
                    continue
        
        finally:
            # 取消订阅
            mq.unsubscribe_session(session_id, queue)
    
    return EventSourceResponse(event_generator())


@router.get("/stream/all")
async def all_events_stream(request: Request):
    """SSE 流式推送所有事件（用于 Dashboard）"""
    
    async def event_generator():
        # 订阅所有事件类型
        event_types = [
            "agent_start", "thinking", "thinking_delta", "content_delta",
            "action", "observation", "token_update", "context_compressed",
            "agent_done", "session_done", "error"
        ]
        
        queues = {et: mq.subscribe(et) for et in event_types}
        
        try:
            while True:
                if await request.is_disconnected():
                    break
                
                # 尝试从所有队列获取消息
                for event_type, queue in queues.items():
                    try:
                        message = queue.get_nowait()
                        
                        event_data = {
                            "event_type": message.event_type,
                            "agent_id": message.agent_id,
                            "data": message.data,
                            "timestamp": message.timestamp
                        }
                        
                        yield {
                            "event": message.event_type,
                            "data": json.dumps(event_data, ensure_ascii=False)
                        }
                    except asyncio.QueueEmpty:
                        continue
                
                # 短暂休眠避免 CPU 占用过高
                await asyncio.sleep(0.01)
        
        finally:
            # 取消所有订阅
            for event_type, queue in queues.items():
                mq.unsubscribe(event_type, queue)
    
    return EventSourceResponse(event_generator())
