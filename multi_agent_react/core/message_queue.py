import asyncio
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class Message:
    event_type: str
    agent_id: Optional[str]
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MessageQueue:
    """异步消息队列，用于 Agent 间通信和 SSE 推送"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._all_events_queue: asyncio.Queue = asyncio.Queue()
        self._session_subscribers: Dict[str, List[asyncio.Queue]] = {}
    
    async def publish(self, event_type: str, agent_id: Optional[str], data: Dict[str, Any]):
        """发布消息到队列"""
        message = Message(event_type=event_type, agent_id=agent_id, data=data)
        
        # 推送到特定事件类型的订阅者
        if event_type in self._subscribers:
            for queue in self._subscribers[event_type]:
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for event type {event_type}")
        
        # 推送到全事件队列
        await self._all_events_queue.put(message)
        
        # 推送到会话订阅者
        session_id = data.get("session_id")
        if session_id and session_id in self._session_subscribers:
            for queue in self._session_subscribers[session_id]:
                try:
                    queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning(f"Queue full for session {session_id}")
        
        # 调用事件处理器
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(message))
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
        
        logger.debug(f"Published event: {event_type} from agent: {agent_id}")
    
    def subscribe(self, event_type: str) -> asyncio.Queue:
        """订阅特定类型的事件"""
        queue = asyncio.Queue(maxsize=1000)
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(queue)
        return queue
    
    def subscribe_session(self, session_id: str) -> asyncio.Queue:
        """订阅特定会话的所有事件"""
        queue = asyncio.Queue(maxsize=1000)
        if session_id not in self._session_subscribers:
            self._session_subscribers[session_id] = []
        self._session_subscribers[session_id].append(queue)
        return queue
    
    def unsubscribe(self, event_type: str, queue: asyncio.Queue):
        """取消订阅"""
        if event_type in self._subscribers and queue in self._subscribers[event_type]:
            self._subscribers[event_type].remove(queue)
    
    def unsubscribe_session(self, session_id: str, queue: asyncio.Queue):
        """取消会话订阅"""
        if session_id in self._session_subscribers and queue in self._session_subscribers[session_id]:
            self._session_subscribers[session_id].remove(queue)
    
    def on(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def off(self, event_type: str, handler: Callable):
        """移除事件处理器"""
        if event_type in self._event_handlers and handler in self._event_handlers[event_type]:
            self._event_handlers[event_type].remove(handler)
    
    async def get_all_events(self) -> Message:
        """获取所有事件（用于日志记录等）"""
        return await self._all_events_queue.get()


# 事件类型常量
class EventType:
    AGENT_START = "agent_start"
    THINKING = "thinking"
    THINKING_DELTA = "thinking_delta"
    CONTENT_DELTA = "content_delta"
    ACTION = "action"
    OBSERVATION = "observation"
    TOKEN_UPDATE = "token_update"
    CONTEXT_COMPRESSED = "context_compressed"
    AGENT_DONE = "agent_done"
    SESSION_DONE = "session_done"
    ERROR = "error"


# 全局消息队列实例
mq = MessageQueue()
