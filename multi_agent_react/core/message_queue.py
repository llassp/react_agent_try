import asyncio
from collections import deque
from typing import Dict, List, Callable, Any, Optional, Deque
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


# 每个 session 缓冲的最大事件数，防止长运行下的内存暴涨
SESSION_BUFFER_MAX = 5000


@dataclass
class Message:
    event_type: str
    agent_id: Optional[str]
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MessageQueue:
    """异步消息队列，用于 Agent 间通信和 SSE 推送

    特别地，对带 ``session_id`` 的事件会额外维护一个重放缓冲：
    ``/api/query`` 改成后台任务后，客户端拿到 session_id 再去建 SSE 连接之间存在时差，
    期间产生的 agent_start / thinking 等事件必须保留，否则 Dashboard 会看不到前期的执行轨迹。
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._all_events_queue: asyncio.Queue = asyncio.Queue()
        self._session_subscribers: Dict[str, List[asyncio.Queue]] = {}
        # session_id -> 事件重放缓冲（环形队列）
        self._session_buffer: Dict[str, Deque[Message]] = {}

    def ensure_session(self, session_id: str) -> None:
        """预先记录 session，保证第一条事件之前已存在重放缓冲。"""
        if session_id not in self._session_buffer:
            self._session_buffer[session_id] = deque(maxlen=SESSION_BUFFER_MAX)

    def discard_session(self, session_id: str) -> None:
        """会话完全结束后回收重放缓冲（按需调用）。"""
        self._session_buffer.pop(session_id, None)
        self._session_subscribers.pop(session_id, None)
    
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
        
        # 会话级：先进重放缓冲再派发给当前订阅者
        session_id = data.get("session_id")
        if session_id:
            buf = self._session_buffer.setdefault(session_id, deque(maxlen=SESSION_BUFFER_MAX))
            buf.append(message)
            if session_id in self._session_subscribers:
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
        """订阅特定会话的所有事件。

        连接建立时会把重放缓冲里已经发生的事件先完整复播一遍，
        再将该队列加入活跃订阅者列表，避免订阅窗口内的竞态条件丢事件。
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        buf = self._session_buffer.get(session_id)
        if buf:
            for msg in list(buf):
                try:
                    queue.put_nowait(msg)
                except asyncio.QueueFull:
                    logger.warning(f"Replay buffer overflowed queue for session {session_id}")
                    break
        self._session_subscribers.setdefault(session_id, []).append(queue)
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
