import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from storage.db import init_database, db
from core.message_queue import mq, Message
from api.routes import router as api_router
from api.sse import router as sse_router
from logger.log import get_logger

# 加载环境变量
load_dotenv()

logger = get_logger()


async def _persist_event_to_db(message: Message) -> None:
    """把 MessageQueue 发布的每个事件落进 ``event_logs`` 表。

    有了这条持久化路径，``GET /api/events/{session_id}`` 才能返回真实历史，
    前端才能对已完成的 session 做"录像回放"。失败只记日志，不影响主流程。
    """
    session_id = message.data.get("session_id") if isinstance(message.data, dict) else None
    if not session_id:
        # 极个别无 session_id 的事件就不落库，反正前端也按 session_id 回捞。
        return
    try:
        await db.log_event(
            session_id=session_id,
            agent_id=message.agent_id,
            event_type=message.event_type,
            event_data=message.data,
        )
    except Exception as e:
        logger.warning(f"Failed to persist event {message.event_type}: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Starting Multi-Agent ReAct System...")
    await init_database()
    logger.info("Database initialized")
    mq.set_persistence_handler(_persist_event_to_db)
    logger.info("MessageQueue persistence handler attached (event_logs)")
    yield
    # 关闭时清理
    mq.set_persistence_handler(None)
    logger.info("Shutting down...")


# 创建 FastAPI 应用
app = FastAPI(
    title="Multi-Agent ReAct System",
    description="基于 DeepSeek 的多 Agent ReAct 框架",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix="/api")
app.include_router(sse_router, prefix="/sse")

# 静态文件服务（前端）
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "multi-agent-react"}


if __name__ == "__main__":
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
