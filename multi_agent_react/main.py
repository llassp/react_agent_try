import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from storage.db import init_database
from api.routes import router as api_router
from api.sse import router as sse_router
from logger.log import get_logger

# 加载环境变量
load_dotenv()

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Starting Multi-Agent ReAct System...")
    await init_database()
    logger.info("Database initialized")
    yield
    # 关闭时清理
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
