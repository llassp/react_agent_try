import os
import sys
from pathlib import Path
from loguru import logger


def setup_logger():
    """配置 loguru 日志"""
    log_path = os.environ.get("LOG_PATH", "./logs/app.log")
    
    # 确保日志目录存在
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    # 添加文件输出
    logger.add(
        log_path,
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        encoding="utf-8",
    )
    
    return logger


# 全局日志实例
logger = setup_logger()


def get_logger():
    """获取配置好的 logger 实例"""
    return logger
