import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import aiosqlite

from core.llm import TokenUsage


class Database:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.environ.get("DB_PATH", "./data/sessions.db")
        # 确保数据目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def init_tables(self):
        """初始化数据库表"""
        async with aiosqlite.connect(self.db_path) as db:
            # 会话表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    final_answer TEXT,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            
            # Agent 执行记录表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS agent_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    agent_id TEXT,
                    task TEXT,
                    final_answer TEXT,
                    trajectory TEXT,
                    status TEXT DEFAULT 'running',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Token 使用统计表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    agent_id TEXT,
                    step_index INTEGER,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # 事件日志表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS event_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    agent_id TEXT,
                    event_type TEXT,
                    event_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            await db.commit()
    
    async def create_session(self, session_id: str, query: str):
        """创建新会话"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO sessions (id, query, status) VALUES (?, ?, ?)",
                (session_id, query, "running")
            )
            await db.commit()
    
    async def update_session(self, session_id: str, final_answer: str, status: str = "completed"):
        """更新会话状态"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE sessions SET final_answer = ?, status = ?, completed_at = ? WHERE id = ?",
                (final_answer, status, datetime.now().isoformat(), session_id)
            )
            await db.commit()
    
    async def create_agent_execution(self, session_id: str, agent_id: str, task: str):
        """创建 Agent 执行记录"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO agent_executions (session_id, agent_id, task, status) VALUES (?, ?, ?, ?)",
                (session_id, agent_id, task, "running")
            )
            await db.commit()
    
    async def update_agent_execution(
        self,
        session_id: str,
        agent_id: str,
        final_answer: str,
        trajectory: List[Dict],
        status: str = "completed"
    ):
        """更新 Agent 执行记录"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE agent_executions 
                   SET final_answer = ?, trajectory = ?, status = ?, completed_at = ?
                   WHERE session_id = ? AND agent_id = ?""",
                (final_answer, json.dumps(trajectory, ensure_ascii=False), status, 
                 datetime.now().isoformat(), session_id, agent_id)
            )
            await db.commit()
    
    async def save_token_usage(
        self,
        session_id: str,
        agent_id: str,
        step_index: int,
        usage: TokenUsage
    ):
        """保存 Token 使用记录"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO token_usage 
                   (session_id, agent_id, step_index, prompt_tokens, completion_tokens, total_tokens)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, agent_id, step_index, usage.prompt_tokens, 
                 usage.completion_tokens, usage.total_tokens)
            )
            await db.commit()
    
    async def get_session_token_summary(self, session_id: str) -> Dict[str, Any]:
        """获取会话 Token 汇总"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT agent_id, 
                          SUM(prompt_tokens) as total_prompt,
                          SUM(completion_tokens) as total_completion,
                          SUM(total_tokens) as total
                   FROM token_usage 
                   WHERE session_id = ?
                   GROUP BY agent_id""",
                (session_id,)
            )
            rows = await cursor.fetchall()
            
            agent_stats = {}
            total_prompt = 0
            total_completion = 0
            total_all = 0
            
            for row in rows:
                agent_id, prompt, completion, total = row
                agent_stats[agent_id] = {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total
                }
                total_prompt += prompt
                total_completion += completion
                total_all += total
            
            return {
                "session_id": session_id,
                "agent_breakdown": agent_stats,
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_all
            }
    
    async def log_event(self, session_id: str, agent_id: Optional[str], event_type: str, event_data: Dict):
        """记录事件日志"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO event_logs (session_id, agent_id, event_type, event_data) VALUES (?, ?, ?, ?)",
                (session_id, agent_id, event_type, json.dumps(event_data, ensure_ascii=False))
            )
            await db.commit()
    
    async def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话的所有事件"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT agent_id, event_type, event_data, created_at FROM event_logs WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            )
            rows = await cursor.fetchall()
            
            events = []
            for row in rows:
                agent_id, event_type, event_data, created_at = row
                events.append({
                    "agent_id": agent_id,
                    "event_type": event_type,
                    "event_data": json.loads(event_data),
                    "created_at": created_at
                })
            
            return events
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, query, final_answer, status, created_at, completed_at FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                return {
                    "id": row[0],
                    "query": row[1],
                    "final_answer": row[2],
                    "status": row[3],
                    "created_at": row[4],
                    "completed_at": row[5]
                }
            return None

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """按创建时间倒序列出会话，供 Replay UI 拉取历史。"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT id, query, final_answer, status, created_at, completed_at
                   FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "query": r[1],
                    "final_answer": r[2],
                    "status": r[3],
                    "created_at": r[4],
                    "completed_at": r[5],
                }
                for r in rows
            ]


# 全局数据库实例
db = Database()


async def init_database():
    """初始化数据库"""
    await db.init_tables()
