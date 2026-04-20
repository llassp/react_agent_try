# 踩坑经验总结

## 1. aiosqlite 连接使用方式

**问题**: 使用 `async with await self._get_conn() as db` 会导致 `RuntimeError: threads can only be started once`

**原因**: `aiosqlite.connect()` 返回的连接对象本身已经是异步上下文管理器，不需要再 `await` 一次

**正确做法**:
```python
# 错误
async with await self._get_conn() as db:
    ...

# 正确
async with aiosqlite.connect(self.db_path) as db:
    ...
```

## 2. Python 异步数据库连接模式

**规则**: 对于支持异步上下文管理器的数据库库（如 aiosqlite、asyncpg），直接使用 `async with` 包装连接创建函数，而不是先 await 获取连接对象再使用。

**通用模式**:
```python
# 推荐模式
async with aiosqlite.connect(db_path) as db:
    await db.execute(...)

# 避免模式
db = await aiosqlite.connect(db_path)  # 不要在 async with 外创建
async with db:
    ...
```

## 3. FastAPI 生命周期中的数据库初始化

**规则**: 在 FastAPI 的 `@asynccontextmanager` lifespan 函数中初始化数据库时，确保数据库连接代码是线程安全的，避免在应用启动阶段创建多个连接池实例。

**建议**: 使用单例模式管理数据库连接，或在 lifespan 中只执行一次初始化逻辑。
