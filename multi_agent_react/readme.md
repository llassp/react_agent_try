# Multi-Agent ReAct Framework

基于 ReAct（Reasoning + Acting）架构的多 Agent 协同系统，支持并行任务执行、上下文共享、消息队列、结构化日志、前端实时展示，以及 Agent 推理进度与 Token 用量的可视化 Dashboard。

## 项目结构

```
multi_agent_react/
├── CLAUDE.md              # 项目说明文档
├── requirements.txt       # Python 依赖
├── .env                   # 环境变量配置
├── main.py                # FastAPI 启动入口
├── core/                  # 核心模块
│   ├── llm.py            # DeepSeek 客户端封装
│   ├── agent.py          # 单 Agent ReAct 循环
│   ├── orchestrator.py   # 多 Agent 调度器
│   ├── context.py        # 上下文共享 + 摘要压缩
│   ├── message_queue.py  # 消息队列（asyncio.Queue）
│   └── tools/            # 工具模块
│       ├── base.py       # 工具基类
│       ├── calculator.py # 计算器工具
│       └── search.py     # 搜索工具
├── storage/              # 数据持久化
│   └── db.py             # SQLite 数据库操作
├── logger/               # 日志模块
│   └── log.py            # loguru 配置
├── api/                  # API 接口
│   ├── routes.py         # REST API 路由
│   └── sse.py            # SSE 流式推送
├── frontend/             # 前端
│   └── index.html        # 实时 Dashboard
└── tests/                # 测试目录
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| Agent 框架 | 手写 ReAct 循环（不依赖 LangChain） |
| LLM 接入 | DeepSeek V3.2，通过 OpenAI SDK 兼容接口调用 |
| 异步并发 | `asyncio` + `asyncio.Queue` |
| 持久化 | SQLite（`aiosqlite`） |
| 日志 | `loguru` |
| 后端 API | `FastAPI` + SSE 流式推送 |
| 前端 | 原生 HTML + JS（EventSource）含实时 Dashboard |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 模型配置
MODEL_CHAT=deepseek-chat
MODEL_REASONER=deepseek-reasoner
USE_THINKING_MODE=false

# 上下文管理（基于 128k 窗口）
MAX_CONTEXT_TOKENS=128000
CONTEXT_COMPRESS_THRESHOLD=0.7
TOOL_RESULT_MAX_CHARS=20000

# Agent 配置
MAX_AGENT_ITERATIONS=10

# 基础设施
DB_PATH=./data/sessions.db
LOG_PATH=./logs/app.log
```

### 3. 启动服务

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问 Dashboard

打开浏览器访问：http://localhost:8000

## API 接口

### 创建查询

```bash
POST /api/query
Content-Type: application/json

{
  "query": "你的问题",
  "num_agents": 3,
  "max_iterations": 10
}
```

### 获取会话信息

```bash
GET /api/sessions/{session_id}
```

### 获取 Token 统计

```bash
GET /api/tokens/{session_id}
```

### SSE 流式事件

```bash
GET /sse/stream/{session_id}
```

## 核心特性

### 1. ReAct Agent

- 支持 function calling 模式
- 支持思考模式（deepseek-reasoner）
- 自动工具调用和结果处理
- 最大迭代次数限制

### 2. 多 Agent 调度

- 任务自动拆分
- 并行执行
- 结果合并

### 3. 上下文管理

- 128k token 上下文窗口
- 自动摘要压缩（阈值 70%）
- 工具结果压缩

### 4. 实时 Dashboard

- Token 消耗实时监控
- Agent 状态展示
- 思考过程可视化
- 事件日志流
- Token 趋势图表

### 5. 数据持久化

- SQLite 数据库存储
- Token 使用统计
- 事件日志记录

## 事件类型

| 事件 | 说明 |
|------|------|
| agent_start | Agent 开始执行 |
| thinking | 思考内容（deepseek-reasoner） |
| action | 工具调用 |
| observation | 工具返回结果 |
| token_update | Token 消耗更新 |
| context_compressed | 上下文压缩 |
| agent_done | Agent 完成 |
| session_done | 会话完成 |
| error | 错误事件 |

## 工具列表

- `calculator`: 数学表达式计算
- `calculate`: 简化版计算器
- `search`: 网络搜索（模拟）
- `weather`: 天气查询（模拟）
- `datetime`: 日期时间获取

## 注意事项

1. 需要设置 `DEEPSEEK_API_KEY` 环境变量
2. 首次启动会自动创建数据库表
3. 思考模式（USE_THINKING_MODE=true）会消耗更多 token
4. 上下文压缩阈值可通过环境变量调整
