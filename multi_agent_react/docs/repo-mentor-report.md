# 《Multi-Agent ReAct》项目解读报告（Repo Mentor 版）

> 目标读者：想从这份代码里学会"如何工程化地写一个多 Agent 框架"的开发者。
> 讲解原则：**不把坏实践包装成最佳实践**。下面每一条结论都要能指到文件和行号，
> 不对号入座的空话一律删掉。

本报告基于分支 `main` 在合并 PR #1 ~ PR #4 之后的代码状态：
- PR #1 修好 ReAct function calling 链路 + `/api/query` 异步化 + SSE 重放
- PR #2 补 SSE 具名事件订阅 / 后台任务强引用
- PR #3 组合拳压 token + `event_logs` 写通 + Replay UI
- PR #4 DAG 节点图编排（`classic` + `critic_loop` 两个模板）

---

## 1. 项目鸟瞰图

### 1.1 一句话定位
一个**教学级**的"基于 DeepSeek 的多 Agent ReAct 框架"：FastAPI 后端驱动、
SQLite 做持久化、SSE 把运行事件实时推给原生 HTML Dashboard；**不依赖
LangChain / LangGraph**，ReAct 循环、工具注册、上下文压缩、图编排全是手写。
代码总量约 2.7k 行 Python，单可读性高。

### 1.2 技术栈
- **Python 3.11+** / asyncio / dataclass
- **FastAPI** + `sse-starlette`（SSE）
- **OpenAI 兼容 SDK**（`openai.AsyncOpenAI` 指向 DeepSeek base_url）
- **aiosqlite**（4 张表，零索引）
- **loguru**（日志）/ `python-dotenv`
- **原生 HTML/CSS/JS** 一个文件（`frontend/index.html`，1.3k 行）

### 1.3 代码分层
```
multi_agent_react/
├── main.py                     # lifespan / FastAPI 装配 / 持久化 handler 绑定
├── api/
│   ├── routes.py               # /api/query、/api/sessions 等 REST 路由
│   └── sse.py                  # /sse/stream/{session_id} 服务端推送
├── core/
│   ├── graph.py       (Golden) # DAG 内核：GraphState / Node / Edge / GraphRunner
│   ├── templates.py            # 内置模板：classic / critic_loop（节点工厂 + 图工厂）
│   ├── orchestrator.py         # 按模板编图、驱动 Runner、发 session_done 收尾
│   ├── agent.py                # ReactAgent：ReAct 循环 + function calling
│   ├── context.py     (Toxic)  # SharedContext：按 agent_id 维护 messages + 压缩
│   ├── llm.py         (Golden) # DeepSeekClient：重试 + reasoner 特判 + tool_calls 解析
│   ├── message_queue.py (Golden) # 事件总线 + session 重放缓冲 + 持久化 handler
│   └── tools/
│       ├── base.py             # BaseTool（OpenAI tools schema）
│       ├── calculator.py       # 表达式求值（eval 沙箱）
│       └── search.py           # mock 的 search / weather / datetime
├── storage/
│   └── db.py          (Toxic)  # Database：每次操作一次 connect，四张表无索引
├── logger/log.py
└── frontend/index.html         # Dashboard（agent 卡片 / token 图 / Replay / DAG 面板）
```

### 1.4 核心业务主链路

```
POST /api/query  ── 立即返回 session_id (202) ──┐
                                                 │
   asyncio.create_task(_run_query_background) ───┘
                        │
                        ▼
      Orchestrator.run(query, session_id)
                        │
                        ▼
      build_graph("classic" | "critic_loop", deps)   ← core/templates.py
                        │
                        ▼
      GraphRunner.run(state)         ← core/graph.py
          │  (publish graph_start)
          │
          │  planner_node ──► executor_node ──► merger_node ──► (critic_node) ──► end
          │                        │                                   │
          │                        ▼                                   │回路(不通过)
          │                asyncio.gather(run_one, ...)                │
          │                        │                                   │
          │                        ▼                                   │
          │                  ReactAgent.run(task)                      │
          │                        │ (ReAct 循环)                      │
          │                        ├─ context.maybe_compress           │
          │                        ├─ llm.call(messages, tools)        │
          │                        ├─ tool_calls? ─ y ─ _execute_tool  │
          │                        │                     ▲             │
          │                        │                     │             │
          │                        └─ no tool_calls → final_answer     │
          │                                                            │
          │  (node_start / node_done / agent_start / thinking /        │
          │   action / observation / token_update / agent_done 全部    │
          │   经由 MessageQueue.publish)                               │
          ▼
   session_done 事件

GET /sse/stream/{session_id}
      └─► MessageQueue.subscribe_session(session_id)
             └─► 重放缓冲 + 实时推送 → Dashboard
      (同时 MessageQueue.persistence_handler 把每个事件写进 event_logs 表，
       供 GET /api/events/{session_id} 做历史回放)
```

### 1.5 变更热度分区（从 git log + 文件大小）

| 分区 | 文件 | 变更频率 | 说明 |
| --- | --- | --- | --- |
| **核心稳定区** | `core/graph.py`, `core/message_queue.py` | 低 | 新内核，API 面向模板层，抽象稳定 |
| **高变更区** | `core/templates.py`, `core/agent.py`, `api/routes.py` | 高 | 业务/prompt/模板落点，几乎每个 PR 都动 |
| **高耦合区** | `core/context.py`, `frontend/index.html` | 中 | context.py 内聚了"消息结构 + token 估算 + 压缩策略"三件事 |
| **陈旧区** | `storage/db.py`, `core/tools/search.py` | 低 | 接口粗糙没有索引；search/weather 仍是 mock |

### 1.6 工程化现状（被迫说真话）
- **没有单测**（`tests/` 目录不存在）
- **没有 CI 外的 lint/format**（Devin Review 之外），没有 pre-commit、没有 ruff/black 配置
- **没有 Dockerfile**
- **`__pycache__/` 和 `data/*.db` 没有进 `.gitignore`**（仓库里能看到 `.pyc`）
- **CORS 全开 `allow_origins=["*"]`**，没有任何鉴权

---

## 2. 学习区 vs 避雷区（Golden / Toxic）

### 🟢 Golden Code #1：`core/graph.py` —— 小而美的 DAG 内核

**证据点**
1. 把"编排"抽成 4 个 dataclass + 2 个类（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/graph.py" lines="40-95" />），总共 263 行。
2. 出边"按加入顺序短路选第一条条件为真的"（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/graph.py" lines="218-232" />），语义极易推理。
3. `max_steps=64` 当成"坏图保险丝"（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/graph.py" lines="184-189" />）。有回路就必须有这种上限，否则 `critic_loop` 一旦反馈失灵会把钱烧光。
4. 节点级事件 `node_start/node_done/node_error` 统一打给 MQ，前端能直接渲染"当前走到哪"。
5. `Graph.describe()` 把结构序列化成 `nodes + edges` list（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/graph.py" lines="140-153" />），这是让前端"零业务耦合"画流程图的关键。

**学习建议（该精读什么）**
- `GraphRunner.run` 的循环：看"节点副作用写 state / 出边读 state"怎么约束并发复杂度。
- `extra_flag("critic_passed", True)` 这种 edge condition helper——**条件应该是数据，不是代码**。

**可迁移方法论**
- 当你看到代码里出现"A 跑完然后 B，除非某条件满足就跳到 C"的 if-else 树，立刻把它想成"节点 + 条件边"。状态机/DAG 换成显式数据模型后，单测和可视化都变得廉价。

---

### 🟢 Golden Code #2：`core/message_queue.py` —— 事件总线怎么写才不踩坑

**证据点**
1. **同一份事件走三路**：事件类型订阅、session 订阅、全局订阅（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/message_queue.py" lines="68-94" />），互不阻塞。
2. **session 重放缓冲（ring buffer）**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/message_queue.py" lines="52-60" />）：解决"POST 立刻返回 session_id，客户端再来订 SSE"的"已发完的事件怎么补"这一常见异步痛点；最大 5000 条封顶，内存不会失控。
3. **异步 handler 有强引用保护**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/message_queue.py" lines="62-66" />）：Python 官方文档明确警告 `asyncio.create_task` 只持弱引用，生产代码里忘了这一点就会"任务在跑一半被 GC 掉"。
4. **持久化 handler 解耦**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/message_queue.py" lines="48-50" />）：MQ 本身不依赖 DB；想关就 `set_persistence_handler(None)`。
5. 订阅建立时**先复播缓冲再推新事件**，避免"先看到后半段，前半段看不到"。

**学习建议**
- 读 `publish()` 的顺序：**先推订阅者，再落 buffer，再派发持久化任务**。顺序错了会出现"持久化 handler 慢 → 订阅推送被拖慢"。
- `ensure_session / discard_session` 这种"显式生命周期方法"比"lazy create"更好调试。

**可迁移方法论**
- **热路径里不要 await I/O**。持久化、外部 webhook 一律 `asyncio.create_task`，主路径只负责把事件塞队列。

---

### 🟢 Golden Code #3：`core/llm.py` —— LLM Client 的小而正

**证据点**
1. `LLMResponse` 把 `tool_calls` 做成**结构化字段**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/llm.py" lines="17-29" />），同时保留 `raw`——结构化好用，原始能原样回放进上下文。
2. **deepseek-reasoner 特判**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/llm.py" lines="74-78" /> + <ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/llm.py" lines="93-94" />）：不传 `temperature`、读 `reasoning_content`——这就是"provider 抽象之下的模型差异适配"落点。
3. **指数退避区分 429 / 5xx / 其他**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/llm.py" lines="113-131" />）：429 是"调用方活该等"，5xx 是"服务端的锅值得重试"，其他 4xx 直接 raise——这三档区分是生产级客户端的标配。
4. 返回 `usage` 是**真实的 API usage**（`response.usage.prompt_tokens`），而不是 `len(content)//4` 瞎算。

**学习建议**
- 复制这份重试骨架到你自己的项目：429/5xx 用指数退避、其他 4xx 直接抛、透传原因到日志。
- 看 `_parse_tool_calls` 的容错——JSON 解析失败用 `{"_raw": ...}` 兜住，不要让下游 agent 代码处理 None。

**可迁移方法论**
- 任何 provider 客户端都应该有 **"1 份结构化 + 1 份 raw 透传"** 的双字段模式；结构化给调用方用，raw 给需要原样复演的地方用。

---

### 🔴 Toxic Code #1：`core/context.py` —— 多个坏味道揉在一个文件里

**证据点（一眼可见的）**
1. **用 `len(content)//4` 估 token**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/context.py" lines="40-41" /> / <ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/context.py" lines="55-56" />）。对 **中文** 严重低估，一个汉字平均≥1 token；压缩策略会**太晚触发**。这事儿已经在 lessons-learned 里记过，但代码里仍是 `//4`。
2. **压缩=丢细节**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/context.py" lines="179-185" />）：触发压缩就把所有 non-system 消息替换成一条"[历史摘要] ..."。
   代价：**丢 tool_calls 结构**、**丢最近 1-2 轮原始细节**。生产做法通常是"保留最近 N 条 + 摘要前面"（sliding window summary）。
3. **LLM 被硬编码为 `deepseek-chat`**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/context.py" lines="174-175" />）。SharedContext 本来和具体 provider 无关，这里直接耦合死。
4. **`_shared_memory` 死字段**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/context.py" lines="30-31" /> + <ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/context.py" lines="223-229" />）：`SharedContext` 叫"共享"，实际只是 per-agent messages + 一个从没被真正写入过的 dict。命名和现实严重脱节。
5. **压缩阈值一刀切**：默认 `MAX_CONTEXT_TOKENS=16000`、阈值 0.5 是好的收敛，但仍然是"整个对话一视同仁"，不区分"system / user / tool_result / assistant thinking"各自的保留优先级。

**避雷建议（不要模仿的写法）**
- 不要自己估 token。`tiktoken` 有，或者直接用上一轮 API 返回的 `usage.prompt_tokens` 做决策。
- 不要"压缩=丢"，**保留最近 N 条原始消息**是摘要策略的底线。
- 把"用哪个模型做摘要"通过参数暴露出来，别写死。
- 名字和职责要对齐：`SharedContext` 要么真的支持跨 agent 共享，要么就改名叫 `AgentMessageStore`。

---

### 🔴 Toxic Code #2：`storage/db.py` —— 面向"能跑"而不是面向"能扩"

**证据点**
1. **每次操作一次新 `aiosqlite.connect`**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/storage/db.py" lines="78-94" />）。demo 无妨，稍微多 QPS 就要命。连接池/常驻连接是起码的。
2. **四张表无任何索引**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/storage/db.py" lines="17-74" />）。`event_logs` / `agent_executions` 都按 `session_id` 查，随便跑几十个 session 就会把全表扫一遍。
3. **`event_logs.event_data` 用 TEXT 存 JSON**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/storage/db.py" lines="63-74" />）。SQLite 3.38+ 有 JSON1 支持，至少应该上 `JSON` 约束 / 用 `json_extract` 索引；不然"拉某类事件"必须全表 JSON 解析。
4. **模块级单例 + `init_database()` 重复建表**：文件末尾 `db = Database()`（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/storage/db.py" lines="253-259" />），同时 `lifespan` 里 `await init_database()`（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/main.py" lines="47-50" />）——单例 + 生命周期钩子两套混用，容易写出"第二次引入时又建一次"的 bug。
5. **`log_event` 和 `get_session_events` 没分页**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/storage/db.py" lines="190-209" />），长 session 的事件流会 OOM。

**避雷建议**
- `aiosqlite` 真要扩就绑到 FastAPI `lifespan` 拿一个常驻连接；或干脆换 PostgreSQL + asyncpg。
- 任何"按 X 查"的字段都应该有索引（本项目至少 `sessions.created_at`、`event_logs.session_id`、`agent_executions.session_id`）。
- 大列表接口一律带 `limit/offset` 或 cursor。

---

## 3. 关键设计决策（The Why）

### 决策 #1：为什么 `Orchestrator` 要缩水到 "只负责驱动 Runner"？

**背景**
之前 `Orchestrator.run` 里硬编码了 `decompose → gather → merge`，再加一个 "critic 反思" 这种工作流就要把主干撕开重写一遍。

**决策**
把编排搬到 `core/graph.py` + `core/templates.py`，`Orchestrator` 只剩三件事：
**挑模板 → `GraphRunner.run` → 发 `session_done` 收尾**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/orchestrator.py" lines="74-132" />）。

**Why**
- 新增工作流 = 新增一个 `build_xxx_graph(deps)` 工厂，不用动现有代码路径；`classic` 模板原样保留旧行为。
- 前端拿 `GET /api/templates` 就能动态出下拉菜单（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/routes.py" lines="164-167" />）。

**Trade-off / 反例后果**
- 收益：扩展性从 O(编排变更 × 主干文件) 降到 O(1)。
- 代价：多了"节点/边"这一层，刚上手的人要多读一个文件。
- 替代方案：
  - **类 LangGraph 派生（推荐方向）**：允许并行出边、内置持久化 checkpoint。但依赖重、学习曲线陡。
  - **状态机库（`transitions`）**：更轻，但表达"并行 N 个 agent"要自己拼。
- 不这样做的后果：之后每多一种玩法（Tree-of-Thought、ReWOO、Self-Consistency）都要改 `Orchestrator.run`，PR diff 失控。

---

### 决策 #2：为什么 `POST /api/query` 是 202 异步受理？

**背景**
同步阻塞模式下，客户端等到拿到 `session_id` 时，整个会话已经结束——**SSE 订阅时没有事件可订**。

**决策**
`POST /api/query` 立刻生成 `session_id`、在 MQ 里预建重放缓冲、然后 `asyncio.create_task` 跑后台 orchestrator（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/routes.py" lines="123-161" />）；SSE 连接建立时，MQ 会**先把缓冲里已产生的事件复播一遍**，再进实时推送。

**Why**
- SSE 的含义是"实时"，异步后端才有"实时"可言。
- 重放缓冲解决了"HTTP 响应返回 → 建立 SSE 连接"窗口期的事件丢失。
- Task 加入 `_background_tasks` 强引用集合（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/routes.py" lines="20-23" />），避免被 `asyncio` 的弱引用 GC 悄悄回收。

**Trade-off**
- 收益：Dashboard 真正拿到事件流。
- 代价：**没有"我查询失败了"这条同步信号** —— 错误必须通过 SSE 推给客户端；<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/routes.py" lines="103-120" /> 显式把后台异常翻译成 `error + session_done` 事件解决这个。
- 替代方案：返回 `WebSocket` 升级 URL（更现代但客户端实现负担重）；或用 `GET /api/query/{id}/result` 轮询（对客户端友好，但丢了中间事件）。

---

### 决策 #3：为什么用"工厂函数返回 Node"而不是"继承 Node"？

**证据**
见 `planner_node(deps) -> Node`、`executor_node(deps) -> Node` 等（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/templates.py" lines="92-145" />）。

**Why**
- **依赖注入通过闭包**：`deps` 在闭包里被 `_fn` 捕获，调用方只拿到 `NodeFn`，签名始终是 `state → None`。图结构完全不感知 LLM / DB / 工具。
- **纵向换实现很廉价**：想换个评估器就换 `critic_node` 工厂，图结构不变。

**Trade-off**
- 收益：节点是"值对象"（dataclass），好比较、好序列化。
- 代价：生命周期钩子（"节点进入/离开时跑点啥"）没法用 OO 覆写，只能靠 Runner 统一发事件。好处是**行为可观测性集中**；坏处是**节点不能抢断"自己被再次踩到"这种上下文**（靠 `state.extra["executor_round"]` 手工维护）。
- 替代方案：`class Node(ABC)` + `async def run()` 继承体系——表面面向对象，但 DAG 图对"节点运行时行为"的唯一需求就是"吃 state 输出副作用"，用函数更直接。

---

### 决策 #4：为什么 `critic_loop` 选择"真有环"而不是"预先展开重跑的 N 份子图"？

**Why**
- 预先展开需要编译期知道"要重跑几次"，而 critic 的判定是运行时的；展开派只能退化成"最多展开 max_retries 次的串行 else-else-else"。
- 真环 + `max_steps` 保险丝（`GraphRunner` 里），既表达力强又不可能无限循环。
- `extra_flag("critic_passed", True)` 让"通过就结束"变成**数据驱动的条件边**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/templates.py" lines="382-388" />）。

**Trade-off**
- 收益：写法对称、支持任意闭环。
- 代价：图不再是 DAG，类型名和行业习惯（"DAG 编排"）严格对不上——但大多数"DAG 编排"框架其实都允许回路，属于历史遗留叫法。
- 反例：如果坚持 DAG，则"critic 失败"只能让整个 session 结束，重试得由前端/上层触发——体验差。

---

## 4. 标准起手式（Best Practices Playbook）

### 4.1 异常与错误处理

**本项目的推荐模板**
1. **业务函数就地捕获、转成领域错误/领域返回**——见 `ReactAgent.run` 的 try/except（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/agent.py" lines="203-209" />）：异常被翻译成"发 `ERROR` 事件 + `final_answer = "执行出错: ..."`"。
2. **并行任务必须包到"无抛"的 runner 里**——见修好的 `run_one`（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/templates.py" lines="179-213" />）：整个函数体包一层 try，一定返回 `AgentResult`（成功或 status=error）。`asyncio.gather(return_exceptions=False)` 才不会把其他兄弟 cancel 掉。
3. **后台任务必须发"终态事件"**——`_run_query_background` 的兜底（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/routes.py" lines="103-120" />）：不管出什么事，最后一定发 `session_done`，否则 Dashboard 永远停在"处理中"。
4. **SDK 层按错误码分档重试**——`DeepSeekClient.call` 区分 `RateLimitError` / `APIError(5xx)` / 其他（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/llm.py" lines="113-131" />）。

**反模式**
- 并行任务里只 try 其中一次 I/O，另一次 I/O 抛异常 → 整批协程被 cancel（**就是 PR #4 Devin Review 找出来的那条 bug**）。
- 捕获后静默吞掉（`except Exception: pass`）。即便你真的不想让它抛，也至少 `logger.exception(...)`。
- 在异步热路径 await 日志 I/O、数据库落盘。

**伪代码骨架**
```python
# ✅ 并行 agent 的正确姿势
async def run_one(task) -> AgentResult:
    try:
        ... init
        await db.create_agent_execution(...)
        result = await agent.run(task)
        await db.update_agent_execution(...)
        return result
    except Exception as e:
        logger.exception("agent crashed")
        return AgentResult(..., status="error", final_answer=f"失败: {e}")

results = await asyncio.gather(*(run_one(t) for t in tasks))  # 外层安心

# ❌ 反模式：I/O 散落在 try 之外
async def run_one_bad(task):
    agent = ReactAgent(...)
    await db.create(...)          # <-- 抛了就 cancel 全部
    try:
        result = await agent.run(task)  # 只保护这里没用
    except ...
```

---

### 4.2 数据校验

**本项目的推荐模板**
1. **HTTP 边界用 `pydantic.BaseModel`**——`QueryRequest`（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/routes.py" lines="27-34" />）：类型错/缺字段在进路由函数前就被 FastAPI 拒绝，返回 422。
2. **LLM 给你的 JSON 必须剥 code fence + 容错降级**——`_strip_code_fence` + `json.loads` + fallback（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/templates.py" lines="67-72" /> + <ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/templates.py" lines="120-139" />）：prompt 里要求"纯 JSON"，但 LLM 仍然会用 ` ```json ... ``` ` 包起来，必须剥壳；解析失败 → 降级成 `[query]` 单元素数组，**不让 LLM 把服务搞挂**。
3. **工具参数声明 JSON Schema**——`BaseTool.input_schema` 直接喂给 OpenAI `tools`（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/tools/base.py" lines="17-26" />）。function calling 这层由 LLM 按 schema 构造参数，工具 runner 只要处理"名字对不上"这一种错。
4. **入参边界值夹逼**——`limit = max(1, min(limit, 200))`（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/routes.py" lines="171-175" />）：防止超大 limit 吃爆 DB。

**反模式**
- 只在业务里校验：`if not query: raise`——等到查到业务中间才发现已经晚了。
- `LLM 返回 JSON` 不做 try/except，直接 `json.loads(resp.content)` 就用——生产环境里 5% 的失败率。
- 工具参数不走 schema，让 LLM 自由填字典——各种"把 int 写成 str" 的典型错。

**伪代码骨架**
```python
class QueryRequest(BaseModel):
    query: str
    num_agents: Optional[int] = 3      # FastAPI 自动 422 坏 payload
    template: Optional[str] = None
    max_retries: Optional[int] = 1

# LLM 输出校验：剥壳 → 解析 → 容错兜底
content = _strip_code_fence(response.content)
try:
    tasks = json.loads(content)
    assert isinstance(tasks, list) and tasks
except (json.JSONDecodeError, AssertionError):
    logger.warning("planner fallback")
    tasks = [state.query]    # 降级永远不让流程死
```

---

### 4.3 异步与并发

**本项目的推荐模板**
1. **耗时任务 → 后台 task + 强引用**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/routes.py" lines="144-157" /> / <ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/message_queue.py" lines="62-66" />）：`create_task` + `set.add` + `task.add_done_callback(set.discard)` 三件套是 Python 文档原文推荐的模式。
2. **CPU / I/O 边界清晰**：MQ 的 publish 是"推队列 + 丢副任务"，绝不 await DB（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/message_queue.py" lines="95-103" />）。
3. **并行必须保证元素级异常隔离**：见 4.1 的 `run_one`。
4. **带生命周期的资源绑 lifespan**：MQ 的持久化 handler 在 `lifespan` 里 attach、退出时 detach（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/main.py" lines="42-54" />）。
5. **SSE 心跳 + 终态关闭**：`/sse/stream/{session_id}` timeout 1s 发心跳，收到 `session_done` drain 0.5s 后主动 break（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/api/sse.py" lines="37-64" />）——避免连接无限 hang。

**反模式**
- `create_task(coro)` 不留引用 → 任务中途被 GC。
- 热路径里 `await db.log_event(...)` → MQ 吞吐跟着 DB 抖。
- `gather(..., return_exceptions=False)` 但内部协程会抛异常而没兜底 → 全批被 cancel。
- SSE 不检查 `request.is_disconnected()`，客户端关浏览器后任务还在跑。

**伪代码骨架**
```python
_bg: set[asyncio.Task] = set()  # 模块级强引用集合

def fire_and_forget(coro, *, name: str):
    t = asyncio.create_task(coro, name=name)
    _bg.add(t)
    t.add_done_callback(_bg.discard)
    return t

# 用法
fire_and_forget(run_query_background(...), name=f"query:{sid}")
```

---

## 5. 抽象复用案例：`MessageQueue` —— 从"事件推送"演进成"事件基础设施"

选这块逆向讲，是因为它在仓库里被**三个以上上游复用**（SSE、持久化、全局监控、handler 钩子），是本项目里泛化程度最高的组件。

### 5.1 原始痛点（抽象前的世界）

早期版本里，`agent.py` 要同时做三件事：
1. 把事件 log 出来给开发者看
2. 通过某种方式推给前端 Dashboard
3. 写库方便后续查询

如果每个需求分别写一份代码，就会散落：
- `agent.py` 里到处 `print(...)` + `db.log_event(...)` + `await sse_push(...)`
- 增加一个订阅者（比如"接入 Sentry"）要动 N 个文件
- 单测极难（要 mock 三种副作用）

### 5.2 抽象过程

抽出的"共性"：
- **唯一发布入口 `publish(event_type, agent_id, data)`**：业务只管发，不管谁在听
- **订阅者 fan-out**：按事件类型 / 按 session / 全局，三种维度通吃
- **持久化 handler 作为"插件"**：MQ 本体不依赖 DB，`main.py` lifespan 里 attach 一个 handler 就把"所有事件落库"拧进来

保留的"差异"：
- 订阅者可以选自己关心的维度（事件类型 vs session_id）
- 持久化 handler 可插拔（测试环境 None）
- 同步 handler（`event_handlers`）和异步 handler 都支持（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/message_queue.py" lines="105-114" />）

### 5.3 扩展性评估

想加新能力，不用动 MQ 本身：
- **接 Sentry**：`mq.on("error", sentry_reporter)` 注册一个同步 handler。
- **接 OTel**：`set_persistence_handler(otel_span_writer)`。
- **接 Redis Pub/Sub**：在 `publish` 末尾 fire-and-forget 一个"推 Redis"的 handler；切多进程时订阅端 fan-in。

已经验证的扩展：**PR #3 的"事件写 event_logs"完全不用改 MQ 一行**（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/main.py" lines="21-50" />）。这是抽象做对的直接证据。

### 5.4 如果重做，你会如何优化

现实、不理想化的 3 条：
1. **事件 schema 强化**：现在 `data: Dict[str, Any]` 太自由，任何人都可以塞任何字段。把每个 `event_type` 绑到一个 pydantic model 上（或者用 `TypedDict`），至少前端侧面代码补全友好很多。
2. **backpressure**：目前订阅队列 `maxsize=1000`，满了就 `warning`——慢消费者会丢事件。更稳是换成"慢消费者断连 + 客户端用 `Last-Event-ID` 从 `event_logs` 里补齐"。
3. **进程内 → 分布式**：加一层 adapter，本地用 asyncio.Queue，生产换成 Redis Streams / NATS JetStream。接口不变，部署形态可切换。

---

## 6. 练习任务：接入真实 Web Search 工具

### 6.1 任务背景
当前 `SearchTool` 是 mock 的（<ref_file file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/tools/search.py" />），agent 永远只能拿到假数据，所以整个系统在"真实世界查询"上其实跑不通。这也是 PR 列表里从没碰过的区域——正好适合练手。

### 6.2 需求
新增一个真实的 `WebSearchTool`，接入 [Tavily](https://tavily.com) 或 [Serper](https://serper.dev) 二选一（两家都有免费额度）。要求：

1. **位置**：新建 `core/tools/web_search.py`，不要动老的 mock `SearchTool`；老的改名 `MockSearchTool`，注释说明"仅作 demo 兜底"。
2. **API key 走环境变量**：`TAVILY_API_KEY` 或 `SERPER_API_KEY`，**没配就让这个 Tool 在 `api/routes.py:_build_default_tools` 里被跳过**，不要在启动时就 raise。
3. **网络层用 `httpx.AsyncClient`**（requirements.txt 已有 httpx 传递依赖；如果没有就加）。超时 10s，不要用默认无限超时。
4. **错误容忍**：请求失败时返回字符串 `f"搜索失败: {reason}"`，不抛异常（和 `ReactAgent._execute_tool` 现有风格一致）。
5. **结果压缩**：接入时注意 `ReactAgent._execute_tool` 的超 20k 字符自动摘要路径（<ref_snippet file="/home/ubuntu/repos/react_agent_try/multi_agent_react/core/agent.py" lines="259-263" />），所以工具直接返回完整原文就行，不要在工具内部再做二次摘要。
6. **`input_schema` 用 JSON Schema**：至少包含 `query: string`、可选 `max_results: integer (default 5)`。

### 6.3 验收标准

**功能**
- [ ] 设置 `TAVILY_API_KEY` 后，agent 真的能拿到带 URL 和 snippet 的搜索结果
- [ ] **不设置** API key 时，`/api/query` 仍然正常工作，只是工具列表里没有 `web_search`
- [ ] 空 `query` 或网络错误返回 `"搜索失败: ..."` 字符串，`/api/query` 不 500

**代码质量**
- [ ] 新文件 `core/tools/web_search.py` ≤ 120 行
- [ ] 和 `calculator.py` 同风格：类注释说明"为什么这样写"，description 指导 LLM
- [ ] **无全局可变状态**（httpx client 要么在 `__init__` 里建且有 `aclose`，要么每次 `async with httpx.AsyncClient()`；前者更高效）
- [ ] 任何 `kwargs` 都要校验，不要直接 `kwargs["query"]` 硬取
- [ ] 工具描述里明示"这是实时 Web 搜索，结果可能过时 1~60 秒"

### 6.4 改动范围（避免 scope 爆炸）
| 文件 | 是否允许动 | 说明 |
| --- | --- | --- |
| `core/tools/web_search.py` | 新建 | 主要实现 |
| `core/tools/search.py` | 只能改类名 + 加注释 | 让路给真工具，保留兜底 |
| `api/routes.py` | `_build_default_tools` 里条件添加 | 可选判断 env 决定是否注册 |
| `requirements.txt` | 补 `httpx`（如果没有） | 仅此 |
| 其它文件 | **不允许动** | — |

### 6.5 提交格式建议
- 分支名：`devin/TIMESTAMP-real-web-search`
- Commit message 结构：
  ```
  feat(tools): add TavilyWebSearchTool as real search backend

  - 新建 core/tools/web_search.py，通过 TAVILY_API_KEY 走真实 API
  - 老 SearchTool 改名 MockSearchTool 作为兜底
  - 环境变量缺失时自动跳过注册，不阻塞启动

  Trade-off: 引入 httpx 同步超时；免费额度 1000 次/月够 demo
  ```
- PR 模板里回答：
  1. 测试步骤（最好 mock 一份 httpx response，写个 `tests/test_web_search.py`；
     本仓库没单测体系，先凑合"带 key 手动跑 curl 截图 + 不带 key 启动日志"也行）
  2. 为什么没改 `search.py`（答：限制改动范围）
  3. API 失败时的降级行为证明

---

## 7. 你提交后我会看什么（Code Review 视角）

这是给自己对照用的 checklist，提交前自查一遍能干掉 70% 的典型问题：

| 维度 | 会被怎么批 |
| --- | --- |
| **结构** | 工具被注册进 `_build_default_tools` 的方式是否和其他工具一致？是否引入了"工具知道 orchestrator 存在"这种逆向依赖？ |
| **命名** | `WebSearchTool.name` 是 `"web_search"` 还是 `"search"`？如果和老 mock 撞名，LLM 会调错。 |
| **边界** | `query` 为空字符串、超过 1000 字符、包含换行符时会发生什么？`max_results` 给 0 / -1 / 999 呢？ |
| **异常** | `httpx.TimeoutException` / `httpx.HTTPStatusError` / 网络 DNS 失败各自走哪条分支？全部落回"搜索失败"字符串？ |
| **性能** | 每次 `run()` 新建一个 `AsyncClient`（会 TCP 重连）还是复用？如果复用，进程退出时是否 `aclose`？ |
| **可测试** | 能不能不真的发网络请求就测出来？（`httpx.MockTransport` 是答案） |
| **Nit** | description 里有没有教 LLM "如果只是算术题不要用这个工具"？否则 LLM 会滥用网络搜索加钱。 |

**问题等级我会这么标：**
- `Critical`：影响正确性（API key 泄漏到日志、异常冒泡把 session 搞死）
- `Major`：影响扩展性（工具注册强依赖 env）
- `Minor`：影响可读性（命名、无关的一行 import）
- `Nit`：审美建议（文档 Markdown 格式）

每条都会带**修改建议 + 原因**，不留"这里不好"这种空话。

---

## 附：下一步提升计划（个人层面的 3 条）

1. **Token 级别的精确化**：在本仓库里把 `len(content)//4` 换成 `tiktoken`，顺便给压缩策略加 sliding window。这条的收益是**可测量**的：同样的"2 的 10 次方"查询，token 会再下降一档。
2. **给 DAG 加一个 checkpoint**：`GraphState` 序列化到 sqlite，跑到一半 kill 服务也能从上一个 `node_done` 恢复。是 LangGraph 的看家本领，自己动手一次能彻底理解"为什么要 checkpoint"。
3. **把本项目跑通 Eval Harness**：维护 50 个 Q/A，PR 里自动跑 `critic_loop` vs `classic`，打分对比。有 eval 的 agent 项目和没 eval 的是两个物种。

---

**结语**：本仓库是个"教学级做得很好、生产级还差很多"的项目——结构清爽、概念明确、可观测性做足了一半。如果你是来学"多 Agent 框架怎么拆"的，`core/graph.py`、`core/message_queue.py`、`core/llm.py` 这三块非常值得**逐行抄一遍**。至于 `core/context.py` 和 `storage/db.py` 里看到的写法，**学会"它为什么不该被学"比学它本身更值钱**。
