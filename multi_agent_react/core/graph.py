"""DAG 编排内核。

之前的 Orchestrator 把 ``decompose → gather → merge`` 这三段写死在 ``run`` 里，
新增一个"带 critic 反思重跑"的工作流都没地方插。这个模块把编排重新切成
"节点 + 有向边"：

- ``GraphState`` 是节点之间共享的可变载荷（query / tasks / agent_results / 等）。
- ``Node`` 是一个纯异步单元：接一个 ``GraphState``，返回 ``None``，副作用写进 state。
- ``Edge`` 描述"A 跑完后，在什么条件下跳到 B"。条件是一个同步 callable，看 state 判断。
- ``Graph`` 把以上组装成一张图，``GraphRunner`` 驱动执行。

入口节点跑完后枚举它的出边，第一条条件为真的边决定下一个节点。``None`` 意味着
终止。同一节点可以被多次踩到（比如 critic 反思时跳回 executor），所以这是真正的
**有向图**，不是 DAG，但沿用行业里"DAG 编排"的叫法。

设计约束：
1. 节点和边不能引用具体的 LLM/工具实现，只接收 state 与在 state 里带着的外部依赖，
   这样同一张图能用在不同模型/工具组合下。
2. 进出节点都会发布 ``node_start`` / ``node_done`` 事件，前端可以看"当前走到哪了"。
3. 异常不吞掉：节点抛异常 → ``node_error`` 事件 + 抛给 Runner，由 Runner 决定转成
   session 级 error（默认行为）。
4. 防止坏图死循环：``GraphRunner.run`` 有 ``max_steps`` 上限，默认 64 个节点步；
   超限抛 ``GraphRunError``。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from loguru import logger

from core.message_queue import MessageQueue


class GraphRunError(RuntimeError):
    """图执行失败：超步、找不到节点等。"""


@dataclass
class GraphState:
    """节点之间共享的可变状态。

    除了 query / tasks / agent_results 这些业务字段外，还有一个 ``extra`` 字典，
    给模板作者随便放中间变量用（比如 critic 的 ``retry_count``）。把 extra 单独拆出
    是为了让核心字段在 IDE 里有类型提示。
    """

    session_id: str
    query: str
    tasks: List[str] = field(default_factory=list)
    agent_results: List[Any] = field(default_factory=list)  # AgentResult 的 list
    final_answer: Optional[str] = None
    # critic 等节点可以塞反馈／重试计数
    extra: Dict[str, Any] = field(default_factory=dict)


NodeFn = Callable[[GraphState], Awaitable[None]]
EdgeCondition = Callable[[GraphState], bool]


@dataclass
class Edge:
    to: str
    # 条件为真才走这条边。默认恒真；由节点的出边顺序决定优先级——越先加进来越先判断。
    condition: EdgeCondition = field(default=lambda _state: True)
    # 给前端/debug 一个人类可读的名字（比如 "critic_says_bad"）
    label: str = ""


@dataclass
class Node:
    name: str
    fn: NodeFn
    # 额外元数据，前端展示"当前节点"用
    description: str = ""


class Graph:
    """节点 + 出边的有向图。

    使用方式：
        g = Graph(entry="planner")
        g.add_node(Node("planner", planner_fn))
        g.add_node(Node("executor", executor_fn))
        g.add_edge("planner", "executor")
    """

    def __init__(self, entry: str):
        self.entry = entry
        self._nodes: Dict[str, Node] = {}
        self._out_edges: Dict[str, List[Edge]] = {}

    def add_node(self, node: Node) -> "Graph":
        if node.name in self._nodes:
            raise ValueError(f"Duplicate node name: {node.name!r}")
        self._nodes[node.name] = node
        self._out_edges.setdefault(node.name, [])
        return self

    def add_edge(
        self,
        src: str,
        dst: Optional[str],
        condition: Optional[EdgeCondition] = None,
        label: str = "",
    ) -> "Graph":
        """在 ``src`` 后添加一条到 ``dst`` 的条件边。

        ``dst=None`` 表示"如果条件满足，就在这里终止图的执行"，配合 ``condition`` 用
        来实现"若 critic 通过则结束"。
        """
        if src not in self._nodes:
            raise ValueError(f"Unknown src node: {src!r}")
        # dst 可能是 None 或尚未注册的节点；后者在 compile 时再检查
        edge = Edge(to=dst or "__end__", condition=condition or (lambda _s: True), label=label)
        self._out_edges.setdefault(src, []).append(edge)
        return self

    def compile(self) -> None:
        """简单的一致性检查：所有边的目标要么是注册过的节点，要么是 ``__end__``。"""
        if self.entry not in self._nodes:
            raise ValueError(f"Entry node {self.entry!r} not registered")
        for src, edges in self._out_edges.items():
            for edge in edges:
                if edge.to != "__end__" and edge.to not in self._nodes:
                    raise ValueError(
                        f"Edge {src} -> {edge.to!r} points to unknown node"
                    )

    def get_node(self, name: str) -> Node:
        if name not in self._nodes:
            raise GraphRunError(f"Node {name!r} not found in graph")
        return self._nodes[name]

    def out_edges(self, name: str) -> List[Edge]:
        return self._out_edges.get(name, [])

    def describe(self) -> Dict[str, Any]:
        """序列化图结构，方便前端画流程图或记日志。"""
        return {
            "entry": self.entry,
            "nodes": [
                {"name": n.name, "description": n.description}
                for n in self._nodes.values()
            ],
            "edges": [
                {"from": src, "to": e.to, "label": e.label}
                for src, edges in self._out_edges.items()
                for e in edges
            ],
        }


class GraphRunner:
    """驱动一张 Graph 一步步跑到终点，并把节点级事件推到 MessageQueue。

    可以理解成"单线程的 while 循环 + 出边选择器"，刻意不支持并行边——多 agent 并行
    放在节点内部（比如 ExecutorNode 里用 ``asyncio.gather``），这样 state 的可变性
    好推理，不容易写出竞争。
    """

    def __init__(
        self,
        graph: Graph,
        mq: MessageQueue,
        max_steps: int = 64,
    ):
        graph.compile()
        self.graph = graph
        self.mq = mq
        self.max_steps = max_steps

    async def run(self, state: GraphState) -> GraphState:
        current: Optional[str] = self.graph.entry
        steps = 0

        await self.mq.publish("graph_start", None, {
            "session_id": state.session_id,
            "graph": self.graph.describe(),
        })

        while current and current != "__end__":
            if steps >= self.max_steps:
                raise GraphRunError(
                    f"Graph exceeded max_steps={self.max_steps} "
                    f"(last node: {current}); likely a bad loop."
                )
            steps += 1

            node = self.graph.get_node(current)
            await self.mq.publish("node_start", None, {
                "session_id": state.session_id,
                "node": node.name,
                "description": node.description,
                "step": steps,
            })
            logger.info(f"[graph] step={steps} enter node={node.name}")

            try:
                await node.fn(state)
            except Exception as e:
                logger.exception(f"[graph] node {node.name} failed: {e}")
                await self.mq.publish("node_error", None, {
                    "session_id": state.session_id,
                    "node": node.name,
                    "error": str(e),
                })
                raise

            await self.mq.publish("node_done", None, {
                "session_id": state.session_id,
                "node": node.name,
                "step": steps,
            })

            # 挑第一条条件为真的出边
            next_node: Optional[str] = None
            chosen_label: str = ""
            for edge in self.graph.out_edges(node.name):
                try:
                    ok = bool(edge.condition(state))
                except Exception as e:
                    logger.warning(
                        f"[graph] edge {node.name} -> {edge.to} condition raised: {e}"
                    )
                    ok = False
                if ok:
                    next_node = edge.to
                    chosen_label = edge.label
                    break

            if next_node and next_node != "__end__":
                logger.info(
                    f"[graph] {node.name} -> {next_node}"
                    + (f" via '{chosen_label}'" if chosen_label else "")
                )

            current = next_node

        await self.mq.publish("graph_done", None, {
            "session_id": state.session_id,
            "steps": steps,
        })
        return state


def always(_state: GraphState) -> bool:
    """默认条件：恒真。"""
    return True


def extra_flag(key: str, expected: Any = True) -> EdgeCondition:
    """从 ``state.extra[key]`` 读值，等于 ``expected`` 就走这条边。

    ``extra_flag("critic_passed", True)`` 用来表达"critic 通过 → 去 merger"。
    """

    def _cond(state: GraphState) -> bool:
        return state.extra.get(key) == expected

    return _cond
