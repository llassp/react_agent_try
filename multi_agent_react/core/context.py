import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

from core.llm import DeepSeekClient, TokenUsage


MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", "128000"))
COMPRESS_THRESHOLD = float(os.environ.get("CONTEXT_COMPRESS_THRESHOLD", "0.7"))
TOOL_RESULT_MAX_CHARS = int(os.environ.get("TOOL_RESULT_MAX_CHARS", "20000"))


@dataclass
class ContextMessage:
    role: str
    content: str
    token_count: int = 0
    metadata: Dict[str, Any] = None


class SharedContext:
    """共享上下文管理器，支持多 Agent 间的上下文共享和压缩"""
    
    def __init__(self):
        self._contexts: Dict[str, List[Dict]] = {}  # agent_id -> messages
        self._shared_memory: Dict[str, Any] = {}  # 共享内存
    
    def create(self, agent_id: str, system_prompt: Optional[str] = None):
        """为 Agent 创建上下文"""
        self._contexts[agent_id] = []
        if system_prompt:
            self._contexts[agent_id].append({
                "role": "system",
                "content": system_prompt,
                "_token_count": len(system_prompt) // 4
            })
    
    def get(self, agent_id: str) -> List[Dict]:
        """获取 Agent 的上下文"""
        return self._contexts.get(agent_id, [])
    
    def add_message(self, agent_id: str, role: str, content: str, metadata: Dict = None):
        """添加消息到上下文"""
        if agent_id not in self._contexts:
            self._contexts[agent_id] = []
        
        message = {
            "role": role,
            "content": content,
            "_token_count": len(content or "") // 4,
        }
        if metadata:
            message["_metadata"] = metadata
        
        self._contexts[agent_id].append(message)
    
    def add_assistant_tool_calls(
        self,
        agent_id: str,
        tool_calls: List[Dict[str, Any]],
        content: str = "",
    ):
        """添加带 tool_calls 的 assistant 消息（OpenAI function calling 规范）。

        ``tool_calls`` 必须是 OpenAI 兼容接口期望的原始结构，例如::

            [{"id": "call_x", "type": "function",
              "function": {"name": "calc", "arguments": "{...}"}}]

        调用方随后必须用 :meth:`add_tool_result` 为 **每一个** id 回填一条 role=tool
        的结果消息，否则下一次 LLM 调用会因 tool_call_id 不匹配而报 400。
        """
        if agent_id not in self._contexts:
            self._contexts[agent_id] = []

        content_for_tokens = content or ""
        # tool_calls 本身也占 token；粗略估计一下以便触发压缩
        serialized_calls = ""
        try:
            import json as _json
            serialized_calls = _json.dumps(tool_calls, ensure_ascii=False)
        except Exception:
            serialized_calls = str(tool_calls)

        self._contexts[agent_id].append({
            "role": "assistant",
            "content": content_for_tokens,
            "tool_calls": tool_calls,
            "_token_count": (len(content_for_tokens) + len(serialized_calls)) // 4,
        })

    def add_tool_result(self, agent_id: str, tool_call_id: str, content: str):
        """添加工具调用结果（role=tool，必须携带 tool_call_id）。"""
        if agent_id not in self._contexts:
            self._contexts[agent_id] = []

        self._contexts[agent_id].append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
            "_token_count": len(content or "") // 4,
        })
    
    def get_messages_for_llm(self, agent_id: str) -> List[Dict[str, Any]]:
        """获取用于 LLM 调用的消息列表（去除内部字段，保留 tool_calls / tool_call_id）"""
        messages = self._contexts.get(agent_id, [])
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "tool":
                tool_call_id = m.get("tool_call_id") or m.get("_metadata", {}).get("tool_call_id")
                if not tool_call_id:
                    # 缺 tool_call_id 的 tool 消息一定会让 OpenAI 兼容接口报错，
                    # 宁可丢弃也不要整条请求失败。
                    logger.warning("Dropping tool message without tool_call_id")
                    continue
                out.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": m["content"],
                })
            elif role == "assistant" and m.get("tool_calls"):
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": m["tool_calls"],
                }
                # OpenAI 规范允许 content 为 null；部分模型更偏好空字符串
                entry["content"] = m.get("content") or ""
                out.append(entry)
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out
    
    def get_token_count(self, agent_id: str) -> int:
        """计算上下文的 token 数量"""
        messages = self._contexts.get(agent_id, [])
        return sum(msg.get("_token_count", len(msg.get("content") or "") // 4) for msg in messages)
    
    def get_usage_ratio(self, agent_id: str) -> float:
        """获取上下文使用率"""
        return self.get_token_count(agent_id) / MAX_CONTEXT_TOKENS
    
    async def maybe_compress(self, agent_id: str, llm: DeepSeekClient) -> bool:
        """检查并压缩上下文（如果需要）"""
        ratio = self.get_usage_ratio(agent_id)
        
        if ratio < COMPRESS_THRESHOLD:
            return False
        
        logger.info(f"Context for {agent_id} at {ratio:.1%}, triggering compression")
        
        # 保留 system prompt
        messages = self._contexts.get(agent_id, [])
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]
        
        if not other_msgs:
            return False
        
        # 格式化历史记录
        history_text = self._format_messages(other_msgs)
        
        # 调用 LLM 进行摘要
        summary_response = await llm.call(
            messages=[{
                "role": "user",
                "content": f"请将以下对话历史压缩为简洁摘要（不超过500字）：\n\n{history_text}"
            }],
            model="deepseek-chat"
        )
        
        summary = summary_response.content
        
        # 替换为摘要
        self._contexts[agent_id] = system_msgs + [{
            "role": "assistant",
            "content": f"[历史摘要] {summary}",
            "_token_count": len(summary) // 4,
            "_is_summary": True
        }]
        
        logger.info(f"Context compressed for {agent_id}, new ratio: {self.get_usage_ratio(agent_id):.1%}")
        return True
    
    async def compress_tool_result(self, result: str, llm: DeepSeekClient) -> str:
        """压缩过长的工具结果"""
        if len(result) <= TOOL_RESULT_MAX_CHARS:
            return result
        
        # 截断并摘要
        truncated = result[:TOOL_RESULT_MAX_CHARS]
        
        summary_response = await llm.call(
            messages=[{
                "role": "user",
                "content": f"摘要以下内容（100字以内）：\n{truncated}"
            }],
            model="deepseek-chat"
        )
        
        summary = summary_response.content
        return f"[已压缩，原长{len(result)}字] {summary}"
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """格式化消息列表为文本"""
        lines = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                lines.append(f"用户: {content}")
            elif role == "assistant":
                lines.append(f"助手: {content}")
            elif role == "tool":
                lines.append(f"工具结果: {content[:200]}...")
        return "\n".join(lines)
    
    def set_shared(self, key: str, value: Any):
        """设置共享内存"""
        self._shared_memory[key] = value
    
    def get_shared(self, key: str, default: Any = None) -> Any:
        """获取共享内存"""
        return self._shared_memory.get(key, default)
    
    def clear(self, agent_id: str):
        """清除指定 Agent 的上下文"""
        if agent_id in self._contexts:
            del self._contexts[agent_id]
    
    def clear_all(self):
        """清除所有上下文"""
        self._contexts.clear()
        self._shared_memory.clear()
