from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from core.llm import DeepSeekClient, TokenUsage, LLMResponse
from core.context import SharedContext
from core.message_queue import MessageQueue, EventType
from core.tools.base import BaseTool


@dataclass
class AgentResult:
    agent_id: str
    task: str
    final_answer: str
    trajectory: List[Dict[str, Any]]
    total_tokens: TokenUsage
    status: str = "success"


@dataclass
class Step:
    iteration: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    reasoning: Optional[str] = None


class ReactAgent:
    """ReAct Agent，支持 function calling 模式"""
    
    def __init__(
        self,
        agent_id: str,
        tools: List[BaseTool],
        context: SharedContext,
        llm: DeepSeekClient,
        message_queue: MessageQueue,
        max_iterations: int = 10,
        use_thinking: bool = False,
        session_id: Optional[str] = None
    ):
        self.agent_id = agent_id
        self.tools = {tool.name: tool for tool in tools}
        self.tools_list = tools
        self.context = context
        self.llm = llm
        self.mq = message_queue
        self.max_iterations = max_iterations
        self.use_thinking = use_thinking
        self.session_id = session_id
        
        self.trajectory: List[Step] = []
        self.total_tokens = TokenUsage()
    
    async def run(self, task: str) -> AgentResult:
        """执行 ReAct 循环"""
        logger.info(f"Agent {self.agent_id} starting task: {task}")
        
        # 初始化上下文
        system_prompt = self._build_system_prompt()
        self.context.create(self.agent_id, system_prompt)
        
        # 添加用户任务
        self.context.add_message(self.agent_id, "user", task)
        
        # 发布开始事件
        await self.mq.publish(EventType.AGENT_START, self.agent_id, {
            "session_id": self.session_id,
            "task": task,
            "max_iterations": self.max_iterations
        })
        
        final_answer = None
        
        try:
            for iteration in range(self.max_iterations):
                # 检查是否需要压缩上下文
                compressed = await self.context.maybe_compress(self.agent_id, self.llm)
                if compressed:
                    await self.mq.publish(EventType.CONTEXT_COMPRESSED, self.agent_id, {
                        "session_id": self.session_id,
                        "new_ratio": self.context.get_usage_ratio(self.agent_id)
                    })
                
                # 调用 LLM
                messages = self.context.get_messages_for_llm(self.agent_id)
                tools = [tool.to_openai_tool() for tool in self.tools_list] if self.tools_list else None
                
                response = await self.llm.call(
                    messages=messages,
                    tools=tools,
                    use_thinking=self.use_thinking
                )
                
                # 更新 token 统计
                if response.usage:
                    self.total_tokens.prompt_tokens += response.usage.prompt_tokens
                    self.total_tokens.completion_tokens += response.usage.completion_tokens
                    self.total_tokens.total_tokens += response.usage.total_tokens
                
                # 处理 reasoning_content（思考模式）
                if response.reasoning:
                    await self.mq.publish(EventType.THINKING, self.agent_id, {
                        "session_id": self.session_id,
                        "text": response.reasoning
                    })
                
                # 发布本轮 token 更新（无论是否有工具调用都应该发一次）
                await self.mq.publish(EventType.TOKEN_UPDATE, self.agent_id, {
                    "session_id": self.session_id,
                    "iteration": iteration,
                    "prompt": response.usage.prompt_tokens if response.usage else 0,
                    "completion": response.usage.completion_tokens if response.usage else 0,
                    "cumulative": self.total_tokens.total_tokens
                })
                
                tool_calls = response.tool_calls or []
                
                if tool_calls:
                    # 先把带 tool_calls 的 assistant 消息原样回放进上下文，
                    # 确保后续 role=tool 的 tool_call_id 能与 assistant.tool_calls[*].id 对齐。
                    raw_tool_calls = [tc["raw"] for tc in tool_calls]
                    self.context.add_assistant_tool_calls(
                        agent_id=self.agent_id,
                        tool_calls=raw_tool_calls,
                        content=response.content or "",
                    )
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call["name"]
                        tool_input = tool_call["arguments"]
                        tool_id = tool_call["id"]
                        
                        thought = (
                            response.content.strip()
                            if response.content and response.content.strip()
                            else f"调用 {tool_name} 工具"
                        )
                        
                        step = Step(
                            iteration=iteration,
                            thought=thought,
                            action=tool_name,
                            action_input=tool_input,
                            token_usage=response.usage,
                            reasoning=response.reasoning
                        )
                        self.trajectory.append(step)
                        
                        # 发布 action 事件
                        await self.mq.publish(EventType.ACTION, self.agent_id, {
                            "session_id": self.session_id,
                            "iteration": iteration,
                            "tool": tool_name,
                            "input": tool_input,
                            "tool_call_id": tool_id,
                        })
                        
                        # 执行工具
                        observation = await self._execute_tool(tool_name, tool_input)
                        step.observation = observation
                        
                        # 发布 observation 事件
                        await self.mq.publish(EventType.OBSERVATION, self.agent_id, {
                            "session_id": self.session_id,
                            "iteration": iteration,
                            "observation": observation,
                            "tool_call_id": tool_id,
                        })
                        
                        # 回填 tool 结果（role=tool + tool_call_id）
                        self.context.add_tool_result(self.agent_id, tool_id, observation)
                    
                    # 继续下一轮循环，让 LLM 基于工具结果推理
                    continue
                
                # 没有工具调用 → 视为最终答案
                final_answer = (response.content or "").strip() or "(空回答)"
                
                step = Step(
                    iteration=iteration,
                    thought=final_answer,
                    token_usage=response.usage,
                    reasoning=response.reasoning
                )
                self.trajectory.append(step)
                
                # 把最终 assistant 消息写回上下文（方便外部审计）
                self.context.add_message(self.agent_id, "assistant", final_answer)
                
                logger.info(f"Agent {self.agent_id} completed in {iteration + 1} iterations")
                break
            
            else:
                # 达到最大迭代次数
                final_answer = self._generate_final_from_trajectory()
                logger.warning(f"Agent {self.agent_id} reached max iterations")
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error: {e}")
            await self.mq.publish(EventType.ERROR, self.agent_id, {
                "session_id": self.session_id,
                "error": str(e)
            })
            final_answer = f"执行出错: {str(e)}"
        
        # 发布完成事件
        await self.mq.publish(EventType.AGENT_DONE, self.agent_id, {
            "session_id": self.session_id,
            "final_answer": final_answer,
            "total_tokens": {
                "prompt": self.total_tokens.prompt_tokens,
                "completion": self.total_tokens.completion_tokens,
                "total": self.total_tokens.total_tokens
            }
        })
        
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            final_answer=final_answer,
            trajectory=self._trajectory_to_dict(),
            total_tokens=self.total_tokens
        )
    
    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tools_desc = []
        for tool in self.tools_list:
            tools_desc.append(f"- {tool.name}: {tool.description}")
        
        tools_text = "\n".join(tools_desc) if tools_desc else "无可用工具"
        
        return f"""你是一个智能助手，可以使用以下工具来解决问题：

{tools_text}

请按照以下格式思考：
1. 分析问题和当前状态
2. 决定是否需要使用工具
3. 如果需要工具，明确指定工具名称和参数
4. 根据工具结果继续思考或给出最终答案

当你获得足够信息时，直接给出最终答案。"""
    
    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """执行工具"""
        if tool_name not in self.tools:
            return f"错误: 未知工具 '{tool_name}'"
        
        tool = self.tools[tool_name]
        try:
            result = await tool.run(**tool_input)
            
            # 压缩过长的结果
            if len(result) > 20000:
                result = await self.context.compress_tool_result(result, self.llm)
            
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} execution error: {e}")
            return f"工具执行错误: {str(e)}"
    
    def _generate_final_from_trajectory(self) -> str:
        """从轨迹生成最终答案（当达到最大迭代次数时）"""
        if not self.trajectory:
            return "无法生成答案"
        
        # 使用最后一步的思考作为答案
        last_step = self.trajectory[-1]
        return f"{last_step.thought}\n\n（达到最大迭代次数，可能未完成）"
    
    def _trajectory_to_dict(self) -> List[Dict]:
        """将轨迹转换为字典列表"""
        return [
            {
                "iteration": step.iteration,
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
                "token_usage": {
                    "prompt": step.token_usage.prompt_tokens if step.token_usage else 0,
                    "completion": step.token_usage.completion_tokens if step.token_usage else 0,
                },
                "reasoning": step.reasoning
            }
            for step in self.trajectory
        ]
