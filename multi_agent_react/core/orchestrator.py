import os
import json
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from core.llm import DeepSeekClient, TokenUsage
from core.agent import ReactAgent, AgentResult
from core.context import SharedContext
from core.message_queue import MessageQueue, EventType
from core.tools.base import BaseTool
from storage.db import Database


@dataclass
class SessionResult:
    session_id: str
    query: str
    final_answer: str
    agent_results: List[AgentResult]
    total_tokens: TokenUsage


class Orchestrator:
    """多 Agent 调度器，负责任务拆分、并行执行和结果合并"""
    
    def __init__(
        self,
        llm: DeepSeekClient,
        context: SharedContext,
        message_queue: MessageQueue,
        database: Database,
        tools: List[BaseTool],
        num_agents: int = 3,
        max_iterations: int = 10
    ):
        self.llm = llm
        self.context = context
        self.mq = message_queue
        self.db = database
        self.tools = tools
        self.num_agents = num_agents
        self.max_iterations = max_iterations
        
        self.use_thinking = os.environ.get("USE_THINKING_MODE", "false").lower() == "true"
    
    async def run(self, query: str) -> SessionResult:
        """执行完整的多 Agent 工作流"""
        session_id = str(uuid.uuid4())
        logger.info(f"Starting session {session_id} with query: {query}")
        
        # 创建会话记录
        await self.db.create_session(session_id, query)
        
        # 步骤 1: 任务拆分
        tasks = await self._decompose_task(query)
        logger.info(f"Task decomposed into {len(tasks)} subtasks")
        
        # 步骤 2: 并行执行
        agent_results = await self._execute_parallel(session_id, tasks)
        logger.info(f"All agents completed for session {session_id}")
        
        # 步骤 3: 结果合并
        final_answer = await self._merge_results(query, agent_results)
        logger.info(f"Results merged for session {session_id}")
        
        # 更新会话状态
        await self.db.update_session(session_id, final_answer, "completed")
        
        # 计算总 token
        total_tokens = TokenUsage()
        for result in agent_results:
            total_tokens.prompt_tokens += result.total_tokens.prompt_tokens
            total_tokens.completion_tokens += result.total_tokens.completion_tokens
            total_tokens.total_tokens += result.total_tokens.total_tokens
        
        # 发布会话完成事件
        await self.mq.publish(EventType.SESSION_DONE, None, {
            "session_id": session_id,
            "final_answer": final_answer,
            "total_tokens": {
                "prompt": total_tokens.prompt_tokens,
                "completion": total_tokens.completion_tokens,
                "total": total_tokens.total_tokens
            }
        })
        
        return SessionResult(
            session_id=session_id,
            query=query,
            final_answer=final_answer,
            agent_results=agent_results,
            total_tokens=total_tokens
        )
    
    async def _decompose_task(self, query: str) -> List[str]:
        """将任务拆分为子任务"""
        prompt = f"""请将以下用户查询拆分为 {self.num_agents} 个子任务，以便并行处理。

用户查询: {query}

要求:
1. 每个子任务应该是独立的、可并行执行的
2. 子任务应该覆盖原问题的不同方面
3. 返回格式必须是纯 JSON 数组，不要添加 markdown 代码块标记

示例输出:
["子任务1的描述", "子任务2的描述", "子任务3的描述"]"""
        
        try:
            response = await self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                model="deepseek-chat"
            )
            
            content = response.content.strip()
            
            # 尝试解析 JSON
            try:
                tasks = json.loads(content)
                if isinstance(tasks, list) and len(tasks) > 0:
                    # 确保数量匹配
                    if len(tasks) < self.num_agents:
                        # 补充任务
                        while len(tasks) < self.num_agents:
                            tasks.append(f"补充分析: {query}")
                    elif len(tasks) > self.num_agents:
                        # 截断
                        tasks = tasks[:self.num_agents]
                    return tasks
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse task decomposition JSON: {content}")
        
        except Exception as e:
            logger.error(f"Task decomposition error: {e}")
        
        # 降级：平均分配
        return [f"从角度 {i+1} 分析: {query}" for i in range(self.num_agents)]
    
    async def _execute_parallel(self, session_id: str, tasks: List[str]) -> List[AgentResult]:
        """并行执行多个 Agent"""
        
        async def run_agent(agent_id: str, task: str) -> AgentResult:
            agent = ReactAgent(
                agent_id=agent_id,
                tools=self.tools,
                context=self.context,
                llm=self.llm,
                message_queue=self.mq,
                max_iterations=self.max_iterations,
                use_thinking=self.use_thinking,
                session_id=session_id
            )
            
            # 创建执行记录
            await self.db.create_agent_execution(session_id, agent_id, task)
            
            result = await agent.run(task)
            
            # 更新执行记录
            await self.db.update_agent_execution(
                session_id=session_id,
                agent_id=agent_id,
                final_answer=result.final_answer,
                trajectory=result.trajectory,
                status="completed"
            )
            
            return result
        
        # 创建任务
        coroutines = []
        for i, task in enumerate(tasks):
            agent_id = f"agent-{i}"
            coroutines.append(run_agent(agent_id, task))
        
        # 并行执行
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 处理结果
        agent_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {i} failed: {result}")
                # 创建失败结果
                agent_results.append(AgentResult(
                    agent_id=f"agent-{i}",
                    task=tasks[i],
                    final_answer=f"执行失败: {str(result)}",
                    trajectory=[],
                    total_tokens=TokenUsage(),
                    status="error"
                ))
            else:
                agent_results.append(result)
        
        return agent_results
    
    async def _merge_results(self, query: str, agent_results: List[AgentResult]) -> str:
        """合并多个 Agent 的结果"""
        
        # 构建结果汇总
        results_summary = []
        for result in agent_results:
            results_summary.append(f"""[{result.agent_id}]
任务: {result.task}
结果: {result.final_answer}
""")
        
        results_text = "\n---\n".join(results_summary)
        
        prompt = f"""基于以下多个子任务的分析结果，请综合回答用户的原始问题。

用户原始问题: {query}

各子任务分析结果:
{results_text}

请:
1. 综合分析各子任务的结果
2. 整合成一个完整、连贯的答案
3. 如果子任务结果有冲突，请说明并给出最合理的结论
4. 直接给出最终答案，不要提及子任务分配"""
        
        try:
            response = await self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                model="deepseek-chat"
            )
            
            return response.content.strip()
        
        except Exception as e:
            logger.error(f"Result merging error: {e}")
            # 降级：简单拼接
            return "\n\n".join([f"[{r.agent_id}]\n{r.final_answer}" for r in agent_results])
