import os
import asyncio
from dataclasses import dataclass
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI, APIError, RateLimitError
from loguru import logger


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    reasoning: Optional[str] = None
    usage: TokenUsage = None


@dataclass
class StreamChunk:
    type: str  # "thinking", "content", "done"
    text: str = ""
    usage: Optional[TokenUsage] = None


class DeepSeekClient:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # 重试配置
        self.max_retries = 3
        self.initial_delay = 1.0
        self.backoff_factor = 2.0
    
    async def call(
        self,
        messages: list[dict],
        model: str = "deepseek-chat",
        tools: Optional[list] = None,
        use_thinking: bool = False
    ) -> LLMResponse:
        """调用 LLM，支持重试逻辑"""
        if use_thinking:
            model = "deepseek-reasoner"
        
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": False,
        }
        
        # deepseek-reasoner 不支持 temperature 等参数
        if model != "deepseek-reasoner":
            kwargs["temperature"] = 0.7
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        last_exception = None
        delay = self.initial_delay
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                
                msg = response.choices[0].message
                content = msg.content or ""
                
                # 获取 reasoning_content（仅 deepseek-reasoner）
                reasoning = getattr(msg, "reasoning_content", None)
                
                # 获取 token 用量
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
                
                return LLMResponse(
                    content=content,
                    reasoning=reasoning,
                    usage=usage
                )
                
            except RateLimitError as e:
                last_exception = e
                logger.warning(f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}, retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= self.backoff_factor
                
            except APIError as e:
                last_exception = e
                if e.status_code in [500, 502, 503, 504]:
                    logger.warning(f"Server error {e.status_code}, attempt {attempt + 1}/{self.max_retries}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    logger.error(f"API error: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"Unexpected error calling LLM: {e}")
                raise
        
        logger.error(f"Max retries exceeded: {last_exception}")
        raise last_exception
    
    async def call_stream(
        self,
        messages: list[dict],
        model: str = "deepseek-reasoner"
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式调用 LLM，用于 SSE 推送"""
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "stream": True,
        }
        
        # deepseek-reasoner 不支持 temperature
        if model != "deepseek-reasoner":
            kwargs["temperature"] = 0.7
        
        try:
            stream = await self.client.chat.completions.create(**kwargs)
            
            reasoning_buf = ""
            content_buf = ""
            final_usage = None
            
            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                # 处理 reasoning_content（思考模式）
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_buf += delta.reasoning_content
                    yield StreamChunk(type="thinking", text=delta.reasoning_content)
                
                # 处理 content
                if delta.content:
                    content_buf += delta.content
                    yield StreamChunk(type="content", text=delta.content)
                
                # 获取最终用量
                if hasattr(chunk, "usage") and chunk.usage:
                    final_usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
            
            yield StreamChunk(type="done", text="", usage=final_usage)
            
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            raise
