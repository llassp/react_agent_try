from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """工具基类，支持 OpenAI function calling 格式"""
    
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON Schema 格式
    
    @abstractmethod
    async def run(self, **kwargs) -> str:
        """执行工具，返回字符串结果"""
        pass
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """转换为 OpenAI tools 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """获取工具定义（用于文本解析模式）"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }
