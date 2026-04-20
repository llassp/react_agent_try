import random
from .base import BaseTool


class SearchTool(BaseTool):
    """模拟搜索工具（实际项目中可替换为真实搜索 API）"""
    
    name = "search"
    description = "搜索网络信息，获取关于特定主题的知识"
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询关键词"
            }
        },
        "required": ["query"]
    }
    
    # 模拟知识库
    KNOWLEDGE_BASE = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。它以简洁、易读的语法著称。",
        "react": "React 是 Facebook 开发的用于构建用户界面的 JavaScript 库，采用组件化开发模式。",
        "deepseek": "DeepSeek 是一家中国 AI 公司，开发了 DeepSeek-V3 等大型语言模型，以高性价比著称。",
        "人工智能": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习": "机器学习是 AI 的子集，通过数据训练模型，使计算机能够从经验中学习和改进。",
    }
    
    async def run(self, **kwargs) -> str:
        query = kwargs.get("query", "").lower()
        
        if not query:
            return "搜索错误：查询不能为空"
        
        # 模拟搜索延迟
        # await asyncio.sleep(0.5)
        
        # 查找匹配的知识
        results = []
        for key, value in self.KNOWLEDGE_BASE.items():
            if key.lower() in query or query in key.lower():
                results.append(f"[{key}] {value}")
        
        if results:
            return "\n".join(results)
        
        # 如果没有直接匹配，返回模拟结果
        return f"搜索 '{query}' 的结果：\n- 找到相关文档 {random.randint(3, 15)} 篇\n- 主要涉及领域：技术、科学\n- 建议进一步细化查询以获得更精确的结果"


class WeatherTool(BaseTool):
    """模拟天气查询工具"""
    
    name = "weather"
    description = "查询指定城市的天气信息"
    input_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"]
    }
    
    async def run(self, **kwargs) -> str:
        city = kwargs.get("city", "")
        
        if not city:
            return "错误：请提供城市名称"
        
        # 模拟天气数据
        conditions = ["晴朗", "多云", "阴天", "小雨", "大雨"]
        condition = random.choice(conditions)
        temp = random.randint(15, 35)
        humidity = random.randint(40, 90)
        
        return f"{city} 天气：{condition}，温度 {temp}°C，湿度 {humidity}%"


class DateTimeTool(BaseTool):
    """获取当前日期时间"""
    
    name = "datetime"
    description = "获取当前日期和时间信息"
    input_schema = {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["full", "date", "time"],
                "description": "返回格式：full=完整日期时间，date=仅日期，time=仅时间"
            }
        },
        "required": []
    }
    
    async def run(self, **kwargs) -> str:
        from datetime import datetime
        
        format_type = kwargs.get("format", "full")
        now = datetime.now()
        
        if format_type == "date":
            return now.strftime("%Y-%m-%d")
        elif format_type == "time":
            return now.strftime("%H:%M:%S")
        else:
            return now.strftime("%Y-%m-%d %H:%M:%S")
