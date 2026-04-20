import json
import math
from .base import BaseTool


class CalculatorTool(BaseTool):
    """计算器工具，支持基本数学运算"""
    
    name = "calculator"
    description = "执行数学计算，支持基本运算（加减乘除、幂运算、开方、对数等）"
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，例如 '2 + 2', 'sqrt(16)', 'log(100)'"
            }
        },
        "required": ["expression"]
    }
    
    async def run(self, **kwargs) -> str:
        expression = kwargs.get("expression", "")
        
        try:
            # 安全评估表达式
            result = self._safe_eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    def _safe_eval(self, expression: str) -> float:
        """安全地评估数学表达式"""
        # 定义允许的名称
        safe_dict = {
            'sqrt': math.sqrt,
            'pow': math.pow,
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
            'sum': sum,
            'len': len,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
        }
        
        # 清理表达式
        expression = expression.strip()
        
        # 使用 eval 计算（在安全环境中）
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return result
        except Exception as e:
            raise ValueError(f"无法计算表达式 '{expression}': {str(e)}")


class CalculatorToolSimple(BaseTool):
    """简化版计算器，用于演示"""
    
    name = "calculate"
    description = "执行基本数学运算：加、减、乘、除"
    input_schema = {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "第一个数字"
            },
            "b": {
                "type": "number",
                "description": "第二个数字"
            },
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "运算类型"
            }
        },
        "required": ["a", "b", "operation"]
    }
    
    async def run(self, **kwargs) -> str:
        a = kwargs.get("a", 0)
        b = kwargs.get("b", 0)
        operation = kwargs.get("operation", "")
        
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return "错误：除数不能为零"
                result = a / b
            else:
                return f"错误：不支持的运算类型 '{operation}'"
            
            return f"{a} {self._get_op_symbol(operation)} {b} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    def _get_op_symbol(self, operation: str) -> str:
        symbols = {
            "add": "+",
            "subtract": "-",
            "multiply": "×",
            "divide": "÷"
        }
        return symbols.get(operation, operation)
