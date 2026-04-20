import math
from .base import BaseTool


class CalculatorTool(BaseTool):
    """计算器工具，支持表达式求值（加减乘除、幂、开方、对数等）。

    之前代码里同时注册了这个 ``calculator`` 和一个简化版 ``calculate``（只支持 +-×÷）。
    简化版会把"2 的 10 次方"这种问题逼着 LLM 连续十次 ``multiply(2,2)`` 迭代，
    每轮都重算整个 messages + tools schema，对 token 和耗时都是二次方放大。
    所以这里只保留表达式版，在 description 里**明确告知 LLM 幂运算应该一次搞定**。
    """

    name = "calculator"
    description = (
        "执行数学计算。**幂运算、开方、对数都应通过一次调用直接求值，"
        "不要用乘法多次迭代**。expression 是完整的算术表达式。\n"
        "示例：\n"
        "  - 2 的 10 次方        → expression=\"2**10\"\n"
        "  - 开平方根            → expression=\"sqrt(16)\"\n"
        "  - 自然对数            → expression=\"log(100)\"\n"
        "  - 组合表达式          → expression=\"(3+4)*5 - sqrt(9)\""
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "完整的数学表达式，例如 '2+2'、'2**10'、'sqrt(16)'、'log(100)'。"
                    "支持 + - * / ** ()、sqrt/pow/abs/round/max/min/sin/cos/tan/log/log10/exp/pi/e。"
                ),
            }
        },
        "required": ["expression"],
    }

    async def run(self, **kwargs) -> str:
        expression = kwargs.get("expression", "")

        try:
            result = self._safe_eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    def _safe_eval(self, expression: str) -> float:
        """在受限命名空间下对数学表达式求值。

        显式禁用 __builtins__，只暴露数学相关符号。这比起让 LLM 自己迭代更安全、更省 token。
        """
        safe_dict = {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            "len": len,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        expression = (expression or "").strip()
        if not expression:
            raise ValueError("空表达式")

        try:
            return eval(expression, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise ValueError(f"无法计算表达式 '{expression}': {str(e)}")
