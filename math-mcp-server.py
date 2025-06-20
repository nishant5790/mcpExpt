#!/usr/bin/env python3
"""
Math MCP Server with SSE Support
A Model Context Protocol server that provides mathematical operations and can be deployed with Server-Sent Events.
"""

import asyncio
import json
import logging
import math
import statistics
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# MCP SDK imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.server.sse import sse_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    CallToolResult,
    ListToolsResult,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MathMCPServer:
    """Math MCP Server providing various mathematical operations."""
    
    def __init__(self):
        self.server = Server("math-server")
        self.setup_tools()
        self.setup_handlers()
    
    def setup_tools(self):
        """Define the mathematical tools available in this server."""
        self.tools = [
            Tool(
                name="basic_math",
                description="Perform basic mathematical operations: add, subtract, multiply, divide",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The mathematical operation to perform"
                        },
                        "a": {
                            "type": "number",
                            "description": "First number"
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            ),
            Tool(
                name="advanced_math",
                description="Perform advanced mathematical operations: power, square root, logarithm, trigonometry",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["power", "sqrt", "log", "log10", "sin", "cos", "tan", "factorial"],
                            "description": "The advanced mathematical operation to perform"
                        },
                        "value": {
                            "type": "number",
                            "description": "The input value"
                        },
                        "base": {
                            "type": "number",
                            "description": "Base for power operation or logarithm base (optional)",
                            "default": 10
                        }
                    },
                    "required": ["operation", "value"]
                }
            ),
            Tool(
                name="statistics",
                description="Calculate statistical measures for a list of numbers",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of numbers to calculate statistics for"
                        },
                        "measures": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["mean", "median", "mode", "stdev", "variance", "min", "max", "sum", "count"]
                            },
                            "description": "Statistical measures to calculate",
                            "default": ["mean", "median", "stdev"]
                        }
                    },
                    "required": ["numbers"]
                }
            ),
            Tool(
                name="solve_quadratic",
                description="Solve quadratic equations of the form ax² + bx + c = 0",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "Coefficient of x²"
                        },
                        "b": {
                            "type": "number",
                            "description": "Coefficient of x"
                        },
                        "c": {
                            "type": "number",
                            "description": "Constant term"
                        }
                    },
                    "required": ["a", "b", "c"]
                }
            ),
            Tool(
                name="number_theory",
                description="Number theory operations: GCD, LCM, prime check, prime factors",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["gcd", "lcm", "is_prime", "prime_factors", "fibonacci"],
                            "description": "Number theory operation to perform"
                        },
                        "a": {
                            "type": "integer",
                            "description": "First integer (or the integer for single-value operations)"
                        },
                        "b": {
                            "type": "integer",
                            "description": "Second integer (for GCD and LCM operations)",
                            "default": None
                        }
                    },
                    "required": ["operation", "a"]
                }
            )
        ]
    
    def setup_handlers(self):
        """Set up the MCP server handlers."""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """Return the list of available tools."""
            return ListToolsResult(tools=self.tools)
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                if name == "basic_math":
                    result = await self._handle_basic_math(arguments)
                elif name == "advanced_math":
                    result = await self._handle_advanced_math(arguments)
                elif name == "statistics":
                    result = await self._handle_statistics(arguments)
                elif name == "solve_quadratic":
                    result = await self._handle_solve_quadratic(arguments)
                elif name == "number_theory":
                    result = await self._handle_number_theory(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return CallToolResult(
                    content=[TextContent(type="text", text=result)]
                )
            
            except Exception as e:
                logger.error(f"Error handling tool call {name}: {str(e)}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )
    
    async def _handle_basic_math(self, args: Dict[str, Any]) -> str:
        """Handle basic mathematical operations."""
        operation = args["operation"]
        a = args["a"]
        b = args["b"]
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero is not allowed")
            result = a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return f"{a} {operation} {b} = {result}"
    
    async def _handle_advanced_math(self, args: Dict[str, Any]) -> str:
        """Handle advanced mathematical operations."""
        operation = args["operation"]
        value = args["value"]
        base = args.get("base", 10)
        
        if operation == "power":
            if "base" not in args:
                raise ValueError("Base is required for power operation")
            result = value ** base
            return f"{value}^{base} = {result}"
        elif operation == "sqrt":
            if value < 0:
                raise ValueError("Cannot calculate square root of negative number")
            result = math.sqrt(value)
            return f"√{value} = {result}"
        elif operation == "log":
            if value <= 0:
                raise ValueError("Logarithm undefined for non-positive numbers")
            result = math.log(value, base)
            return f"log_{base}({value}) = {result}"
        elif operation == "log10":
            if value <= 0:
                raise ValueError("Logarithm undefined for non-positive numbers")
            result = math.log10(value)
            return f"log₁₀({value}) = {result}"
        elif operation == "sin":
            result = math.sin(math.radians(value))
            return f"sin({value}°) = {result}"
        elif operation == "cos":
            result = math.cos(math.radians(value))
            return f"cos({value}°) = {result}"
        elif operation == "tan":
            result = math.tan(math.radians(value))
            return f"tan({value}°) = {result}"
        elif operation == "factorial":
            if value < 0 or value != int(value):
                raise ValueError("Factorial is only defined for non-negative integers")
            result = math.factorial(int(value))
            return f"{int(value)}! = {result}"
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _handle_statistics(self, args: Dict[str, Any]) -> str:
        """Handle statistical calculations."""
        numbers = args["numbers"]
        measures = args.get("measures", ["mean", "median", "stdev"])
        
        if not numbers:
            raise ValueError("Cannot calculate statistics for empty list")
        
        results = []
        
        for measure in measures:
            if measure == "mean":
                value = statistics.mean(numbers)
                results.append(f"Mean: {value}")
            elif measure == "median":
                value = statistics.median(numbers)
                results.append(f"Median: {value}")
            elif measure == "mode":
                try:
                    value = statistics.mode(numbers)
                    results.append(f"Mode: {value}")
                except statistics.StatisticsError:
                    results.append("Mode: No unique mode found")
            elif measure == "stdev":
                if len(numbers) < 2:
                    results.append("Standard Deviation: Need at least 2 values")
                else:
                    value = statistics.stdev(numbers)
                    results.append(f"Standard Deviation: {value}")
            elif measure == "variance":
                if len(numbers) < 2:
                    results.append("Variance: Need at least 2 values")
                else:
                    value = statistics.variance(numbers)
                    results.append(f"Variance: {value}")
            elif measure == "min":
                value = min(numbers)
                results.append(f"Minimum: {value}")
            elif measure == "max":
                value = max(numbers)
                results.append(f"Maximum: {value}")
            elif measure == "sum":
                value = sum(numbers)
                results.append(f"Sum: {value}")
            elif measure == "count":
                value = len(numbers)
                results.append(f"Count: {value}")
        
        return "\n".join(results)
    
    async def _handle_solve_quadratic(self, args: Dict[str, Any]) -> str:
        """Solve quadratic equations."""
        a = args["a"]
        b = args["b"]
        c = args["c"]
        
        if a == 0:
            raise ValueError("Coefficient 'a' cannot be zero for quadratic equation")
        
        discriminant = b**2 - 4*a*c
        
        if discriminant > 0:
            x1 = (-b + math.sqrt(discriminant)) / (2*a)
            x2 = (-b - math.sqrt(discriminant)) / (2*a)
            return f"Quadratic equation {a}x² + {b}x + {c} = 0\nSolutions: x₁ = {x1}, x₂ = {x2}"
        elif discriminant == 0:
            x = -b / (2*a)
            return f"Quadratic equation {a}x² + {b}x + {c} = 0\nSolution: x = {x} (double root)"
        else:
            real_part = -b / (2*a)
            imaginary_part = math.sqrt(-discriminant) / (2*a)
            return f"Quadratic equation {a}x² + {b}x + {c} = 0\nComplex solutions: x₁ = {real_part} + {imaginary_part}i, x₂ = {real_part} - {imaginary_part}i"
    
    async def _handle_number_theory(self, args: Dict[str, Any]) -> str:
        """Handle number theory operations."""
        operation = args["operation"]
        a = args["a"]
        b = args.get("b")
        
        if operation == "gcd":
            if b is None:
                raise ValueError("Second number 'b' is required for GCD operation")
            result = math.gcd(a, b)
            return f"GCD({a}, {b}) = {result}"
        elif operation == "lcm":
            if b is None:
                raise ValueError("Second number 'b' is required for LCM operation")
            result = abs(a * b) // math.gcd(a, b)
            return f"LCM({a}, {b}) = {result}"
        elif operation == "is_prime":
            if a < 2:
                return f"{a} is not prime"
            for i in range(2, int(math.sqrt(a)) + 1):
                if a % i == 0:
                    return f"{a} is not prime"
            return f"{a} is prime"
        elif operation == "prime_factors":
            if a < 2:
                return f"{a} has no prime factors"
            factors = []
            d = 2
            while d * d <= a:
                while a % d == 0:
                    factors.append(d)
                    a //= d
                d += 1
            if a > 1:
                factors.append(a)
            return f"Prime factors of {args['a']}: {factors}"
        elif operation == "fibonacci":
            if a < 0:
                raise ValueError("Fibonacci sequence is defined for non-negative integers")
            if a == 0:
                return f"F({a}) = 0"
            elif a == 1:
                return f"F({a}) = 1"
            else:
                fib_a, fib_b = 0, 1
                for _ in range(2, a + 1):
                    fib_a, fib_b = fib_b, fib_a + fib_b
                return f"F({a}) = {fib_b}"
        else:
            raise ValueError(f"Unknown operation: {operation}")

# Server deployment functions
async def run_stdio_server():
    """Run the server using stdio transport."""
    server_instance = MathMCPServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="math-server",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

async def run_sse_server(host: str = "localhost", port: int = 8000):
    """Run the server using SSE transport."""
    server_instance = MathMCPServer()
    
    async with sse_server(host, port) as server:
        await server_instance.server.run(
            server.read_stream,
            server.write_stream,
            InitializationOptions(
                server_name="math-server",
                server_version="1.0.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        # Run with SSE transport
        host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
        
        print(f"Starting Math MCP Server with SSE on {host}:{port}")
        asyncio.run(run_sse_server(host, port))
    else:
        # Run with stdio transport (default)
        print("Starting Math MCP Server with stdio transport")
        asyncio.run(run_stdio_server())