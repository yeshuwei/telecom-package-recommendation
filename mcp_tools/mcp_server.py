"""
MCP 服务器 - 提供通用工具的 MCP 接口
"""
import json
import logging
from typing import Any, Dict
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from mcp_tools.llm_tools import get_llm_tools
from mcp_tools.extraction_tools import extraction_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 MCP 服务器
app = Server("telecom-tools-server")


# 采用旧的API格式 types.ServerResult(types.ListToolsResult(tools=result))将我们返回的列表封装在了新的 ListToolsResult 对象的 tools 属性中。
@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用的工具"""
    return [
        Tool(
            name="generate_content",
            description="使用 Gemini 大模型生成内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "输入的提示词"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "温度参数，控制输出随机性，默认0.1",
                        "default": 0.1
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="extract_json",
            description="从文本中提取 JSON 数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "包含 JSON 的文本"
                    },
                    "is_array": {
                        "type": "boolean",
                        "description": "是否提取 JSON 数组",
                        "default": False
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="generate_and_extract_json",
            description="使用大模型生成内容并自动提取 JSON",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "输入的提示词"
                    },
                    "is_array": {
                        "type": "boolean",
                        "description": "是否提取 JSON 数组",
                        "default": False
                    },
                    "temperature": {
                        "type": "number",
                        "description": "温度参数",
                        "default": 0.1
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="extract_budget",
            description="从文本中提取预算信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "输入文本"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="extract_data_needs",
            description="从文本中提取流量需求（轻度/中度/重度或具体GB数）",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "输入文本"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="extract_call_minutes",
            description="从文本中提取通话时长（分钟）",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "输入文本"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="extract_device_type",
            description="从文本中提取设备类型（5G/4G）",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "输入文本"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="extract_all_slots",
            description="一次性从文本中提取所有槽位信息（预算、流量需求、通话时长、设备类型）",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "输入文本"
                    }
                },
                "required": ["text"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """调用工具"""
    try:
        llm_tools = get_llm_tools()
        
        if name == "generate_content":
            prompt = arguments.get("prompt")
            temperature = arguments.get("temperature", 0.1)
            result = llm_tools.generate_content(prompt, temperature)
            return [TextContent(type="text", text=result or "生成失败")]
        
        elif name == "extract_json":
            text = arguments.get("text")
            is_array = arguments.get("is_array", False)
            if is_array:
                result = llm_tools.extract_json_array_from_response(text)
            else:
                result = llm_tools.extract_json_from_response(text)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False) if result else "null")]
        
        elif name == "generate_and_extract_json":
            prompt = arguments.get("prompt")
            is_array = arguments.get("is_array", False)
            temperature = arguments.get("temperature", 0.1)
            result = llm_tools.generate_and_extract_json(prompt, is_array, temperature)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False) if result else "null")]
        
        elif name == "extract_budget":
            text = arguments.get("text")
            result = extraction_tools.extract_budget(text)
            return [TextContent(type="text", text=str(result) if result is not None else "null")]
        
        elif name == "extract_data_needs":
            text = arguments.get("text")
            result = extraction_tools.extract_data_needs(text)
            return [TextContent(type="text", text=result or "null")]
        
        elif name == "extract_call_minutes":
            text = arguments.get("text")
            result = extraction_tools.extract_call_minutes(text)
            return [TextContent(type="text", text=str(result) if result is not None else "null")]
        
        elif name == "extract_device_type":
            text = arguments.get("text")
            result = extraction_tools.extract_device_type(text)
            return [TextContent(type="text", text=result or "null")]
        
        elif name == "extract_all_slots":
            text = arguments.get("text")
            result = extraction_tools.extract_all_slot_values(text)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        
        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]
    
    except Exception as e:
        logger.error(f"调用工具失败: {e}")
        return [TextContent(type="text", text=f"错误: {str(e)}")]


async def main():
    """运行 MCP 服务器"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

