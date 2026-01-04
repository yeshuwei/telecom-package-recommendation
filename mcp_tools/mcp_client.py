"""
MCP 客户端 - 方便智能体调用 MCP 工具
"""
import json
import logging
from typing import Any, Dict, Optional

# 直接导入工具，避免MCP协议的复杂性
from mcp_tools.llm_tools import get_llm_tools
from mcp_tools.extraction_tools import extraction_tools
from mcp_tools.mysql_tools import mysql_tools
from milvus.rag_knowledge_base import RAGKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 客户端包装器"""
    
    def __init__(self):
        """初始化 MCP 客户端"""
        self.llm_tools = get_llm_tools()
        self.extraction_tools = extraction_tools
        self.mysql_tools = mysql_tools
        
        # 初始化RAG知识库
        self.rag_tools = RAGKnowledgeBase()
        self.rag_tools.connect()
        
        logger.info("MCP客户端初始化完成（包含RAG工具）")
    
    def chat_once(self, messages, tools=None, tool_choice: str = "auto", temperature: float = 0.0, model: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        """转发到 LLMTools.chat_once，支持OpenAI兼容的tools调用。"""
        return self.llm_tools.chat_once(messages=messages, tools=tools, tool_choice=tool_choice, temperature=temperature, model=model, extra=extra)

    def generate_content(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """
        调用大模型生成内容
        Args:
            prompt: 提示词
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        return self.llm_tools.generate_content(prompt, temperature)
    
    def extract_json(self, text: str, is_array: bool = False) -> Optional[Any]:
        """
        从文本中提取 JSON
        
        Args:
            text: 包含 JSON 的文本
            is_array: 是否提取 JSON 数组
            
        Returns:
            解析后的数据
        """
        if is_array:
            return self.llm_tools.extract_json_array_from_response(text)
        else:
            return self.llm_tools.extract_json_from_response(text)
    
    def generate_and_extract_json(
        self,
        prompt: str,
        is_array: bool = False,
        temperature: float = 0.1
    ) -> Optional[Any]:
        """
        生成内容并提取 JSON（一步完成）
        
        Args:
            prompt: 提示词
            is_array: 是否提取 JSON 数组
            temperature: 温度参数
            
        Returns:
            解析后的数据
        """
        return self.llm_tools.generate_and_extract_json(prompt, is_array, temperature)
    
    def extract_budget(self, text: str) -> Optional[int]:
        """从文本中提取预算"""
        return self.extraction_tools.extract_budget(text)
    
    def extract_data_needs(self, text: str) -> Optional[str]:
        """从文本中提取流量需求"""
        return self.extraction_tools.extract_data_needs(text)
    
    def extract_call_minutes(self, text: str) -> Optional[int]:
        """从文本中提取通话时长"""
        return self.extraction_tools.extract_call_minutes(text)
    
    def extract_device_type(self, text: str) -> Optional[str]:
        """从文本中提取设备类型"""
        return self.extraction_tools.extract_device_type(text)
    
    def extract_all_slots(self, text: str) -> Dict[str, Any]:
        """一次性提取所有槽位信息"""
        return self.extraction_tools.extract_all_slot_values(text)


# 全局单例
_mcp_client_instance = None


def get_mcp_client() -> MCPClient:
    """获取 MCP 客户端单例"""
    global _mcp_client_instance
    if _mcp_client_instance is None:
        _mcp_client_instance = MCPClient()
    return _mcp_client_instance

