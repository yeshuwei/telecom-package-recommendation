"""
MCP 工具：大模型相关通用函数
"""
import json
import re
import logging
from typing import Dict, Any, Optional, List
import os
import openai

from configs.config import DASHSCOPE_API_KEY, MODEL_NAME, OPENAI_BASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMTools:
    """大模型相关的通用工具"""
    
    def __init__(self, api_key: str = None):
        """初始化 LLM 工具（使用阿里云 DashScope 的 OpenAI 兼容接口）"""
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or DASHSCOPE_API_KEY
        
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY")
        
        # 也设置环境变量，兼容部分SDK/三方库的读取方式
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=OPENAI_BASE_URL
        )

        logger.info("LLM工具初始化完毕 (Qwen via DashScope OpenAI-compatible)")
    
    def chat_once(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        temperature: float = 0.1,
        model: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        以 OpenAI 兼容格式进行一次对话调用，支持 function calling (tools)。
        返回原始 response；调用方自行读取 message/content 或 tool_calls。
        """
        try:
            kwargs: Dict[str, Any] = {
                "model": model or MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
            }
            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
            if extra:
                kwargs.update(extra)

            response = self.client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            logger.error(f"chat_once 调用失败: {e}")
            return None
    
    def generate_content(self, prompt: str, temperature: float = 0.8) -> Optional[str]:
        """
        调用大模型生成内容
        
        Args:
            prompt: 提示词
            temperature: 温度参数，控制输出的随机性
            
        Returns:
            生成的文本内容，失败返回 None
        """
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )

            if response and response.choices and response.choices[0].message.content:
                logger.info("成功生成内容")
                return response.choices[0].message.content
            else:
                logger.error("API返回为空")
                return None
                
        except Exception as e:
            logger.error(f"生成内容失败: {e}")
            return None
    
    def extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        从响应文本中提取 JSON
        
        Args:
            response_text: 响应文本
            
        Returns:
            解析后的字典，失败返回 None
        """
        try:
            # 尝试直接解析
            return json.loads(response_text)
        except json.JSONDecodeError:
            # 尝试提取 JSON 代码块
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 尝试提取大括号内容
            brace_pattern = r'\{.*?\}'
            match = re.search(brace_pattern, response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"无法从响应中提取有效JSON: {response_text[:200]}")
            return None
    
    def extract_json_array_from_response(self, response_text: str) -> Optional[list]:
        """
        从响应文本中提取 JSON 数组
        
        Args:
            response_text: 响应文本
            
        Returns:
            解析后的列表，失败返回 None
        """
        try:
            text = response_text.strip()
            
            # 去除可能的 markdown 格式
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            
            # 尝试找到 JSON 数组的开始和结束位置
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_text = text[start_idx:end_idx]
                result = json.loads(json_text)
                if isinstance(result, list):
                    return result
                else:
                    logger.warning("JSON解析结果不是列表格式")
                    return None
            else:
                logger.warning("未找到有效的JSON数组")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON数组解析失败: {e}")
            return None
    
    def generate_and_extract_json(
        self, 
        prompt: str, 
        is_array: bool = False,
        temperature: float = 0.1
    ) -> Optional[Any]:
        """
        生成内容并提取 JSON（组合操作）
        
        Args:
            prompt: 提示词
            is_array: 是否提取 JSON 数组
            temperature: 温度参数
            
        Returns:
            解析后的数据结构，失败返回 None
        """
        response_text = self.generate_content(prompt, temperature)
        
        if not response_text:
            return None
        
        if is_array:
            return self.extract_json_array_from_response(response_text)
        else:
            return self.extract_json_from_response(response_text)


# 全局单例
_llm_tools_instance = None


def get_llm_tools() -> LLMTools:
    """获取 LLM 工具单例"""
    global _llm_tools_instance
    if _llm_tools_instance is None:
        _llm_tools_instance = LLMTools()
    return _llm_tools_instance
