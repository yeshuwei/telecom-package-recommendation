"""
MCP 工具：信息提取相关通用函数
总的来说就是将各种降级匹配工具定义在此
"""
import re
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionTools:
    """信息提取相关的通用工具"""

    # Python内置装饰器 用于将类中的方法标记为静态方法
    @staticmethod
    def extract_budget(text: str) -> Optional[int]:
        """
        从文本中提取预算信息
        
        Args:
            text: 输入文本
            
        Returns:
            预算金额（元），未找到返回 None
        """
        budget_patterns = [
            r'(\d+)元以下', r'预算(\d+)', r'不超过(\d+)', r'(\d+)块钱',
            r'(\d+)元左右', r'大概(\d+)', r'(\d+)以内', r'(\d+)元'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, text)
            if match:
                budget = int(match.group(1))
                logger.info(f"提取到预算: {budget}元")
                return budget
        
        return None
    
    @staticmethod
    def extract_data_needs(text: str) -> Optional[str]:
        """
        从文本中提取流量需求
        
        Args:
            text: 输入文本
            
        Returns:
            流量需求（"轻度"/"中度"/"重度" 或具体GB数），未找到返回 None
        """
        text_lower = text.lower()
        
        # 检查关键词
        if any(word in text for word in ["大流量", "无限流量", "不限流量", "流量多", "很多流量", "重度"]):
            logger.info("提取到流量需求: 重度")
            return "重度"
        elif any(word in text for word in ["轻度", "少量", "基本", "不常用", "流量少"]):
            logger.info("提取到流量需求: 轻度")
            return "轻度"
        elif any(word in text for word in ["中等", "一般", "正常", "够用", "中度"]):
            logger.info("提取到流量需求: 中度")
            return "中度"
        
        # 尝试提取具体GB数
        gb_pattern = r'(\d+)\s*[GgＧ][Bb]?'
        match = re.search(gb_pattern, text)
        if match:
            gb_value = f"{match.group(1)}GB"
            logger.info(f"提取到流量需求: {gb_value}")
            return gb_value
        
        return None
    
    @staticmethod
    def extract_call_minutes(text: str) -> Optional[int]:
        """
        从文本中提取通话时长
        
        Args:
            text: 输入文本
            
        Returns:
            通话分钟数，未找到返回 None
        """
        call_patterns = [
            r'(\d+)分钟', r'通话(\d+)', r'打电话(\d+)', r'(\d+)分'
        ]
        
        for pattern in call_patterns:
            match = re.search(pattern, text)
            if match:
                minutes = int(match.group(1))
                logger.info(f"提取到通话时长: {minutes}分钟")
                return minutes
        
        return None
    
    @staticmethod
    def extract_device_type(text: str) -> Optional[str]:
        """
        从文本中提取设备类型
        
        Args:
            text: 输入文本
            
        Returns:
            设备类型（"5G"/"4G"），未找到返回 None
        """
        if "5G" in text or "5g" in text or "五G" in text:
            logger.info("提取到设备类型: 5G")
            return "5G"
        elif "4G" in text or "4g" in text or "四G" in text:
            logger.info("提取到设备类型: 4G")
            return "4G"
        
        return None
    
    @staticmethod
    def extract_all_slot_values(text: str) -> Dict[str, Any]:
        """
        一次性提取所有槽位信息
        
        Args:
            text: 输入文本
            
        Returns:
            包含所有提取信息的字典
        """
        extracted = {}
        
        budget = ExtractionTools.extract_budget(text)
        if budget is not None:
            extracted["budget"] = budget
        
        data_needs = ExtractionTools.extract_data_needs(text)
        if data_needs is not None:
            extracted["data_needs"] = data_needs
        
        call_minutes = ExtractionTools.extract_call_minutes(text)
        if call_minutes is not None:
            extracted["call_minutes"] = call_minutes
        
        device_type = ExtractionTools.extract_device_type(text)
        if device_type is not None:
            extracted["device_type"] = device_type
        
        logger.info(f"提取到的信息: {extracted}")
        return extracted
    
    @staticmethod
    def extract_keywords(text: str, keyword_lists: Dict[str, list]) -> Dict[str, list]:
        """
        从文本中提取匹配的关键词
        
        Args:
            text: 输入文本
            keyword_lists: 关键词列表字典，格式如 {"category1": ["word1", "word2"], ...}
            
        Returns:
            匹配到的关键词字典
        """
        text_lower = text.lower()
        matched = {}
        
        for category, keywords in keyword_lists.items():
            matched_keywords = [kw for kw in keywords if kw in text_lower]
            if matched_keywords:
                matched[category] = matched_keywords
        
        return matched


# 全局实例
extraction_tools = ExtractionTools()

