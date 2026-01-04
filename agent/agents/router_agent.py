"""
路由智能体 - 负责分析用户输入，识别用户意图并路由到对应的智能体
"""
import json
import re
from typing import Dict, Any, List, Optional
from enum import Enum

import logging
import os

from agent.state import AgentState
# from configs.config import GEMINI_API_KEY  # 未使用的导入
from mcp_tools.mcp_client import get_mcp_client

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """用户意图类型枚举"""
    UNCLEAR_RECOMMENDATION = "unclear_recommendation"  # 用户没有明确需求的推荐
    CLARIFY_RECOMMENDATION = "clarify_recommendation"  # 用户有明确需求的推荐
    INQUIRY = "inquiry"  # 信息查询
    COMPARISON = "comparison"  # 比较决策
    GENERAL = "general"  # 闲聊与非任务


# 意图到智能体的路由映射（全局常量）
INTENT_ROUTING_MAP = {
    IntentType.UNCLEAR_RECOMMENDATION.value: "user_info_agent",  # 流程1：无明确需求
    IntentType.CLARIFY_RECOMMENDATION.value: "slot_filling_agent",  # 流程2：有明确需求
    IntentType.INQUIRY.value: "knowledge_comparison_agent",
    IntentType.COMPARISON.value: "knowledge_comparison_agent",
    IntentType.GENERAL.value: "general_agent"
}


class RouterAgent:
    """路由智能体类"""

    def __init__(self, gemini_api_key: str = None):  # 保留参数但设置默认值
        """
        初始化路由智能体
        Args: gemini的APIkey
        """

        self.intent_keywords = self._init_intent_keywords()
        self.intent_prompt = self._create_intent_prompt()
        
        # 使用 MCP 客户端
        self.mcp_client = get_mcp_client()
        logger.info("router_agent初始化完毕（使用MCP工具）")

    def _init_intent_keywords(self) -> Dict[IntentType, List[str]]:
        """初始化意图关键词字典"""
        return {
            IntentType.CLARIFY_RECOMMENDATION: [
                "推荐", "建议", "适合", "选择", "套餐", "方案", "预算", "需要", "想要",
                "流量", "通话", "5G", "4G", "月费", "资费", "便宜", "划算", "性价比",
                "学生", "老人", "家庭", "商务", "无限", "不限", "大流量"
            ],
            IntentType.INQUIRY: [
                "什么是", "如何", "怎么", "详细", "介绍", "说明", "规则", "条款", "办理",
                "流程", "手续", "要求", "条件", "限制", "范围", "覆盖", "速度", "网络",
                "查询", "了解", "知道", "告诉我", "解释", "具体", "细节"
            ],
            IntentType.COMPARISON: [
                "比较", "对比", "差别", "区别", "哪个好", "哪个更", "优劣", "优缺点",
                "vs", "和", "与", "相比", "对照", "分析", "评估", "选哪个", "差异"
            ],
            IntentType.GENERAL: [
                "你好", "谢谢", "再见", "聊天", "天气", "心情", "无聊", "开玩笑",
                "哈哈", "呵呵", "不错", "好的", "明白", "知道了", "没事", "随便"
            ]
        }

    def _create_intent_prompt(self) -> str:
        """创建意图识别的提示词"""
        return """你是一个电信套餐客服助手的意图识别模块。请分析用户输入，识别用户意图类型。
                    
                    用户意图类型：
                    1. clarify_recommendation - 明确推荐需求：用户表达了具体需求（如预算、流量等），希望根据需求推荐套餐
                       示例："我要50元以下的套餐"、"需要大流量的5G套餐"、"预算100元，通话多的套餐"
                    
                    2. unclear_recommendation - 模糊推荐需求：用户想要推荐但没有提供具体需求
                       示例："帮我推荐个套餐"、"我想办理套餐"、"有什么合适的套餐吗"
                    
                    3. inquiry - 信息查询：用户想了解套餐详情、办理流程、服务条款等
                       示例："什么是5G套餐"、"如何办理"、"这个A套餐包含什么"、"你能详细介绍下套餐B吗"
                    
                    4. comparison - 比较决策：用户想要对比不同套餐的差异
                       示例："A套餐和B套餐有什么区别"、"哪个更划算"、"对比这两个套餐"
                    
                    5. general - 闲聊非任务：日常问候、闲聊或与业务无关的对话
                       示例："你好"、"谢谢"、"天气怎么样"
                    
                    请分析以下用户输入，判断其意图类型：
                    
                    用户输入：{user_input}
                    
                    返回格式（只需返回JSON，不要其他说明）：
                    {{
                        "intent": "意图类型(clarify_recommendation/unclear_recommendation/inquiry/comparison/general)",
                        "confidence": 0.0-1.0的置信度,
                        "reasoning": "简短的判断理由（一句话）"
                    }}
                    
                    注意：只需要分类意图，不需要提取具体信息。"""

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """从响应文本中提取JSON（使用MCP工具）"""
        return self.mcp_client.extract_json(response_text, is_array=False)

    def _simple_keyword_match(self, user_input: str) -> Dict[str, Any]:
        """
        简单的关键词匹配降级策略
        当API调用失败时使用
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            包含意图分析结果的字典
        """
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        # 计算每种意图的匹配分数
        for intent_type, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                intent_scores[intent_type] = score
        
        # 确定最高分意图
        if not intent_scores:
            predicted_intent = IntentType.GENERAL
            confidence = 0.3
            reasoning = "未匹配到任何关键词，默认为通用意图"
        else:
            predicted_intent = max(intent_scores, key=intent_scores.get)
            max_score = intent_scores[predicted_intent]
            # 判断用户需求是否明确（推荐类意图）
            # 使用相对比例判断：如果匹配关键词数量少，可能是模糊推荐
            total_keywords = len(self.intent_keywords[IntentType.CLARIFY_RECOMMENDATION])
            keyword_ratio = max_score / total_keywords
            if predicted_intent == IntentType.CLARIFY_RECOMMENDATION and keyword_ratio < 0.2:
                predicted_intent = IntentType.UNCLEAR_RECOMMENDATION
            # 置信度基于匹配的关键词数量，最高0.6（表示降级方案）
            confidence = min(0.3 + max_score * 0.03, 0.6)
            reasoning = f"关键词匹配降级策略，匹配到{max_score}个关键词"
        
        logger.warning(f"使用降级策略识别意图: {predicted_intent.value} (置信度: {confidence:.2f})")
        
        return {
            "intent": predicted_intent.value,
            "confidence": confidence,
            "reasoning": reasoning
        }

    def analyze_intent(self, user_input: str) -> Dict[str, Any]:
        """
        分析用户意图
        Args:
            user_input: 用户输入文本
        Returns:
            包含意图分析结果的字典
        """
        logger.info(f"开始分析用户意图: {user_input}")

        try:
            # 构建提示词
            prompt = self.intent_prompt.format(user_input=user_input)

            # 使用 MCP 工具生成并提取 JSON
            result = self.mcp_client.generate_and_extract_json(prompt, is_array=False, temperature=0.1)

            if result:
                logger.info("成功调用API分析用户意图")
                # 验证和标准化结果
                return self._validate_intent_result(result, user_input)
            else:
                logger.warning("API返回内容无法解析，使用降级策略")
                return self._simple_keyword_match(user_input)

        except Exception as e:
            logger.error(f"API调用失败: {e}，使用降级策略")
            return self._simple_keyword_match(user_input)

    def _validate_intent_result(self, result: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """
        验证和标准化意图分析结果
        
        Args:
            result: LLM返回的原始结果
            user_input: 用户输入文本
            
        Returns:
            标准化后的意图分析结果（只包含意图分类信息）
        """
        # 确保必要字段存在
        validated_result = {
            "intent": result.get("intent", "general"),
            "confidence": max(0.0, min(1.0, result.get("confidence", 0.5))),
            "reasoning": result.get("reasoning", "AI模型分析结果")
        }

        # 验证意图类型
        valid_intents = [intent.value for intent in IntentType]
        if validated_result["intent"] not in valid_intents:
            logger.warning(f"无效的意图类型: {validated_result['intent']}，默认为general")
            validated_result["intent"] = "general"
            validated_result["confidence"] = 0.3

        logger.info(f"意图分析结果: {validated_result['intent']} (置信度: {validated_result['confidence']:.2f})")
        return validated_result

    def route_to_agent(self, intent_result: Dict[str, Any]) -> str:
        """
        根据意图分析结果路由到对应的智能体
        Args:
            intent_result: 意图分析结果
        Returns:
            目标智能体名称
        """
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # 低置信度时路由到通用智能体
        if confidence < 0.4:
            logger.info(f"置信度过低({confidence:.2f})，路由到通用智能体")
            return "general_agent"

        # 根据意图类型路由
        target_agent = INTENT_ROUTING_MAP.get(intent, "general_agent")
        logger.info(f"路由决策: {intent} -> {target_agent}")

        return target_agent

    def process(self, state: AgentState) -> AgentState:
        """
        处理路由逻辑的主入口
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态，包含路由决策
        """
        user_input = state.get("input", "")

        if not user_input.strip():
            logger.warning("用户输入为空")
            state["next_node_to_call"] = "general_agent"
            return state

        try:
            # 分析用户意图（只做意图分类，不提取具体信息）
            intent_result = self.analyze_intent(user_input)

            # 路由到对应智能体
            target_agent = self.route_to_agent(intent_result)
            state["next_node_to_call"] = target_agent

            # 保存意图分析结果到状态中（供其他智能体参考）
            state["intent_analysis"] = intent_result

            logger.info(f"路由完成: {user_input} -> {target_agent} (意图: {intent_result['intent']})")

        except Exception as e:
            logger.error(f"路由处理出错: {e}")
            state["next_node_to_call"] = "general_agent"

        return state


# 创建全局路由智能体实例
# router_agent = RouterAgent(GEMINI_API_KEY)  # 注释掉使用了未定义变量的旧代码
router_agent = RouterAgent()  # 使用无参数的构造函数创建实例


def route_user_input_node(state: AgentState) -> AgentState:
    """
    路由函数入口点，供工作流图调用
    
    Args:
        state: 当前状态
        
    Returns:
        包含路由决策的更新状态
    """
    return router_agent.process(state)