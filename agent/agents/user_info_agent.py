"""
用户信息查询智能体 - 负责从MySQL数据库查询用户信息并总结用户需求
"""
import logging
from typing import Dict, Any, Optional
import pymysql

from agent.state import AgentState
# from configs.config import GEMINI_API_KEY
from mcp_tools.mcp_client import get_mcp_client

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserInfoAgent:
    """用户信息查询智能体类"""

    def __init__(self, gemini_api_key: str = None):
        """
        初始化用户信息查询智能体
        Args:
            gemini_api_key: Gemini的API密钥
        """
        # 使用 MCP 客户端
        self.mcp_client = get_mcp_client()

        logger.info("user_info_agent初始化完毕（使用MCP工具）")

    def summarize_user_needs(self, user_info: Dict[str, Any]) -> str:
        """
        使用大模型将用户信息总结为用户需求
        Args:
            user_info: 用户信息字典
        Returns:
            总结的用户需求描述
        """
        logger.info("开始使用大模型总结用户需求")

        # 构建用户信息描述
        info_lines = []
        for key, value in user_info.items():
            if key not in ['id', 'created_at'] and value is not None:
                info_lines.append(f"- {key}: {value}")

        user_info_text = "\n".join(info_lines)

        # 构建提示词
        prompt = f"""你是一个专业的电信套餐客服分析师。以下是一位用户的详细信息：
                    
                    {user_info_text}
                    
                    请基于以上用户信息，分析并总结该用户的核心需求和偏好。
                    
                    要求：
                    1. 重点关注用户的预算范围、流量需求、通话习惯等关键信息
                    2. 识别用户的使用场景和偏好（如视频、游戏、社交等）
                    3. 注意用户的特殊身份或需求（如学生、老年人、家庭用户等）
                    4. 总结要简洁明了，控制在150字以内
                    5. 用自然的语言表达，便于推荐套餐智能体的理解和使用
                    
                    请直接给出总结结果，不要有其他多余内容。"""

        try:
            # 使用 MCP 工具生成内容
            summary = self.mcp_client.generate_content(prompt, temperature=0.7)

            if summary:
                summary = summary.strip()
                logger.info(f"成功生成用户需求总结: {summary[:50]}...")
                return summary
            else:
                logger.warning("大模型返回为空，使用默认总结")
                return self._generate_default_summary(user_info)

        except Exception as e:
            logger.error(f"调用大模型失败: {e}，使用默认总结")
            return self._generate_default_summary(user_info)

    def _generate_default_summary(self, user_info: Dict[str, Any]) -> str:
        """
        生成默认的用户需求总结（降级方案）
        Args:
            user_info: 用户信息字典
        Returns:
            默认总结
        """
        summary_parts = []

        # 提取关键信息
        if '年龄' in user_info and user_info['年龄']:
            summary_parts.append(f"{user_info['年龄']}岁")

        if '性别' in user_info and user_info['性别']:
            summary_parts.append(f"{user_info['性别']}性用户")

        if '月均消费' in user_info and user_info['月均消费']:
            summary_parts.append(f"月均消费{user_info['月均消费']}元")

        # 识别偏好
        preferences = []
        for key, value in user_info.items():
            if '偏好' in key and value and str(value) not in ['0', '0.0', 'None']:
                preferences.append(key.replace('偏好', ''))

        if preferences:
            summary_parts.append(f"偏好{'/'.join(preferences[:3])}")

        if summary_parts:
            return "该用户是" + "，".join(summary_parts) + "的用户。"
        else:
            return "该用户信息有限，建议通过对话进一步了解需求。"

    def process(self, state: AgentState) -> AgentState:
        """
        处理用户信息查询的主入口（无明确需求流程）
        
        功能：
        1. 查询数据库获取用户原始数据（raw_user_info）
        2. 用原始数据生成语义总结（db_user_summary）
        3. 直接传递给推荐智能体（不再提取结构化画像）
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 从用户状态中提取电话号码（登录时获取）
        phone_number = state.get("phone_number")

        if not phone_number:
            # 异常情况：用户未登录或电话号码丢失
            state["final_response"] = "系统错误：未获取到用户信息，请重新登录"
            state["next_node_to_call"] = "end"
            logger.error("phone_number not found in state")
            return state

        logger.info(f"开始查询电话号码 {phone_number} 的用户数据库信息")

        # 查询数据库获取原始用户信息（通过业务号码查询）
        raw_user_info = self.mcp_client.mysql_tools.query_cursor_by_phone_number(phone_number)

        if raw_user_info is None:
            # 数据库中没有该用户的历史数据
            # 降级策略：引导到槽位填充流程
            logger.warning(f"数据库中无电话号码 {phone_number} 的用户历史数据，降级到槽位填充")
            state["next_node_to_call"] = "slot_filling_agent"
            state["final_response"] = "您好！系统中暂无您的历史使用数据。\n为了给您更好的推荐，请告诉我您的需求（预算、流量、通话等）"
            return state

        logger.info(f"成功查询到电话号码 {phone_number} 的用户数据，字段数: {len(raw_user_info)}")

        # 用原始数据生成语义总结（保留丰富的偏好信息）
        user_needs_summary = self.summarize_user_needs(raw_user_info)
        logger.info(f"生成用户需求总结: {user_needs_summary[:50]}...")

        # 更新状态，传递给推荐智能体
        state["raw_user_info"] = raw_user_info  # 数据库原始数据（固定属性）
        state["merged_user_info"] = raw_user_info  # 无明确需求场景，merged等于raw
        state["db_user_summary"] = user_needs_summary  # 语义总结（用于话术）
        # user_explicit_needs 保持为空（无明确需求场景）
        
        state["final_response"] = f"根据您的使用习惯分析：\n{user_needs_summary}\n\n正在为您推荐合适的套餐..."
        state["next_node_to_call"] = "recommendation_agent"

        logger.info(f"成功处理电话号码 {phone_number} 的用户信息查询，准备推荐")
        # logger.info(f"数据库数据: 预算={raw_user_info.get('近三月平均消费(套餐级)')}, "
        #            f"流量={raw_user_info.get('前三月流量平均消耗')}, "
        #            f"星级={raw_user_info.get('星级服务等级（客户级）')}")

        return state


# # 创建全局用户信息查询智能体实例
user_info_agent = UserInfoAgent()


def user_info_node(state: AgentState) -> AgentState:
    """
    用户信息查询节点入口函数
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    return user_info_agent.process(state)
