"""
槽填充智能体 - 负责收集用户明确需求并填充关键槽位（优化方案A：按需双提取）
"""
import json
import re
from typing import Dict, Any, List, Optional
import logging
import os

from agent.state import (
    AgentState,
    UserExplicitNeeds,
    UserDatabaseOverride,
    update_user_explicit_needs,
    update_database_override,
    merge_user_info,
    format_user_explicit_needs
)
# 移除未使用的GEMINI_API_KEY导入
# from configs.config import GEMINI_API_KEY
from mcp_tools.mcp_client import get_mcp_client

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 已移除必须槽位中文名称映射


class SlotFillingAgent:
    """槽填充智能体类（优化方案A：按需双提取）"""

    def __init__(self, gemini_api_key: str = None):
        """
        初始化槽填充智能体
        Args:
            gemini_api_key: Gemini的API密钥（目前未使用）
        """
        # 使用 MCP 客户端 - 基于项目中其他智能体的实现模式
        self.mcp_client = get_mcp_client()
        logger.info("slot_filling_agent初始化完毕（使用MCP工具，按需双提取模式）")

    def extract_explicit_needs(self, user_input: str) -> Dict[str, Any]:
        """
        第一次LLM调用：从用户输入中提取用户明确需求（布尔意图和偏好）
        
        依据：
        1. 项目设计文档中定义的"按需双提取"模式的第一阶段
        2. 需要提取布尔类型的意图和偏好，用于后续推荐决策
        3. 遵循信息提取的最佳实践，使用结构化提示词确保提取结果的准确性
        
        Args:
            user_input: 用户输入
        Returns:
            提取的明确需求信息字典
        """
        logger.info(f"【第1次提取】开始提取用户明确需求（布尔意图）: {user_input}")

        # 构建提示词 - 基于业务需求和信息提取的最佳实践设计
        prompt = f"""你是一个电信套餐客服助手的信息提取模块。请从用户输入中精确提取**明确的意图和偏好**。

【用户输入】
{user_input}

【提取任务】
请分析用户输入，提取以下**布尔类型的意图和偏好字段**（如果用户没有明确提到则返回null）：

**返回JSON格式：**
{{
    "is_data_overused": 流量是否溢出(布尔)或null,
    "is_voice_exceeds": 语音时长是否溢出(布尔)或null,
    "need_5g": 是否需要5G套餐(布尔)或null,
    "device_replacement_needs": 是否有更换手机需求(布尔)或null,
    "need_broadband": 是否需要宽带(布尔)或null,
    "need_broadband_upgrade": 是否需要宽带升级(布尔)或null,
    "is_family_plan": 是否为家庭办理套餐(布尔)或null,
    "video_needs": 视频需求(布尔)或null,
    "education_needs": 教育学习需求(布尔)或null,
    "smart_home_needs": 智能家居需求(布尔)或null,
    "special_identity": 特殊身份(字符串)或null,
    "prefer_package_type": 偏好套餐类型(字符串)或null,
    "other_needs": 其他需求(字符串)或null
}}

【提取规则】
1. **is_data_overused** (布尔)
   - "流量总是不够用/流量溢出/流量超了" → true
   - "流量够用" → false

2. **is_voice_exceeds** (布尔)
   - "通话时长不够/语音溢出" → true
   - "通话够用" → false

3. **need_5g** (布尔)
   - "想办5G/需要5G/升级5G" → true
   - "不需要5G/4G够用" → false

4. **device_replacement_needs** (布尔)
   - "想换手机/买新手机" → true
   - "手机还能用/不换手机" → false

5. **need_broadband** (布尔)
   - "需要宽带/想装宽带" → true
   - "不需要宽带/已有宽带" → false

6. **need_broadband_upgrade** (布尔)
   - "升级千兆/宽带太慢" → true
   - "宽带够用" → false

7. **is_family_plan** (布尔)
   - "家庭套餐/全家一起用" → true
   - "个人使用" → false

8. **video_needs** (布尔)
   - "天天刷抖音/经常看视频/爱看爱奇艺" → true
   - "不看视频" → false

9. **education_needs** (布尔)
   - "孩子要上网课/需要在线教育" → true

10. **smart_home_needs** (布尔)
    - "家里有智能设备/智能家居" → true

11. **special_identity** (字符串，枚举值)
    - 可选值："老年人"、"军人"、"军人家属"、"残疾人"、"贫困户"、"其他"
    - 示例："我是退伍军人" → "军人"，无法归类到上面四个类别的分到"其他"

12. **prefer_package_type** (字符串)
    - 将用户的目标套餐名称提取出来
    - 示例："只要流量卡" → "流量卡"

13. **other_needs** (字符串，捕获其他需求)
    - 用户提到但以上字段未覆盖的需求
    - 示例："需要国际漫游" → "国际漫游"

【重要提示】
- 只提取用户**明确表达**的意图，不要推测
- 如果用户没有明确提到某个字段，返回null
- 布尔字段只在用户明确表达肯定或否定时才返回true/false
- 保持JSON格式严格有效
- **不要提取**预算、流量需求量、通话需求量等具体数值

请返回提取结果的JSON："""

        try:
            # 使用 MCP 工具生成并提取 JSON - 基于项目中统一的LLM调用方式
            extracted_info = self.mcp_client.generate_and_extract_json(prompt, is_array=False, temperature=0.1)

            if extracted_info:
                logger.info("【第1次提取】成功调用API提取明确需求")
                # 过滤掉null值，只返回有效提取的信息 - 减少数据传输和处理成本
                valid_info = {k: v for k, v in extracted_info.items() if v is not None}
                return valid_info
            else:
                logger.warning("【第1次提取】API返回结果为空")
                return {}
        except Exception as e:
            logger.error(f"【第1次提取】调用API时发生错误: {e}")
            return {}

    def should_extract_override(self, user_input: str) -> bool:
        """
        判断是否需要进行第二次提取（数据库覆盖字段）
        
        依据：
        1. "按需双提取"模式的核心设计，避免不必要的LLM调用
        2. 只有当用户输入包含具体数值或描述性信息时才进行第二次提取
        3. 减少API调用次数，提高系统响应速度，降低成本
        
        Args:
            user_input: 用户输入
        Returns:
            是否需要第二次提取
        """
        user_input = user_input.lower()
        
        # 规则：用户提到了具体的数值或描述相关词汇时需要第二次提取
        # 基于业务经验，这些词汇通常包含需要提取的数据库覆盖字段信息
        if any(keyword in user_input for keyword in ["元", "预算", "流量", "通话", "分钟", "gb", "g"]):
            logger.info("判断：用户提到了具体数值或相关描述 → 需要第二次提取")
            return True
            
        return False

    def extract_database_override(self, user_input: str) -> Dict[str, Any]:
        """
        第二次LLM调用：从用户输入中提取数据库覆盖字段（包含三个必须槽位）
        
        依据：
        1. 项目设计文档中定义的"按需双提取"模式的第二阶段
        2. 需要提取可以覆盖数据库信息的字段，用于后续推荐决策
        3. 提取更详细的用户需求信息，如预算、流量需求、通话需求等
        
        Args:
            user_input: 用户输入
        Returns:
            提取的数据库覆盖字段信息字典
        """
        logger.info(f"【第2次提取】开始提取数据库覆盖字段: {user_input}")

        # 构建提示词 - 基于业务需求和信息提取的最佳实践设计
        prompt = f"""你是一个电信套餐客服助手的信息提取模块。请从用户输入中精确提取**可以覆盖数据库的字段信息**（以简短的中文描述字符串返回）。

【用户输入】
{user_input}

【提取任务】
请分析用户输入，提取以下**数据库可覆盖字段**（如果用户没有提到则返回null）：

**返回JSON格式：**
{{
    "budget": 预算相关描述(字符串)或null,
    "data_needs": 流量使用相关描述(字符串)或null,
    "call_minutes": 通话习惯相关描述(字符串)或null,
    "preference_types": 偏好类型列表(数组)或null,
    "user_age": 用户年龄段(字符串)或null,
    "family_type": 家庭类型(字符串)或null,
    "is_5g_device": 是否5G终端(布尔)或null,
    "has_student": 是否家有学生(布尔)或null,
    "data_overflow": 流量溢出值(整数,单位GB)或null
}}

【提取规则】

对于 budget / data_needs / call_minutes：
- 如果用户提到了具体的数值，那么就优先提取具体数值加单位
- 如果用户只是进行描述，不涉及对应的具体数值，返回简短中文描述字符串，例如："预算偏低"、"每月看视频较多"、"通话不多" 等
  形容词尽量从下面这部分中去选择：“轻度|偏低|不多|较少”， “中度|一般|正常”， “重度|较多|很多|无限|不限”
- 如果用户没有明确表述，返回 null

**其他可覆盖字段：**

4. **preference_types** (数组，字符串列表)
   - **必须从以下枚举值中选择**：["直播", "游戏", "视频", "音乐", "阅读", "教育", "智能家居", "短视频"]
   - "我喜欢看直播和玩游戏" → ["直播", "游戏"]
   - "天天刷抖音" → ["短视频"]
   - "经常看视频和听音乐" → ["视频", "音乐"]
   - "孩子要上网课" → ["教育"]

5. **user_age** (字符串，枚举值)
   - **必须从以下枚举值中选择**：["青年", "中年", "老年"]
   - 映射规则：
     * 18-35岁 或 "年轻" → "青年"
     * 36-59岁 或 "中年人" → "中年"
     * 60岁及以上 或 "老人/老年人" → "老年"
   - 示例："我今年65岁" → "老年"

6. **family_type** (字符串，枚举值)
   - **必须从以下枚举值中选择**：["一口之家", "二口之家", "三口之家", "四口之家", "五口及以上之家"]
   - 示例：
     * "我一个人住" → "一口之家"
     * "我和老婆" → "二口之家"
     * "三口之家" → "三口之家"
     * "我们家四个人" → "四口之家"

7. **is_5g_device** (布尔)
   - "我用的5G手机/手机支持5G" → true
   - "4G手机/没有5G手机" → false

8. **has_student** (布尔)
   - "家里有小孩要上学/孩子要上网课" → true
   - 注意：如果已提取的明确需求中有education_needs=true，则应该返回true

9. **data_overflow** (整数，单位：GB)
   - 提取规则：
     * 用户说了具体数值："溢出了15GB" → 15
     * "溢出不多/轻度溢出" → 10
     * "溢出较多/中度溢出" → 20
     * "严重溢出/重度溢出" → 30

【重要提示】
- 只提取用户**明确提到**的信息，不要过度推测
- 如果用户没有提到某个字段，返回null
- preference_types、user_age、family_type 的值**必须严格从枚举值中选择**
- preference_types 是数组类型，即使只有一个值也要返回数组格式：["视频"]，没有提取到返回null即可

请返回提取结果的JSON："""

        try:
            # 使用 MCP 工具生成并提取 JSON - 基于项目中统一的LLM调用方式
            extracted_info = self.mcp_client.generate_and_extract_json(prompt, is_array=False, temperature=0.1)

            if extracted_info:
                logger.info("【第2次提取】成功调用API提取覆盖字段")
                # 过滤掉null值，只返回有效提取的信息 - 减少数据传输和处理成本
                valid_info = {k: v for k, v in extracted_info.items() if v is not None}
                return valid_info
            else:
                logger.warning("【第2次提取】API返回结果为空")
                return {}
        except Exception as e:
            logger.error(f"【第2次提取】调用API时发生错误: {e}")
            return {}

    def query_database_info(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        从数据库中查询用户信息
        
        依据：
        1. 业务需求需要获取用户的历史数据和属性信息
        2. 用于与用户输入的信息进行合并，生成更全面的用户画像
        3. 遵循数据查询的最佳实践，提供默认返回值处理异常情况
        
        Args:
            phone_number: 用户手机号
        Returns:
            用户信息字典
        """
        # 这里应该是实际的数据库查询逻辑
        # 暂时返回空，实际项目中需要实现与数据库的交互
        # 依据项目架构，应该调用专门的数据访问层或数据库客户端
        logger.info(f"查询用户 {phone_number} 的数据库信息")
        return {}

    def generate_mixed_summary(self, explicit_needs: UserExplicitNeeds, merged_user_info: Dict[str, Any], database_override: UserDatabaseOverride) -> str:
        """
        生成用户需求和属性的混合总结
        
        依据：
        1. 需要将提取的用户需求和属性信息整合成简洁的总结
        2. 用于后续推荐智能体的输入，提高推荐的准确性
        3. 遵循自然语言生成的最佳实践，确保总结的准确性和可读性
        
        Args:
            explicit_needs: 用户明确需求
            merged_user_info: 合并后的用户信息
            database_override: 数据库覆盖字段
        Returns:
            混合总结
        """
        # 格式化明确需求 - 使用项目中统一的格式化函数
        explicit_needs_str = format_user_explicit_needs(explicit_needs)
        explicit_needs_str = explicit_needs_str if explicit_needs_str else "-未提取到明确需求"

        # 格式化合并后的用户信息（只显示关键字段）
        merged_info_str = "\n".join(
            f"-{key}: {value}" for key, value in merged_user_info.items()
        ) if merged_user_info else "-未查询到用户历史数据"

        # 构建提示词 - 基于业务需求和总结生成的最佳实践设计
        prompt = f"""你是一个电信套餐推荐助手。请根据用户的明确需求、覆盖字段和历史数据，生成一个简洁的需求总结。

用户明确需求（优先级高，布尔意图）：
{explicit_needs_str}

用户属性信息（已合并覆盖值）：
{merged_info_str}

请生成一个150字以内的用户需求和属性总结，尽可能的做到概括和全面。

总结要自然、连贯，便于推荐智能体理解用户整体情况和需求。"""

        try:
            # 使用 MCP 工具生成内容 - 基于项目中统一的LLM调用方式
            summary = self.mcp_client.generate_content(prompt, temperature=0.3)

            if summary:
                logger.info(f"生成混合总结成功: {summary[:100]}...")
                return summary.strip()
            else:
                logger.warning("生成总结失败，使用默认总结")
                # 降级处理：使用默认总结生成方法
                return self._generate_default_mixed_summary(database_override, merged_user_info)

        except Exception as e:
            logger.error(f"生成总结时出错: {e}")
            # 降级处理：使用默认总结生成方法
            return self._generate_default_mixed_summary(database_override, merged_user_info)

    def _generate_default_mixed_summary(
            self,
            database_override: Optional[UserDatabaseOverride],
            merged_user_info: Dict[str, Any]
    ) -> str:
        """
        生成默认的混合总结
        
        依据：
        1. 容错设计，当LLM调用失败时提供备选方案
        2. 确保系统在异常情况下仍能提供有意义的输出
        3. 基于业务需求，只包含最关键的用户信息
        
        Args:
            database_override: 数据库覆盖字段
            merged_user_info: 合并后的用户信息
        Returns:
            默认混合总结
        """
        parts = []

        # 数据库覆盖字段 - 优先级高
        if database_override:
            if database_override.budget:
                parts.append(f"用户预算：{database_override.budget}")
            if database_override.data_needs:
                parts.append(f"流量需求：{database_override.data_needs}")
            if database_override.call_minutes:
                parts.append(f"通话需求：{database_override.call_minutes}")

        # 合并后的用户属性
        if merged_user_info.get('用户年龄'):
            parts.append(f"{merged_user_info['用户年龄']}用户")
        if merged_user_info.get('家庭类型'):
            parts.append(f"{merged_user_info['家庭类型']}")

        return "，".join(parts) + "。" if parts else "用户信息不足。"

    def process(self, state: AgentState) -> AgentState:
        """
        处理槽填充逻辑的主入口 - 优化方案A：按需双提取
        
        依据：
        1. 项目架构要求每个智能体都有统一的process方法作为入口
        2. slot_filling_node函数调用了此方法，之前的版本中缺少导致AttributeError
        3. 负责协调所有其他方法的执行，实现完整的槽填充流程
        
        Args:
            state: 当前状态
        Returns:
            更新后的状态
        """
        # 从状态中获取必要的信息
        user_input = state.get("input", "")
        user_explicit_needs = state.get("user_explicit_needs")
        user_database_override = state.get("user_database_override")
        phone_number = state.get("phone_number")

        # 初始化用户明确需求对象（如果不存在）
        if user_explicit_needs is None:
            user_explicit_needs = UserExplicitNeeds()
            state["user_explicit_needs"] = user_explicit_needs

        # 初始化数据库覆盖字段对象（如果不存在）
        if user_database_override is None:
            user_database_override = UserDatabaseOverride()
            state["user_database_override"] = user_database_override

        # 从用户输入中提取信息
        if user_input.strip():
            # ===【第一次LLM调用】提取明确需求（布尔意图）===
            extracted_explicit = self.extract_explicit_needs(user_input)
            if extracted_explicit:
                # 更新用户明确需求 - 使用项目中统一的更新函数
                user_explicit_needs = update_user_explicit_needs(user_explicit_needs, extracted_explicit)
                state["user_explicit_needs"] = user_explicit_needs

                # 记录有效提取的字段数量 - 用于监控和调试
                valid_fields = [
                    k for k in dir(user_explicit_needs)
                    if not k.startswith('_')
                    and not callable(getattr(user_explicit_needs, k))
                    and getattr(user_explicit_needs, k) is not None
                ]
                logger.info(f"更新后的用户明确需求字段: {len(valid_fields)} 个 - {extracted_explicit}")

            # ===【按需判断】是否需要第二次提取===
            if self.should_extract_override(user_input):
                # ===【第二次LLM调用】提取数据库覆盖字段===
                extracted_override = self.extract_database_override(user_input)
                if extracted_override:
                    # 更新数据库覆盖字段 - 使用项目中统一的更新函数
                    user_database_override = update_database_override(user_database_override, extracted_override)
                    state["user_database_override"] = user_database_override
                    logger.info(f"更新后的数据库覆盖字段: budget={user_database_override.budget}, "
                                f"data_needs={user_database_override.data_needs}, "
                                f"call_minutes={user_database_override.call_minutes}")
            else:
                logger.info("本轮不需要第二次提取，继续使用现有的数据库覆盖字段")

        # 直接查询并合并，不再进行必须槽位检查与追问 - 基于项目设计的简化流程
        if not phone_number:
            # 如果没有手机号，直接进入推荐环节
            state["next_node_to_call"] = "recommendation_agent"
            state["final_response"] = "好的，正在为您推荐合适的套餐..."
            return state

        # 查询用户数据库信息
        raw_user_info = self.query_database_info(phone_number)

        if raw_user_info:
            # 合并用户信息 - 使用项目中统一的合并函数
            merged_user_info = merge_user_info(raw_user_info, user_database_override)
            # 生成混合总结
            db_user_summary = self.generate_mixed_summary(
                user_explicit_needs,
                merged_user_info,
                user_database_override
            )

            # 更新状态
            state["raw_user_info"] = raw_user_info
            state["merged_user_info"] = merged_user_info
            state["db_user_summary"] = db_user_summary
            logger.info(f"混合总结: {db_user_summary}")
        else:
            # 如果没有查询到用户信息，使用提取的覆盖字段
            state["merged_user_info"] = {}
            state["db_user_summary"] = (f"用户预算：{user_database_override.budget}，"
                                        f"流量需求：{user_database_override.data_needs}，"
                                        f"通话需求：{user_database_override.call_minutes}")

        # 设置下一个节点和响应信息
        state["next_node_to_call"] = "recommendation_agent"
        state["final_response"] = "好的，我已经了解您的需求，正在为您推荐合适的套餐..."

        return state


# 创建全局槽填充智能体实例 - 不使用GEMINI_API_KEY（依据：该参数未被使用）
slot_filling_agent = SlotFillingAgent()


def slot_filling_node(state: AgentState) -> AgentState:
    """
    槽填充节点入口函数
    
    依据：
    1. 项目的工作流架构要求每个智能体提供一个节点入口函数
    2. 用于工作流引擎调用槽填充智能体的process方法
    3. 遵循函数式编程的最佳实践，提供统一的接口
    
    Args:
        state: 当前状态
    Returns:
        更新后的状态
    """
    # 调用槽填充智能体的process方法处理请求
    return slot_filling_agent.process(state)