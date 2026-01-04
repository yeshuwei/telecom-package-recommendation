from typing import List, Optional, TypedDict, Dict, Any
from enum import Enum
import logging

try:
    from pydantic import BaseModel
except Exception:
    # 兜底：如果未安装 pydantic，提供极简替代，避免运行时报错
    # # type: ignore类型检查器的注释，表示忽略这一行的类型检查
    class BaseModel:  # type: ignore
        # Any表示允许任何参数类型
        def __init__(self, **data: Any) -> None:
            for k, v in data.items():
                # 为当前对象self新增一个名为k，值为v的新属性
                setattr(self, k, v)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserExplicitNeeds(BaseModel):
    """
    用户明确提出的需求（布尔类型的意图和偏好）
    """

    is_data_overused: Optional[bool] = None
    # 流量是否溢出
    # 示例："我的流量总是不够用"、"我的流量溢出了"

    is_voice_exceeds: Optional[bool] = None
    # 语音时长是否溢出
    # 示例："我的语音时长总是不够用"、"我的语音时长溢出了"

    need_5g: Optional[bool] = None  
    # 是否需要5G套餐
    # 示例："我想办5G"、"不需要5G"、"我手机支持5G"
    
    device_replacement_needs: Optional[bool] = None
    # 用户是否有更换手机的需求
    # 示例："我想买个新手机"、"我用的旧手机"
    
    need_broadband: Optional[bool] = None
    # 是否需要宽带
    # 示例："想办个有宽带的套餐"、"家里有宽带"、"不需要宽带"
    
    need_broadband_upgrade: Optional[bool] = None  
    # 是否需要宽带升级（升千兆）
    # 示例："想升级千兆宽带"、"现在宽带太慢了"
    
    is_family_plan: Optional[bool] = None
    # 是否为家庭办理套餐
    # 示例："有没有家庭类型的套餐"、"我想为我的家庭办理一个套餐"
    
    # === 内容偏好（用户可能明确提出） ===
    video_needs: Optional[bool] = None  
    # 视频需求（布尔型）
    # 示例："我天天刷抖音"→True、"经常看爱奇艺"→True、"不看视频"→False
    
    education_needs: Optional[bool] = None  
    # 教育/学习需求
    # 示例："孩子要上网课"、"需要在线教育"
    
    smart_home_needs: Optional[bool] = None
    # 是否有智能家居需求
    # 示例："我家里有许多智能设备"、"我想让我的家居更加智能化"
    
    # === 特殊身份与权益 ===
    special_identity: Optional[str] = None  
    # 特殊身份,枚举：老年人/军人/残疾人/贫困户/其他
    # 示例："我是学生"、"我是退伍军人"、"有残疾证"、"我家庭条件比较困难"
    
    # === 其他明确需求 ===
    prefer_package_type: Optional[str] = None  
    # 偏好的套餐类型：如"融合套餐"、"单卡套餐"、"流量卡"
    # 示例："我只要流量卡"、"想办融合套餐"

    other_needs: Optional[str] = None  
    # 其他需求（捕获未匹配到具体字段的需求）
    # 存储用户提出但大模型没有匹配到对应字段的需求
    # 示例："我需要国际漫游"、"经常出差"、"需要固定IP"等


class UserDatabaseOverride(BaseModel):
    """
    用户输入可以覆盖的数据库字段
    这些字段原本来自数据库，但用户输入可以覆盖它们
    """
    
    budget: Optional[str] = None
    # 预算
    # 示例："我预算100元"、"每月愿意花50元"
    
    data_needs: Optional[str] = None
    # 流量需求量
    # 示例："轻度"->5、"中度"->30、"重度"->100、"20GB"->20、"30GB->30"
    
    call_minutes: Optional[str] = None
    # 语音通话需求量
    # 示例："每月500分钟"、"基本不打电话"→50
    
    # === 其他可覆盖的数据库字段 ===
    preference_types: Optional[List[str]] = None
    # 偏好类型（列表，必须是数据库中已有的字段）
    # 可选值：直播、游戏、视频、音乐、阅读、教育、智能家居
    # 示例："我喜欢看直播和玩游戏" → ["直播", "游戏"]
    
    user_age: Optional[str] = None
    # 用户年龄段（字符串，枚举值）
    # 可选值："青年"、"中年"、"老年"
    # 映射规则：18-35岁→青年，36-59岁→中年，60岁及以上→老年
    
    family_type: Optional[str] = None
    # 家庭类型（字符串）
    # 可选值："一口之家"、"二口之家"、"三口之家"、"四口之家"、"五口及以上之家"
    # 示例："我和老婆两个人" → "二口之家"
    
    is_5g_device: Optional[bool] = None
    # 是否5G终端（布尔型）
    # 示例："我用的5G手机" → True，"我是4G手机" → False
    
    has_student: Optional[bool] = None
    # 是否家有学生（布尔型）
    # 示例："家里有小孩要上学" → True，"孩子要上网课" → True
    
    data_overflow: Optional[int] = None
    # 流量溢出值（整数，单位：GB）
    # 提取规则：
    # - 用户说了具体数值：直接使用
    # - "溢出不多/轻度溢出" → 10
    # - "溢出较多/中度溢出" → 20
    # - "严重溢出/重度溢出" → 30


# 定义一个字典的结构类型，规定这个字典结构可以包含哪些键值对，total=False代表这些键值对不是必须全部包含
class AgentState(TypedDict, total=False):
    """
    智能体状态
    
    核心数据流：
    1. 用户登录 → phone_number
    2. 无明确需求 → user_info_agent → raw_user_info + db_user_summary
    3. 有明确需求 → slot_filling_agent → user_explicit_needs + user_database_override
    4. 推荐 → recommendation_agent 使用 user_explicit_needs + merged_user_info
    """
    phone_number: str  # 用户电话号码（登录时获取，对应数据库user_data表中的"业务号码"字段）
    input: str  # 用户的最新输入
    intent_analysis: Dict[str, Any]  # 用户意图分析结果
    chat_history: List[str]  # 聊天记录
    
    # === 核心数据（优化方案A：双槽位模式） ===
    user_explicit_needs: UserExplicitNeeds  # 用户明确提出的需求（布尔意图和偏好）
    user_database_override: UserDatabaseOverride  # 用户输入覆盖的数据库字段
    raw_user_info: Dict[str, Any]  # 数据库原始数据（固定，30+字段）
    merged_user_info: Dict[str, Any]  # 合并后的用户信息（优先使用覆盖值）
    db_user_summary: str  # 数据库用户需求的语义总结（AI生成，用于话术）
    
    # === 推荐流程数据 ===
    filtered_package_categories: List[str]  # 筛选后的套餐类别（推荐智能体输出）
    search_results: List[Dict[str, Any]]  # 查询到的套餐
    price_selection_results: List[Dict[str, Any]]  # 价位推荐结果
    scored_package_series: List[Dict[str, Any]]  # 带分数的系列列表（series, score, source）

    # === 知识/比较查询数据 ===
    # selected_package_series: List[str]  # 识别出的套餐系列名称
    # series_details: List[Dict[str, Any]]  # 每个系列下所有价位套餐及属性
    reply_generation_prompt: str  # 传递给回复生成智能体的提示词
    
    # === 流程控制 ===
    final_response: str  # 最终给用户的回复
    next_node_to_call: str  # 路由决策结果


# 数据库字段映射（UserDatabaseOverride字段 → 数据库字段名）
DATABASE_FIELD_MAPPING: Dict[str, str] = {
    # 由于原来的三个必须字段现在改成了提取描述性质的字符串 然后到价位筛选部分的时候再进行映射，所以此处删除了那三个
    "preference_types": "偏好类型",  # 需要特殊处理，可能有多个偏好字段
    "user_age": "用户年龄",
    "family_type": "家庭类型",
    "is_5g_device": "是否5G终端",
    "has_student": "是否家有学生",
    "data_overflow": "流量溢出值(GB)",
}


def format_user_explicit_needs(user_needs: Optional[UserExplicitNeeds]) -> str:
    """
    格式化用户明确需求为可读文本

    Args:
        user_needs: 用户明确需求对象

    Returns:
        格式化的文本
    """
    explicit_needs_dict = {}
    for field_name in dir(user_needs):
        if not field_name.startswith('_'):
            value = getattr(user_needs, field_name, None)
            if value is not None and not callable(value):
                explicit_needs_dict[field_name] = value

    explicit_needs_str = "\n".join(f"-{key}: {value}" for key, value in explicit_needs_dict.items()) if (
        explicit_needs_dict) else "-没有明确需求"

    return explicit_needs_str

def is_needs_sufficient(database_override: Optional[UserDatabaseOverride]) -> bool:
    return True


def update_user_explicit_needs(user_needs: UserExplicitNeeds, extracted_info: Dict[str, Any]) -> UserExplicitNeeds:
    """
    根据提取的信息更新用户明确需求对象
    
    Args:
        user_needs: 当前用户明确需求对象
        extracted_info: 从用户输入中提取的信息
    
    Returns:
        更新后的用户明确需求对象
    """
    # 更新明确需求的各个字段
    for key, value in extracted_info.items():
        if value is not None and hasattr(user_needs, key):
            setattr(user_needs, key, value)
            logger.info(f"更新用户明确需求 {key}: {value}")
    
    return user_needs


def update_database_override(database_override: UserDatabaseOverride, extracted_info: Dict[str, Any]) -> UserDatabaseOverride:
    """
    根据提取的信息更新数据库覆盖字段对象
    
    Args:
        database_override: 当前数据库覆盖字段对象
        extracted_info: 从用户输入中提取的信息
    
    Returns:
        更新后的数据库覆盖字段对象
    """
    # 更新数据库覆盖字段
    for key, value in extracted_info.items():
        if value is not None and hasattr(database_override, key):
            setattr(database_override, key, value)
            logger.info(f"更新数据库覆盖字段 {key}: {value}")
    
    return database_override


def merge_user_info(raw_user_info: Dict[str, Any], database_override: Optional[UserDatabaseOverride]) -> Dict[str, Any]:
    """
    合并数据库原始数据和用户覆盖数据
    优先级: database_override > raw_user_info
    
    Args:
        raw_user_info: 数据库原始数据
        database_override: 用户数据库覆盖字段
    
    Returns:
        合并后的用户信息
    """
    merged_data = raw_user_info.copy() if raw_user_info else {}
    
    if database_override is None:
        return merged_data
    
    # 遍历覆盖字段并更新到合并数据中
    for override_field, db_field in DATABASE_FIELD_MAPPING.items():
        override_value = getattr(database_override, override_field, None)
        if override_value is not None:
            # 特殊处理：偏好类型需要更新多个字段
            if override_field == "preference_types" and isinstance(override_value, list):
                for pref in override_value:
                    pref_field = f"{pref}偏好"
                    merged_data[pref_field] = 4  # 设置为高偏好
                    logger.info(f"字段覆盖: {pref_field} = 4")
            else:
                merged_data[db_field] = override_value
                logger.info(f"字段覆盖: {db_field} = {override_value}")
    
    return merged_data


def update_user_explicit_needs_in_state(state: AgentState, extracted_info: Dict[str, Any]) -> AgentState:
    """
    在状态中更新用户明确需求
    
    Args:
        state: 当前状态
        extracted_info: 从用户输入中提取的信息
    
    Returns:
        更新后的状态
    """
    # 如果用户明确需求不存在，创建新的
    if "user_explicit_needs" not in state or state["user_explicit_needs"] is None:
        state["user_explicit_needs"] = UserExplicitNeeds()
    
    state["user_explicit_needs"] = update_user_explicit_needs(state["user_explicit_needs"], extracted_info)
    
    return state
