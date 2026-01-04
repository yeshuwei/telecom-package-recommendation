"""
工作流图 - 定义整个对话系统的状态流转
"""
from typing import Dict, Any
import logging

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.agents.price_selection_agent import price_selection_node
from agent.agents.router_agent import route_user_input_node
from agent.agents.slot_filling_agent import slot_filling_node
from agent.agents.user_info_agent import user_info_node
from agent.agents.recommendation_agent import recommendation_node
from agent.agents.response_generation_agent import ResponseGenerationAgent
from agent.agents.knowledge_comparison_agent import knowledge_comparison_agent_node

# 初始化回复生成智能体
response_generation_agent = ResponseGenerationAgent()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def response_generation_node(state: AgentState) -> AgentState:
    """
    回复生成节点 - 生成最终的推荐回复
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    return response_generation_agent.process(state)

def general_agent_node(state: AgentState) -> AgentState:
    """
    通用智能体节点 - 处理闲聊和通用请求
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    logger.info("进入通用智能体节点")
    
    user_input = state.get("input", "")
    
    # 简单的闲聊回复
    if "你好" in user_input or "您好" in user_input:
        response = "您好！我是电信套餐智能客服，很高兴为您服务。请问有什么可以帮您的吗？"
    elif "谢谢" in user_input:
        response = "不客气！如果还有其他问题，随时可以问我哦。"
    elif "再见" in user_input:
        response = "再见！祝您生活愉快！"
    else:
        response = "我是电信套餐智能客服，可以帮您推荐套餐、查询套餐信息或对比套餐。请问需要什么帮助呢？"
    
    state["final_response"] = response
    state["next_node_to_call"] = "END"
    
    return state


def entry_point_router(state: AgentState) -> AgentState:
    """
    入口点路由函数 - 检查是否需要直接进入槽位填充模式
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """

    logger.info("进入正常路由流程")
    state["next_node_to_call"] = "router"
    
    return state


def entry_conditional_edges(state: AgentState) -> str:
    """
    入口点的条件边函数
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next_node_to_call", "router")
    logger.info(f"入口点决策: {next_node}")
    return next_node


def router_conditional_edges(state: AgentState) -> str:
    """
    路由器的条件边函数
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next_node_to_call", "general_agent")
    logger.info(f"路由决策: {next_node}")
    return next_node


# 构建工作流图
workflow = StateGraph(AgentState)

# 添加所有节点
workflow.add_node("entry_router", entry_point_router)  # 入口路由节点
workflow.add_node("router", route_user_input_node)
workflow.add_node("slot_filling_agent", slot_filling_node)
workflow.add_node("user_info_agent", user_info_node)
workflow.add_node("recommendation_agent", recommendation_node)
workflow.add_node("price_selection_agent", price_selection_node)
workflow.add_node("response_generation", response_generation_node)
workflow.add_node("knowledge_comparison_agent", knowledge_comparison_agent_node)
workflow.add_node("general_agent", general_agent_node)

# 设置入口点为 entry_router（会智能判断是否需要跳过路由）
workflow.set_entry_point("entry_router")

# 添加入口路由的条件边
workflow.add_conditional_edges(
    "entry_router",
    entry_conditional_edges,
    {
        "router": "router",
        "slot_filling_agent": "slot_filling_agent",
    }
)

# 添加路由器的条件边
workflow.add_conditional_edges(
    "router",
    router_conditional_edges,
    {
        "slot_filling_agent": "slot_filling_agent",
        "user_info_agent": "user_info_agent",
        "knowledge_comparison_agent": "knowledge_comparison_agent",
        "general_agent": "general_agent",
    }
)


def recommendation_conditional_edges(state: AgentState) -> str:
    next_node = state.get("next_node_to_call", "END")
    return next_node if next_node in ["price_selection_agent", "END"] else "END"

workflow.add_conditional_edges(
    "recommendation_agent",
    recommendation_conditional_edges,
    {
        "price_selection_agent": "price_selection_agent",
        END: END
    }
)

def price_selection_conditional_edges(state: AgentState) -> str:
    next_node = state.get("next_node_to_call", "END")
    return next_node if next_node in ["response_generation", "END"] else "END"

workflow.add_conditional_edges(
    "price_selection_agent",
    price_selection_conditional_edges,
    {
        "response_generation": "response_generation",
        END: END
    }
)

# 节点到推荐节点
workflow.add_edge("user_info_agent", "recommendation_agent")
workflow.add_edge("slot_filling_agent", "recommendation_agent")

# 回复生成节点到END
workflow.add_edge("response_generation", END)

# 所有结束节点到END
workflow.add_edge("knowledge_comparison_agent", "response_generation")
workflow.add_edge("general_agent", "response_generation")

# 编译工作流
app = workflow.compile()

logger.info("工作流图构建完成")
