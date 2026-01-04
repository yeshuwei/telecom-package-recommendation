"""
知识/比较智能体 - 负责解析用户查询/比较类意图，定位套餐系列并汇总所有价位套餐信息

流程：
1) 收集所有套餐系列名称（package_intro）
2) 让大模型在【系列名称列表 + 用户输入】的基础上，选择最匹配的系列名（可多选）
3) 按系列名查询：
   - package_intro：系列描述
   - tariff_plan：该系列下所有价位 plan（plan_id, plan_name, price）
   - tariff_feature：按 plan_id 获取属性键值对
4) 生成 reply_generation_prompt（供回复生成智能体使用的提示词/上下文）
5) 将 selected_package_series、series_details、reply_generation_prompt 写入状态，并路由到 reply_generation_agent
"""
import logging
from typing import Dict, Any, List, Optional

from agent.state import AgentState
# from configs.config import GEMINI_API_KEY
from mcp_tools.mcp_client import get_mcp_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeComparisonAgent:
    """知识/比较智能体"""

    def __init__(self, gemini_api_key: str = None):
        self.mcp_client = get_mcp_client()
        logger.info("knowledge_comparison_agent 初始化完毕（使用MCP工具）")

    # ======================== 系列匹配 ========================
    def match_series_with_llm(self, user_input: str, all_series: List[str]) -> List[str]:
        """调用LLM在给定系列名称集合中选择最匹配的系列（可多选）"""
        series_list_text = "\n".join(f"- {name}" for name in all_series)
        prompt = f"""你是一个电信套餐知识助手。下面给出所有的套餐系列名称，以及用户输入。
                    你的任务：
                    - 从给定的套餐系列名称中，选择与用户输入最相关的系列名称；
                    - 用户可能提出比较需求或同时查看多个系列，请保留多个匹配（最多5个）；
                    - 只能从给定的系列名称中选择，不要生成列表里不存在的名称；
                    - 如果完全无法判断，返回空数组。
                    
                    【所有套餐系列名称】
                    {series_list_text}
                    
                    【用户输入】
                    {user_input}
                    
                    只返回JSON：
                    {{
                      "selected_series": ["系列1", "系列2", ...]
                    }}"""
        try:
            result = self.mcp_client.generate_and_extract_json(prompt, is_array=False, temperature=0.1)
            if result and isinstance(result.get("selected_series"), list):
                selected = [s for s in result["selected_series"] if s in all_series]
                logger.info(f"LLM 匹配到系列：{selected}")
                return selected
            logger.warning("LLM 未能返回有效的selected_series，改用规则匹配")
            return self.rule_match_series(user_input, all_series)
        except Exception as e:
            logger.error(f"LLM 匹配系列失败：{e}，改用规则匹配")
            return self.rule_match_series(user_input, all_series)

    def rule_match_series(self, user_input: str, all_series: List[str]) -> List[str]:
        """简易规则匹配：子串/分词包含，按出现顺序取前5个"""
        text = user_input.lower()
        hits: List[str] = []
        for name in all_series:
            n = name.lower()
            if n in text or any(tok and tok in n for tok in text.replace('、', ' ').replace('，', ' ').split()):
                hits.append(name)
        # 去重保序
        seen = set()
        ordered = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                ordered.append(h)
        logger.info(f"规则匹配到系列：{ordered[:5]}")
        return ordered[:5]

    # ======================== 数据查询与组装 ========================
    def fetch_series_bundle(self, series_name: str) -> Optional[Dict[str, Any]]:
        """查询一个系列的简介、全部价位与特性，并组装为结构化字典"""
        try:
            intro = self.mcp_client.mysql_tools.query_package_intro_by_series(series_name)
            if not intro:
                logger.warning(f"未找到系列简介：{series_name}")
                return None
            plans = self.mcp_client.mysql_tools.query_plans_by_series(series_name)
            plan_items: List[Dict[str, Any]] = []
            for p in plans:
                plan_id = p.get("plan_id")
                features = self.mcp_client.mysql_tools.query_features_by_plan_id(plan_id) if plan_id is not None else {}
                plan_items.append({
                    "plan_id": plan_id,
                    "plan_name": p.get("plan_name"),
                    "price": p.get("price"),
                    "features": features
                })
            bundle = {
                "series": intro.get("package_name"),
                "description": intro.get("description", ""),
                "plans": plan_items
            }
            return bundle
        except Exception as e:
            logger.error(f"查询系列明细失败：{series_name}, {e}")
            return None

    # ======================== 提示词生成 ========================
    def build_reply_generation_prompt(self, user_input: str, series_details: List[Dict[str, Any]]) -> str:
        """生成可提供给回复生成智能体使用的提示词/上下文"""
        lines: List[str] = []
        lines.append("你是电信套餐知识答复助手。下面提供了用户需求与候选套餐系列的全量信息，请据此生成面向用户的回答。")
        lines.append("")
        lines.append(f"【用户输入】\n{user_input}")
        lines.append("")
        lines.append("【候选套餐系列明细】")
        for i, s in enumerate(series_details, 1):
            lines.append(f"- 系列{i}: {s.get('series','')}\n  描述: {s.get('description','').strip()}")
            if s.get("plans"):
                for p in s["plans"]:
                    lines.append(f"  • 套餐: {p.get('plan_name','')} | 价格: {p.get('price','')} 元")
                    feats = p.get("features", {})
                    if feats:
                        feat_str = "; ".join([f"{k}: {v}" for k, v in feats.items()])
                        lines.append(f"    属性: {feat_str}")
        lines.append("")
        lines.append("【答复要求】\n- 若是查询：简明介绍所列系列与各价位差异；\n- 若是比较：对比用户点名的系列，突出差异项（价格/流量/权益等）；\n- 保持客观准确，避免虚构；\n- 若系列较多，先概览后分点列出；")
        return "\n".join(lines)

    # ======================== 主流程 ========================
    def process(self, state: AgentState) -> AgentState:
        user_input = state.get("input", "").strip()
        if not user_input:
            state["final_response"] = "请提供您要查询或对比的套餐名称/系列。"
            state["next_node_to_call"] = "end"
            return state

        # 1) 取全量系列名
        all_series = self.mcp_client.mysql_tools.query_all_series_names()
        if not all_series:
            state["final_response"] = "系统暂未配置套餐系列信息。"
            state["next_node_to_call"] = "end"
            return state

        # 2) 选择最匹配系列（可多选）
        selected = self.match_series_with_llm(user_input, all_series)
        # 若依然为空，直接结束
        if not selected:
            state["final_response"] = "没有匹配到相关的套餐系列，您可以换个说法或提供更具体的系列名称。"
            state["next_node_to_call"] = "end"
            return state

        # 3) 查询各系列详细
        bundles: List[Dict[str, Any]] = []
        for name in selected:
            b = self.fetch_series_bundle(name)
            if b:
                bundles.append(b)
        if not bundles:
            state["final_response"] = "未查询到相关套餐的详细信息。"
            state["next_node_to_call"] = "end"
            return state

        # 4) 生成提示词
        reply_prompt = self.build_reply_generation_prompt(user_input, bundles)

        # 5) 写入状态并路由
        # state["selected_package_series"] = selected
        # state["series_details"] = bundles
        state["reply_generation_prompt"] = reply_prompt
        state["next_node_to_call"] = "reply_generation_agent"
        state["final_response"] = "好的，已为您汇总相关套餐系列的详细信息，正在组织解答…"
        return state


# 全局实例与节点入口
# knowledge_comparison_agent = KnowledgeComparisonAgent(GEMINI_API_KEY)



def knowledge_comparison_agent_node(state: AgentState) -> AgentState:
    return knowledge_comparison_agent.process(state)

