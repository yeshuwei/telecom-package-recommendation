"""
推荐智能体 - 基于RAG的两阶段套餐推荐

推荐流程：
1. 使用 db_user_summary 在 Milvus 向量数据库中检索相关推荐原则
2. 提取推荐的套餐类别
3. 基于 user_explicit_needs 和 raw_user_info 在类别内筛选具体价位的套餐
"""
import logging
from typing import Dict, Any, List, Optional
from agent.state import AgentState, UserExplicitNeeds, format_user_explicit_needs
# from configs.config import GEMINI_API_KEY
from mcp_tools.mcp_client import get_mcp_client
from milvus.rag_knowledge_base import RAGKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RULE_SERIES_SCORE_DEFAULT = 0.9
class RecommendationAgent:
    """推荐智能体类"""

    def __init__(self, gemini_api_key: str = None):
        """初始化推荐智能体"""
        self.mcp_client = get_mcp_client()
        self.rag_kb = RAGKnowledgeBase()
        self.rag_kb.connect()
        logger.info("recommendation_agent初始化完毕（使用MCP工具+RAG知识库）")

    def map_explicit_needs_to_packages(self, user_explicit_needs: Optional[UserExplicitNeeds]) -> List[str]:
        """
        基于用户明确需求的规则映射，直接映射到套餐类别
        
        Args:
            user_explicit_needs: 用户明确需求
            
        Returns:
            套餐类别列表
        """
        if user_explicit_needs is None:
            logger.info("用户明确需求为空，跳过规则映射")
            return []
        
        package_categories = []
        
        # 规则1: 流量溢出 → 包月流量包
        if user_explicit_needs.is_data_overused:
            package_categories.append("包月流量包")
            logger.info("规则匹配: is_data_overused=True → 包月流量包")
        
        # 规则2: 语音溢出 → 语音包
        if user_explicit_needs.is_voice_exceeds:
            package_categories.append("语音包")
            logger.info("规则匹配: is_voice_exceeds=True → 语音包")
        
        # 规则3: 5G、宽带、宽带升级的组合逻辑
        need_5g = user_explicit_needs.need_5g
        need_broadband = user_explicit_needs.need_broadband
        need_broadband_upgrade = user_explicit_needs.need_broadband_upgrade
        
        if need_5g and (need_broadband or need_broadband_upgrade):
            # A + (B 或 C) → 5G畅享融合套餐（含千兆宽带）
            package_categories.append("5G畅享融合套餐（含千兆宽带）")
            logger.info("规则匹配: need_5g=True + (need_broadband或need_broadband_upgrade) → 5G畅享融合套餐")
        elif need_5g and not need_broadband and not need_broadband_upgrade:
            # 仅A → 5G畅享套餐
            package_categories.append("5G畅享套餐")
            logger.info("规则匹配: 仅need_5g=True → 5G畅享套餐")
        elif not need_5g and (need_broadband or need_broadband_upgrade):
            # 不包含A，但有B或C → 单独宽带升千兆服务
            package_categories.append("单独宽带升千兆服务")
            logger.info("规则匹配: 不需要5G + (need_broadband或need_broadband_upgrade) → 单独宽带升千兆服务")
        
        # 规则4: 设备更换需求 → 徽金卡单品（终端版）
        if user_explicit_needs.device_replacement_needs:
            package_categories.append("徽金卡单品（终端版）")
            logger.info("规则匹配: device_replacement_needs=True → 徽金卡单品（终端版）")
        
        # 规则5: 家庭套餐或智能家居需求 → 智慧家庭礼包
        if user_explicit_needs.is_family_plan or user_explicit_needs.smart_home_needs:
            package_categories.append("智慧家庭礼包")
            logger.info("规则匹配: is_family_plan=True 或 smart_home_needs=True → 智慧家庭礼包")
        
        # 规则6: 视频需求 → 畅玩卡 或 iTV影视优品包
        if user_explicit_needs.video_needs:
            package_categories.extend(["畅玩卡", "iTV影视优品包"])
            logger.info("规则匹配: video_needs=True → 畅玩卡, iTV影视优品包")
        
        # 规则7: 教育需求 → iTV课堂优品包 或 iTV少儿优品包
        if user_explicit_needs.education_needs:
            package_categories.extend(["iTV课堂优品包", "iTV少儿优品包"])
            logger.info("规则匹配: education_needs=True → iTV课堂优品包, iTV少儿优品包")
        
        # 规则8: 特殊身份映射
        if user_explicit_needs.special_identity:
            identity = user_explicit_needs.special_identity
            identity_mapping = {
                "老年人": "孝心卡",
                "军人": "拥军卡",
                "军人家属": "拥军卡",
                "残疾人": "爱心卡",
                "贫困户": "无忧卡"
            }
            if identity in identity_mapping:
                package_categories.append(identity_mapping[identity])
                logger.info(f"规则匹配: special_identity={identity} → {identity_mapping[identity]}")
        
        # 规则9: 偏好套餐类型
        if user_explicit_needs.prefer_package_type:
            package_categories.append(user_explicit_needs.prefer_package_type)
            logger.info(f"规则匹配: prefer_package_type={user_explicit_needs.prefer_package_type}")
        
        # 去重
        package_categories = list(set(package_categories))
        
        if package_categories:
            logger.info(f"✅ 规则映射完成，共匹配到 {len(package_categories)} 个套餐类别: {package_categories}")
        else:
            logger.info("⚠️ 规则映射未匹配到任何套餐类别")
        
        return package_categories

    def retrieve_package_categories(self, db_user_summary: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        阶段1：使用用户需求总结检索推荐原则，获取套餐类别
        
        Args:
            db_user_summary: 用户需求总结（已由user_info_agent或slot_filling_agent生成）
            top_k: 返回top-k个结果
            
        Returns:
            推荐原则列表（包含套餐类别）
        """
        try:
            logger.info(f"使用需求总结检索推荐原则: {db_user_summary[:50]}...")
            results = self.rag_kb.search(db_user_summary, top_k=top_k)
            
            if results:
                logger.info(f"RAG检索成功，返回{len(results)}条推荐原则")
                # 设置索引从1开始
                for i, r in enumerate(results, 1):
                    logger.info(f"  {i}. {r['title']} (相似度: {r['similarity_score']:.3f})")
                    logger.info(f"     推荐类别: {r['recommended_products']}")
                return results
            else:
                logger.warning("RAG检索未返回结果")
                return []
                
        except Exception as e:
            logger.error(f"RAG检索失败: {e}")
            return []

    def extract_package_categories(self, principles: List[Dict[str, Any]]) -> List[str]:
        """
        从推荐原则中提取套餐类别
        
        Args:
            principles: RAG检索到的推荐原则
            
        Returns:
            套餐类别列表（去重）
        """
        categories = set()

        for principle in principles:
            recommended_products = principle.get("recommended_products", "")
            products = recommended_products.replace("或", ";").split(";")
            for product in products:
                if product:
                    categories.add(product)

        categories_list = list(categories)
        logger.info(f"提取到的套餐类别{categories_list}")
        return categories_list

    def format_principles_context(self, principles: List[Dict[str, Any]]) -> str:
        """
        格式化推荐原则为LLM上下文
        
        Args:
            principles: 推荐原则列表
            
        Returns:
            格式化的文本
        """
        if not principles:
            return "未检索到相关推荐原则"
        
        formatted = []
        for i, p in enumerate(principles, 1):
            formatted.append(f"""
                            【推荐原则 {i}】
                            标题: {p['title']}
                            分类: {p['category']}
                            触发条件: {p['trigger_conditions']}
                            推荐产品: {p['recommended_products']}
                            相似度: {p['similarity_score']:.3f}
                            """)
        
        return "\n".join(formatted)

    def filter_packages_by_trigger_conditions(
        self,
        user_explicit_needs: Optional[UserExplicitNeeds],
        merged_user_info: Dict[str, Any],
        principles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        基于RAG检索到的推荐原则，判断用户是否满足触发条件，筛选出满足的套餐类别
        
        Args:
            user_explicit_needs: 用户明确需求
            merged_user_info: 合并后的用户信息（已包含覆盖值）
            principles: RAG检索到的推荐原则（包含触发条件）
            
        Returns:
            满足触发条件的套餐类别（带分数）列表，如 [{"series": name, "score": 0.8, "source": "llm"}]
        """
        logger.info("开始LLM筛选：判断用户是否满足推荐原则的触发条件")
        
        if not principles:
            logger.warning("无推荐原则可筛选，返回空列表")
            return []
        
        # 格式化用户明确需求
        explicit_needs_str = format_user_explicit_needs(user_explicit_needs)
        
        # 格式化推荐原则（包含触发条件和推荐产品）
        principles_context = self.format_principles_context(principles)

        # 格式化合并后的用户信息
        merged_user_info_str = "\n".join(f"-{key}: {value}" for key, value in merged_user_info.items()) if (
            merged_user_info) else "-未查询到用户信息"
        
        # 构建提示词
        prompt = f"""你是一个电信套餐推荐系统的筛选模块。请根据用户信息，判断用户是否满足候选推荐原则的触发条件，
        筛选出满足条件的套餐类别，并为每个类别给出0.0到1.0之间的匹配分数（score）。

【用户信息】（已合并用户输入的覆盖值）
{merged_user_info_str}

【候选推荐原则】（从知识库RAG检索得到）
{principles_context}

【筛选任务】
请逐条分析【候选推荐原则】，判断用户是否满足每条原则的**触发条件**：

1. **触发条件判断规则**：
   
   - 参考【用户信息】中的属性字段和候选推荐原则中每条原则的触发条件对属性字段的要求做对比，看是否满足。

2. **提取满足条件的套餐类别**：
   - 从满足触发条件的推荐原则中，提取"推荐产品"字段的值
   - 推荐产品可能包含多个套餐（用"或"、";"分隔）
   - 将所有满足条件的套餐类别汇总

**返回JSON格式：**
{{
    "filtered": [
        {{"series": "套餐类别1", "score": 0.85}},
        {{"series": "套餐类别2", "score": 0.70}}
    ]
}}

【重要提示】
- 只返回JSON，不要有其他内容
- 套餐类别名称必须从【候选推荐原则】的"推荐产品"字段中提取
- "filtered" 键对应的是一个数组，数组中的每个元素都是一个包含 "series" 和 "score" 的对象。
- 如果没有满足条件的原则，返回空数组：{{"filtered": []}}

请返回筛选结果的JSON："""

        try:
            # 使用 MCP 工具生成并提取 JSON
            result = self.mcp_client.generate_and_extract_json(prompt, is_array=False, temperature=0.2)

            if result and "filtered" in result:
                items = result["filtered"] or []
                cleaned = []
                for it in items:
                    try:
                        series = it.get("series") or it.get("category")
                        score = float(it.get("score", 0))
                        if series:
                            cleaned.append({"series": series, "score": max(0.0, min(score, 1.0)), "source": "llm"})
                    except Exception:
                        continue
                logger.info(f"✅ LLM筛选成功，返回{len(cleaned)}个带分数的类别")
                return cleaned
            else:
                logger.warning("LLM筛选失败，返回空列表")
                return []
                
        except Exception as e:
            logger.error(f"LLM筛选时出错: {e}")
            return []

    def process(self, state: AgentState) -> AgentState:
        """
        推荐智能体主入口 - 套餐类别筛选（不负责具体价位筛选）
        
        流程：
        1. RAG检索：db_user_summary → 推荐原则（包含触发条件）
        2. LLM筛选：判断用户是否满足触发条件 → 筛选满足的套餐类别
        3. 规则映射：user_explicit_needs → 套餐类别（基于明确意图）
        4. 合并去重：LLM筛选结果 + 规则映射结果 → 最终套餐类别列表
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态（包含filtered_package_categories）
        """
        logger.info("=" * 70)
        logger.info("开始执行套餐类别筛选流程（RAG检索 + LLM筛选 + 规则映射）")
        logger.info("=" * 70)
        
        # 获取数据
        user_explicit_needs = state.get("user_explicit_needs")  # 可能为空（流程1）
        merged_user_info = state.get("merged_user_info", {})  # 合并后的用户信息（优先使用覆盖值）
        db_user_summary = state.get("db_user_summary", "")  # 应该有（由前置agent生成）
        cfg = state.get("_config", {})
        use_rag = cfg.get("use_rag", True)  # 默认启用RAG

        # === 阶段1：RAG检索 - 基于用户需求总结 ===
        logger.info("\n【阶段1】RAG检索：基于用户需求总结检索推荐原则")
        if use_rag:
            principles = self.retrieve_package_categories(db_user_summary, top_k=5)
            if not principles:
                logger.warning("⚠️ RAG检索未返回推荐原则")
        else:
            logger.info("消融配置：跳过RAG检索")
            principles = []

        # === 阶段2：LLM筛选 - 判断触发条件 ===
        logger.info("\n【阶段2】LLM筛选：判断用户是否满足推荐原则的触发条件")
        if use_rag and principles:
            llm_filtered_categories = self.filter_packages_by_trigger_conditions(
                user_explicit_needs,
                merged_user_info,
                principles
            )
        else:
            llm_filtered_categories = []
        logger.info(f"LLM筛选结果（满足触发条件）: {llm_filtered_categories if llm_filtered_categories else '无'}")
        
        # === 阶段3：规则映射 - 基于用户明确需求 ===
        logger.info("\n【阶段3】规则映射：基于用户明确需求映射套餐类别")
        rule_based_categories = self.map_explicit_needs_to_packages(user_explicit_needs)
        logger.info(f"规则映射结果: {rule_based_categories if rule_based_categories else '无'}")
        rule_scored = [{"series": c, "score": RULE_SERIES_SCORE_DEFAULT, "source": "rule"}
                       for c in rule_based_categories]
        
        # === 阶段4：合并去重 ===
        logger.info("\n【阶段4】合并去重：合并LLM筛选结果和规则映射结果（同名取高分）")

        scored_map = {}
        for item in rule_scored + llm_filtered_categories:
            name = item.get("series") if isinstance(item, dict) else str(item)
            score = item.get("score", 0.0) if isinstance(item, dict) else 0.0
            src = item.get("source", "unknown") if isinstance(item, dict) else "unknown"
            if name in scored_map:
                if score > scored_map[name]["score"]:
                    scored_map[name] = {"series": name, "score": score, "source": src}
            else:
                scored_map[name] = {"series": name, "score": score, "source": src}
        scored_list = list(scored_map.values())
        scored_list.sort(key=lambda x: x["score"], reverse=True)
        final_categories = [x["series"] for x in scored_list]
        
        if not final_categories:
            logger.warning("⚠️ 未获取到任何套餐类别")
        else:
            logger.info(f"✅ 最终筛选的套餐类别（共{len(final_categories)}个）:")
            for i, item in enumerate(scored_list, 1):
                logger.info(f"  {i}. {item['series']} （来源: {item['source']}，系列分: {item['score']:.2f}）")
        
        # 更新状态
        state["filtered_package_categories"] = final_categories
        state["scored_package_series"] = scored_list
        state["next_node_to_call"] = "price_selection_agent"  # 下一个智能体：价位筛选
        state["final_response"] = f"已为您筛选出 {len(final_categories)} 个合适的套餐类别，正在为您匹配具体价位..."
        
        logger.info("\n" + "=" * 70)
        logger.info("套餐类别筛选流程完成")
        logger.info("=" * 70)
        return state


# 创建全局推荐智能体实例
# recommendation_agent = RecommendationAgent(GEMINI_API_KEY)
recommendation_agent = RecommendationAgent()


def recommendation_node(state: AgentState) -> AgentState:
    """
    推荐节点入口函数
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态
    """
    return recommendation_agent.process(state)

