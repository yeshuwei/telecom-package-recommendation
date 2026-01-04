"""
推荐回复生成智能体 - 基于推荐结果生成自然语言回复 (MCP-Refactored)

功能：
1. 从AgentState中提取所有推荐相关数据
2. 通过MCP Client从向量数据库检索推荐原则
3. 通过MCP Client从MySQL数据库查询套餐简介
4. 通过MCP Client调用大模型生成自然、专业的推荐回复

数据流：
- 输入：AgentState（包含用户需求、推荐套餐、价位选择等）
- 处理：RAG检索 + MySQL查询 + LLM生成 (All via MCP Client)
- 输出：final_response（自然语言推荐回复）
"""
import logging
from typing import Dict, Any, List
import traceback

# MCP Client for unified tool access
from mcp_tools.mcp_client import get_mcp_client
from agent.state import AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseGenerationAgent:
    """推荐回复生成智能体 (MCP-Refactored)"""

    def __init__(self):
        """
        初始化推荐回复生成智能体
        """
        self.mcp_client = get_mcp_client()
        logger.info("ResponseGenerationAgent初始化完成 (MCP Client)")

    def _extract_user_context(self, state: AgentState) -> str:
        """
        从AgentState中提取用户上下文信息
        """
        context_parts = []
        if state.get("db_user_summary"):
            context_parts.append(f"【用户画像】\n{state['db_user_summary']}")
        if state.get("chat_history"):
            recent_history = state["chat_history"][-3:]
            history_str = "\n".join(recent_history)
            context_parts.append(f"【对话历史】\n{history_str}")
        if state.get("input"):
            context_parts.append(f"【当前输入】\n{state['input']}")
        return "\n\n".join(context_parts)

    def _retrieve_recommendation_principles(self, user_context: str, top_k: int = 3) -> List[str]:
        """
        从向量数据库检索相关推荐原则
        """
        try:
            query_text = user_context[:500]
            search_results = self.mcp_client.rag_tools.search(query_text, top_k=top_k)
            principles = [result["content"] for result in search_results if "content" in result]
            logger.info(f"从RAG检索到 {len(principles)} 条推荐原则")
            return principles
        except Exception as e:
            logger.error(f"RAG检索失败: {e}")
            return []

    def _query_package_introductions(self, package_names: List[str]) -> Dict[str, str]:
        """
        从MySQL数据库查询套餐简介
        """
        introductions = {}
        try:
            for package_name in package_names:
                result = self.mcp_client.mysql_tools.query_package_intro_by_name(package_name)
                if result:
                    introductions[package_name] = result["description"]
                    logger.info(f"查询到套餐简介: {package_name}")
                else:
                    logger.warning(f"未找到套餐简介: {package_name}")
            logger.info(f"从MySQL查询到 {len(introductions)} 个套餐简介")
            return introductions
        except Exception as e:
            logger.error(f"MySQL查询失败: {e}")
            return {}

    def _format_recommendation_data(self, price_selections: List[Dict[str, Any]], introductions: Dict[str, str]) -> str:
        """
        格式化推荐数据为可读文本
        """
        if not price_selections:
            return "暂无推荐结果"

        formatted_parts = ["【具体套餐推荐】"]
        by_series = {}
        for item in price_selections:
            series = item.get("series", "未知系列")
            by_series.setdefault(series, []).append(item)

        for series, items in by_series.items():
            formatted_parts.append(f"\n系列：{series}")
            if series in introductions:
                formatted_parts.append(f"简介：{introductions[series][:100]}...")

            for item in items:
                plan_name = item.get("plan_name", "未知套餐")
                price = item.get("price")
                price_str = f"¥{int(price)}/月" if price else "价格待定"
                score = item.get("score", 0)
                series_score = item.get("series_score", 0)
                price_score = item.get("price_score", 0)
                reason = item.get("reason", "综合匹配")
                breakdown = item.get("score_breakdown", {})
                budget_match = breakdown.get("budget_match", 0)
                data_match = breakdown.get("data_match", 0)
                voice_match = breakdown.get("voice_match", 0)
                details = item.get("package_details", {})
                features_dict = details.get("features", {})
                features = [f"{key}:{value}" for key, value in features_dict.items() if value and str(value).strip()]
                features_str = "、".join(features[:5]) if features else "详见套餐说明"

                formatted_parts.append(
                    f"  • {plan_name}（{price_str}）\n"
                    f"    推荐理由：{reason}\n"
                    f"    套餐内容：{features_str}\n"
                    f"    匹配度分析：\n"
                    f"      - 综合评分：{score:.2f}（系列匹配{series_score:.2f} + 价位匹配{price_score:.2f}）\n"
                    f"      - 预算匹配度：{budget_match:.2f}\n"
                    f"      - 流量匹配度：{data_match:.2f}\n"
                    f"      - 通话匹配度：{voice_match:.2f}"
                )
        return "\n".join(formatted_parts)

    def _build_generation_prompt(self, user_context: str, recommendation_data: str, principles: List[str]) -> str:
        """
        构建大模型生成Prompt
        """
        prompt = f"""你是一位专业的电信套餐推荐顾问，需要根据用户需求和推荐结果，生成一段自然、专业、有说服力的推荐回复。

## 用户信息
{user_context}

## 推荐结果（包含详细评分数据）
{recommendation_data}

## 推荐原则（参考）
{chr(10).join(f"{i+1}. {p[:200]}..." for i, p in enumerate(principles)) if principles else "暂无推荐原则"}

## 核心任务：创造性地生成推荐理由

### 示例对比：
❌ **生硬的表达**："预算贴合度高、流量充足、通话满足（评分：0.88）"
✅ **自然的表达**："这款套餐非常适合您，价格完全在您100元的预算范围内（仅需99元），30GB的流量配额足够您日常刷视频、看新闻，300分钟通话时长也能覆盖您的基本通话需求。综合匹配度高达0.88，是我们为您精心挑选的最佳方案。"

## 生成要求
1. **语气自然**：像真人客服一样，亲切、专业、有温度
2. **创造性理由**：根据匹配度数据，用自然语言解释为什么推荐这个套餐
3. **突出优势**：强调套餐的具体内容（流量多少GB、通话多少分钟、是否支持5G等）
4. **逻辑清晰**：先总述用户需求，再分点介绍套餐，最后引导决策
5. **数据支撑**：引用具体数字（价格、流量、通话、评分），但要自然融入句子中
6. **避免模板化**：不要直接说"预算贴合度高"、"流量充足"这种代码生成的短语
7. **长度适中**：200-400字，不要过长或过短
8. **引导行动**：结尾引导用户咨询或办理

## 请生成推荐回复：
"""
        return prompt

    def generate_response(self, state: AgentState) -> str:
        """
        生成推荐回复
        """
        try:
            user_context = self._extract_user_context(state)
            price_selections = state.get("price_selection_results", [])
            principles = self._retrieve_recommendation_principles(state.get("db_user_summary"), top_k=3)
            package_names = [item["series"] for item in price_selections if item.get("series")]
            introductions = self._query_package_introductions(list(set(package_names)))
            recommendation_data = self._format_recommendation_data(price_selections, introductions)
            prompt = self._build_generation_prompt(user_context, recommendation_data, principles)

            logger.info("🤖 正在调用LLM生成推荐回复...")
            generated_text = self.mcp_client.generate_content(prompt)
            logger.info(f"✅ 推荐回复生成完成（长度：{len(generated_text)}字）")
            return generated_text

        except Exception as e:
            logger.error(f"❌ 生成推荐回复失败: {e}")
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
            return self._generate_fallback_response(state)

    def _generate_fallback_response(self, state: AgentState) -> str:
        """
        生成兜底回复（当大模型调用失败时）
        """
        logger.warning("使用兜底回复生成策略")
        price_selections = state.get("price_selection_results", [])
        response_parts = ["您好！根据您的需求，我为您推荐以下套餐：\n"]

        if price_selections:
            by_series = {}
            for item in price_selections:
                series = item.get("series", "未知系列")
                by_series.setdefault(series, []).append(item)

            for series, items in by_series.items():
                response_parts.append(f"\n【{series}】")
                for item in items:
                    plan_name = item.get("plan_name", "未知套餐")
                    price = item.get("price")
                    price_str = f"¥{int(price)}/月" if price else "价格待定"
                    score = item.get("score", 0)
                    reason = item.get("reason", "综合匹配")
                    response_parts.append(f"• {plan_name}（{price_str}）- {reason}（评分：{score:.2f}）")
        else:
            response_parts.append("暂时没有找到完全匹配的套餐，建议您适当调整预算或需求。")

        response_parts.append("\n如有疑问，欢迎随时咨询！")
        return "\n".join(response_parts)

    def _generate_response_from_knowledge_prompt(self, reply_generation_prompt: str) -> str:
        """
        基于知识/比较智能体生成的提示词生成回复
        
        Args:
            reply_generation_prompt: 知识/比较智能体生成的提示词
            
        Returns:
            生成的回复文本
        """
        try:
            logger.info("🤖 使用知识/比较提示词调用LLM生成回复...")
            generated_text = self.mcp_client.generate_content(reply_generation_prompt)
            logger.info(f"✅ 回复生成完成（长度：{len(generated_text)}字）")
            return generated_text
        except Exception as e:
            logger.error(f"❌ 基于知识提示词生成回复失败: {e}")
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
            return "抱歉，我在处理您的查询时遇到了问题，请稍后重试。"

    def process(self, state: AgentState) -> AgentState:
        """
        处理推荐回复生成流程
        
        根据用户意图选择不同的生成策略：
        - clarify_recommendation / unclear_recommendation: 使用原有推荐流程
        - inquiry / comparison: 使用知识/比较智能体生成的提示词
        """
        logger.info("=" * 80)
        logger.info("开始生成回复")
        logger.info("=" * 80)

        # 获取意图分析结果
        intent_analysis = state.get("intent_analysis", {})
        user_intent = intent_analysis.get("intent", "general")
        
        logger.info(f"用户意图: {user_intent}")

        # 根据意图选择生成策略
        if user_intent in ["inquiry", "comparison"]:
            # 使用知识/比较智能体的提示词
            reply_generation_prompt = state.get("reply_generation_prompt", "")
            if reply_generation_prompt:
                logger.info("使用知识/比较智能体提示词生成回复")
                final_response = self._generate_response_from_knowledge_prompt(reply_generation_prompt)
            else:
                logger.warning("未找到知识/比较提示词，使用兜底回复")
                final_response = self._generate_fallback_response(state)
        else:
            # 使用原有推荐流程（clarify_recommendation / unclear_recommendation / general）
            logger.info("使用原有推荐流程生成回复")
            final_response = self.generate_response(state)

        state["final_response"] = final_response
        state["next_node_to_call"] = "END"

        logger.info("=" * 80)
        logger.info("回复生成完成")
        logger.info("=" * 80)
        logger.info(f"\n{final_response}\n")

        return state

# 创建全局单例
response_generation_agent = ResponseGenerationAgent()

def response_generation_node(state: AgentState) -> AgentState:
    return response_generation_agent.process(state)
