"""
RecommendationAgent: 推荐流程的大脑，负责协调 MemoryManager 和 DataSourceRetriever，
并最终生成推荐结果。该类被设计为一个ReAct风格的自主智能体。
"""
import logging
import json
import re
from typing import Dict, Any, List, Optional

from new_agent.memory_manager import MemoryManager
from new_agent.data_source_retriever import DataSourceRetriever

# 假设有一个LLM客户端用于最终生成回复
try:
    from mcp_tools.mcp_client import get_mcp_client
except ImportError:
    def get_mcp_client():
        logging.warning("MCP Client not found, using fallback.")
        return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationAgent:
    """
    一个能够自主思考和调用工具的推荐智能体。
    """

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.data_retriever = DataSourceRetriever()
        self.llm_client = get_mcp_client()
        self.tools = {
            "perceive_and_understand_user": self.perceive_and_understand_user,
            "get_and_filter_recommendations": self.get_and_filter_recommendations,
            "generate_and_record_final_response": self.generate_and_record_final_response
        }

    # --- 主调度逻辑: Tools(Function Calling) Agent Loop --- #
    def run(self, user_id: str, user_input: str, max_iterations: int = 5) -> str:
        """
        使用 OpenAI 兼容的 tools(function calling) 与消息式对话循环，让模型自行选择调用工具。
        """
        if not self.llm_client:
            return "抱歉，推荐服务当前不可用。"

        # 1) 准备工具Schema（含参数说明），交给模型
        tools_schema = self._build_tools_schema()

        # 2) 初始化对话消息
        system_prompt = (
            "你是专业的电信推荐顾问，一个能够自主思考和使用工具的AI智能体。"
            "在需要时调用提供的工具完成任务；当工具返回了最终推荐话术后再结束对话。"
        )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "user_id:" + user_id + "user_input:" + user_input},
        ]

        # 3) 维护跨轮的Agent状态
        agent_state: Dict[str, Any] = {"user_id": user_id, "user_input": user_input}

        for i in range(max_iterations):
            logger.info(f"--- Agent Iteration {i+1} (tools mode) ---")
            # message含有：系统提示词，用户输入，工具调用决策，工具调用结果。后两个都是在前面的基础上直接进行累加。
            resp = self.llm_client.chat_once(messages=messages, tools=tools_schema, tool_choice="auto", temperature=0.1)
            if not resp or not resp.choices:
                return "抱歉，我暂时无法处理您的请求。"

            msg = resp.choices[0].message

            # 情况A：模型产出了工具调用
            if getattr(msg, "tool_calls", None):
                # 先把assistant带有tool_calls的消息也加入，会提高连贯性
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        } for tc in msg.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                for tool_call in msg.tool_calls:
                    if tool_call.type != "function":
                        continue
                    name = tool_call.function.name
                    raw_args = tool_call.function.arguments or "{}"
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = {}

                    logger.info(f"ToolCall -> {name}({args})")

                    # 执行工具
                    observation = self._execute_tool(name, args, agent_state)

                    # 记录到state，供后续轮次使用
                    agent_state[f"{name}_result"] = observation

                    # 将工具结果作为tool消息反馈给模型
                    tool_content = json.dumps(observation, ensure_ascii=False)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": tool_content,
                    })
                # 完成一个或多个工具调用后，继续下一轮
                continue

            # 情况B：没有工具调用，直接给出assistant自然语言
            content = msg.content or ""
            if content.strip():
                logger.info("Assistant returned final content without further tool calls.")
                return content.strip()

        return "抱歉，经过多次尝试后仍无法完成您的请求。"

    # --- 工具定义 --- #
    def perceive_and_understand_user(self, user_id: str, user_input: str) -> Dict[str, Any]:
        """工具1：感知和理解用户。当你需要开始处理一个新的用户请求时，首先调用此工具。
        
        参数说明：
        - user_id (str): 用户的唯一标识符，从初始请求中获取
        - user_input (str): 用户的本轮输入文本，从初始请求中获取
        
        返回：一个包含以下字段的JSON对象：
        - rewritten_input: 经过指代消解后的用户输入
        - profile_summary: 用户画像摘要
        - detailed_memories: 详细的上下文记忆（按不同维度分类，包括基础属性、行为特征、显性需求、交互反馈）
        
        功能：处理用户的本轮输入，结合历史记录解决指代问题，更新记忆，并形成一份完整的上下文理解。"""
        logger.info(f"[Tool Called] perceive_and_understand_user for user {user_id}")
        rewritten_input = self._rewrite_user_input(user_id, user_input)
        self.memory_manager.update_memory(user_id, rewritten_input)
        aspects_to_query = [
            "基础属性与资格",
            "行为特征与场景",
            "显性需求与痛点",
            "交互反馈与决策历史"
        ]
        detailed_memories = {}
        for aspect_query in aspects_to_query:
            memories = self.memory_manager.query_memory(user_id, "用户最新的" + aspect_query, 2, {"role": "user"})
            if memories:
                detailed_memories[aspect_query] = [mem.get('memory', '') for mem in memories]
        # assistant_history = self.memory_manager.query_memory(user_id, "我作为AI助手过去的回复", 2, {"role": "assistant"})
        # if assistant_history:
        #     detailed_memories["AI历史回复"] = [mem.get('memory', '') for mem in assistant_history]

        profile_summary = self.memory_manager.get_full_profile_summary(user_id, detailed_memories)
        if not profile_summary:
            profile_summary = f"用户说：'{rewritten_input}'"

        return {
            "rewritten_input": rewritten_input,
            "profile_summary": profile_summary,
            "detailed_memories": detailed_memories
        }

    def get_and_filter_recommendations(self, user_id: str, user_context: Dict[str, Any]) -> List[str]:
        """工具2：获取并过滤推荐。当你已经理解了用户画像并需要找出适合他们的套餐系列时，调用此工具。
        
        参数说明：
        - user_id (str): 用户的唯一标识符，从初始请求中获取
        - user_context (Dict[str, Any]): 用户上下文对象，从工具1 perceive_and_understand_user 的返回结果中获取
        
        返回：过滤后的套餐系列名称列表（如 ["5G畅享套餐", "动感地带潮玩卡"]）
        
        功能：根据用户上下文获取初步推荐，并利用LLM进行智能过滤，排除用户已拒绝的套餐系列。"""
        logger.info(f"[Tool Called] get_and_filter_recommendations for user {user_id}")
        candidate_series_objects = self.data_retriever.query_rec_principles(user_context["profile_summary"], top_k=5)
        if not candidate_series_objects:
            return []
        rejection_memories = self.memory_manager.query_memory(user_id, '用户表达过的不满、拒绝或不想要的套餐信息', 5, {"role": "user"})
        rejection_texts = [mem.get('memory', '') for mem in rejection_memories] if rejection_memories else []
        memory_context_parts = []
        for aspect, memories in user_context["detailed_memories"].items():
            if memories:
                memory_texts = "\n  - ".join(memories)
                memory_context_parts.append(f"**关于“{aspect}”的记忆:**\n  - {memory_texts}")
        memory_context_str = "\n\n".join(memory_context_parts)
        filtered_series_names = self._filter_with_llm(candidate_series_objects, rejection_texts, memory_context_str,
                                                      user_id)
        return filtered_series_names

    def generate_and_record_final_response(self, user_id: str, user_context: Dict[str, Any], candidates: List[str]) -> str:
        """工具3：生成并记录最终回复。当你已经获得了过滤后的推荐方案，并且认为信息足够，可以生成最终回复时，调用此工具。
        
        参数说明：
        - user_id (str): 用户的唯一标识符，从初始请求中获取
        - user_context (Dict[str, Any]): 用户上下文对象，从工具1 perceive_and_understand_user 的返回结果中获取
        - candidates (List[str]): 过滤后的套餐系列名称列表，从工具2 get_and_filter_recommendations 的返回结果中获取
        
        返回：最终的推荐话术文本（字符串）
        
        功能：产出最终的推荐话术并记录自己的行为到记忆库中。"""
        logger.info(f"[Tool Called] generate_and_record_final_response for user {user_id}")
        if not candidates:
            return "根据您的信息，我暂时没有找到合适的推荐，可以再聊聊您的具体想法吗？"
        filtered_series_with_plans = self._append_all_series_bundle(candidates)
        memory_context_parts = []
        for aspect, memories in user_context["detailed_memories"].items():
            if memories:
                memory_texts = "\n  - ".join(memories)
                memory_context_parts.append(f"**关于“{aspect}”的记忆:**\n  - {memory_texts}")
        memory_context_str = "\n\n".join(memory_context_parts)
        final_prompt = self._build_final_prompt(user_context["profile_summary"], memory_context_str,
                                                filtered_series_with_plans)
        try:
            final_response = self.llm_client.generate_content(final_prompt, temperature=0.7).strip()
            self.memory_manager.add_system_memory(
                user_id=user_id,
                text=final_response,
                metadata={"type": "recommendation_response"}
            )
            return final_response
        except Exception as e:
            logger.error(f"生成最终回复时出错: {e}", exc_info=True)
            return "抱歉，我在组织推荐信息时遇到了点问题，请稍后再试。"

    # --- Tools schema and execution helpers --- #
    def _build_tools_schema(self) -> List[Dict[str, Any]]:
        """返回OpenAI兼容的tools定义（JSON Schema）。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "perceive_and_understand_user",
                    "description": "工具1：感知和理解用户。先处理新请求，重写输入并生成上下文（rewritten_input--用户重写后的输入,"
                                   "profile_summary--用户画像的总结, detailed_memories--上下文记忆详情）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "用户ID（电话号码）"},
                            "user_input": {"type": "string", "description": "用户本轮输入文本"},
                        },
                        "required": ["user_id", "user_input"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_and_filter_recommendations",
                    "description": "工具2：获取并过滤推荐。根据用户上下文获取候选并过滤，返回系列名称列表",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "用户ID（电话号码）"},
                            "user_context": {
                                "type": "object",
                                "description": "工具1的返回结果（包含profile_summary与detailed_memories）",
                            },
                        },
                        "required": ["user_id", "user_context"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_and_record_final_response",
                    "description": "工具3：生成并记录最终回复。根据过滤后的系列与上下文产出最终话术",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "用户ID（电话号码）"},
                            "user_context": {
                                "type": "object",
                                "description": "工具1的返回结果（包含profile_summary与detailed_memories）",
                            },
                            "candidates": {
                                "type": "array",
                                "description": "由工具2返回的套餐系列名称列表",
                                "items": {"type": "string"}
                            },
                        },
                        "required": ["user_id", "user_context", "candidates"],
                    },
                },
            },
        ]

    def _execute_tool(self, name: str, args: Dict[str, Any], agent_state: Dict[str, Any]) -> Any:
        """根据名称执行本地工具，并自动补齐常用参数。"""
        func = self.tools.get(name)
        if not func:
            return {"error": f"未知工具: {name}"}

        # 动态准备工具参数
        tool_args: Dict[str, Any] = {}
        params = func.__code__.co_varnames
        if 'user_id' in params:
            tool_args['user_id'] = args.get('user_id') or agent_state.get('user_id')
        if 'user_input' in params:
            tool_args['user_input'] = args.get('user_input') or agent_state.get('user_input')
        if 'user_context' in params:
            tool_args['user_context'] = args.get('user_context') or agent_state.get('perceive_and_understand_user_result')
        if 'candidates' in params:
            tool_args['candidates'] = args.get('candidates') or agent_state.get('get_and_filter_recommendations_result')

        return func(**tool_args)

    # --- 内部辅助方法 --- #
    def _build_react_prompt(self, user_input: str, tools_description: str, scratchpad: str) -> str:
        return f"""你是专业的电信推荐顾问，一个能够自主思考和使用工具的AI智能体。
                
                你的任务是：根据用户的输入，通过一系列的思考和工具调用，最终为用户提供一个满意的套餐推荐。约束不必太严格，大致符合范围就能判断通过。
                比如说：用户预算100元，推荐一个129元的也行，用户没有明确表示不需要宽带，那么推荐一个含宽带的套餐也没问题。
                你有以下工具可用：
                {tools_description}
                请严格按照以下格式进行回应：
                
                Thought: 我需要做什么，为什么？我应该使用哪个工具？
                Action: `tool_name`({{"arg1": "value1", "arg2": "value2"}})
                
                `tool_name`必须是上面列出的工具之一。在每次工具调用后，你会得到一个`Observation`，然后你需要再次进行`Thought`和`Action`，直到你认为任务已经完成。
                
                你只有在调用工具3：generate_and_record_final_response 然后得到该工具的生成结果之后才能判断是否可以结束对话。
                认为有必要结束对话时，使用`Finish`作为最终的Action.
                Action: `Finish`({{}})
                
                开始！
                
                用户的初始请求是: "{user_input}"
                
                {scratchpad}"""

    def _parse_llm_output(self, llm_output: str) -> (str, str, Dict):
        thought_match = re.search(r"Thought: (.*?)\nAction: ", llm_output, re.DOTALL)
        action_match = re.search(r"Action: `(.*?)`\((.*?)\)", llm_output, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        if not action_match:
            return "(auto-generated thought)", "Finish", {"answer": llm_output}
            
        action = action_match.group(1).strip()
        action_input_str = action_match.group(2).strip()
        try:
            action_input = json.loads(action_input_str)
        except json.JSONDecodeError:
            action_input = {}

        return thought, action, action_input

    def _rewrite_user_input(self, user_id: str, user_input: str) -> str:
        recent_history = self.memory_manager.get_recent_turns(user_id, n=4)
        if not recent_history:
            return user_input
        history_text = "\n".join([f"{turn['role']}: {turn['text']}" for turn in recent_history])
        prompt = f"""你是一个对话理解专家。你的任务是根据一段对话历史和用户的最新输入，准确地理解用户的意图，并将其重写为一个清晰、独立、无歧义的指令。这个指令将交给另一个AI来执行。

                    ** 注意 **:如果用户的输入没有涉及到对话历史，请忽视掉对话历史。
                    例如：用户并没有说类似于“你上一个推荐的套餐太贵了”，“有没有更便宜的套餐”等等和历史系统回复有关的话语。
                    **核心任务：指代消解**
                    - 解析像“上一个”、“那个”、“第二个”这样的词语，将其替换为对话历史中明确提到的具体内容。
                    - 如果用户的指令已经很清晰，没有歧义，就直接返回原始指令。
                    
                    **对话历史:**
                    {history_text}
                    
                    **用户最新输入:**
                    "{user_input}"
                    
                    **要求:**
                    - 只返回重写后的指令文本，不要包含任何额外的解释或标签。
                    - 确保输出的指令是完整的、自包含的。
                    
                    **重写后的指令:**"""
        try:
            rewritten_input = self.llm_client.generate_content(prompt, temperature=0.1).strip()
            logger.info(f"[Rewrite] 原始输入: '{user_input}' -> 重写后: '{rewritten_input}'")
            return rewritten_input
        except Exception as e:
            logger.error(f"用户输入重写失败: {e}", exc_info=True)
            return user_input

    def _filter_with_llm(self, candidates: List[Dict[str, Any]], rejection_texts: List[str], memory_context_str: str, user_id: str) -> List[str]:
        user_profile = self.data_retriever.query_user_profile_db(user_id)
        if user_profile is None:
            logger.warning(f"未找到用户{user_id}的相关信息。")
            user_profile = []
        user_profile_text = "\n".join(user_profile)
        text = "\n- ".join(rejection_texts)
        prompt = f"""你是一个推荐系统里的智能过滤器。
                
                **任务**:
                1.请阅读用户过去的一些可能存在的负面反馈或拒绝历史（可能用户拒绝历史中并不包含用户的拒绝信息，因为该字段仅仅只是通过相似度查找得到），
                然后从“候选推荐列表”中移除用户明确表示过不想要的套餐系列。
                注意用户是不想要套餐系列还是某个具体的套餐！如：“我不想要孝心卡”，这是不想要孝心卡这个系列的套餐。
                                                    “我不想要5G畅享融合套餐169元”，这是仅仅不要这个套餐而已。
                2.根据用户上下文画像和充当数据补充的用户历史数据同套餐简介和出发条件进行对比，将不合适的套餐系列筛出出去。
                注意：如果发生冲突，以用户上下文画像中的数据为主，缺少的数据可以试着去用户历史数据中去寻找。
                
                **用户拒绝历史:**
                - {text}
                
                **用户上下文画像:**
                - {memory_context_str}
                
                **用户历史数据:**
                - {user_profile_text}
                
                **候选推荐列表 (JSON格式的对象列表):**
                "trigger_conditions" 套餐触发条件。由于用户上下文画像中没有严格的字段，所以你可以比较关联性比较强的部分看看是否合适，
                缺少的信息再去用户历史数据中寻找相关的字段。
                "recommended_products" 推荐的套餐系列名称。
                "package_brief" 对应的套餐简介。
                {json.dumps(candidates, indent=2, ensure_ascii=False)}
                
                **要求**:
                1. 根据每个候选对象`recommended_products` 字段，将其与用户的拒绝历史进行比对。
                2. 返回一个经过过滤的、新的JSON列表，只包含套餐系列名称，如果有系列名称重复，记得去重。
                3. 套餐系列名称格式为“A或B”视为一个套餐，格式为“A，B”的视为A，B两个套餐。
                
                
                **过滤后的JSON列表:**"""
        try:
            filtered_list = self.llm_client.generate_and_extract_json(prompt, is_array=True)
            if isinstance(filtered_list, list):
                return filtered_list
            logger.warning("LLM返回的过滤结果格式不正确，将使用原始列表。")
            result = set()
            for candidate in candidates:
                series_name: str = candidate.get("recommended_products")
                processed_string = series_name.replace("，", ",")
                sections = processed_string.split(",")
                for section in sections:
                    result.add(section)
            return list(result)
        except Exception as e:
            logger.error(f"LLM 智能过滤失败: {e}", exc_info=True)
            return []

    def _fetch_series_bundle(self, series_name: str):
        try:
            plans = self.llm_client.mysql_tools.query_plans_by_series(series_name)
            plan_items: List[Dict[str, Any]] = []
            for plan in plans:
                plan_id = plan.get("plan_id")
                features = self.llm_client.mysql_tools.query_features_by_plan_id(plan_id)
                plan_items.append({
                    "plan_id": plan_id,
                    "plan_name": plan.get("plan_name"),
                    "price": plan.get("price"),
                    "features": features
                })
            return plan_items
        except Exception as e:
            logger.error(f"查询系列明细失败：{series_name}, {e}")
            return None

    def _append_all_series_bundle(self, filtered_series_objects: list[str]) -> list[dict[str, Any]]:
        series_with_bundle = []
        for series_name in filtered_series_objects:
            item = {}
            plan_items = self._fetch_series_bundle(series_name)
            item["plan_items"] = plan_items
            item["recommended_products"] = series_name
            package_brief = self.data_retriever.query_package_brief_db(series_name)
            item["package_brief"] = package_brief
            series_with_bundle.append(item)
        return series_with_bundle

    def rewrite_user_input(self, user_id: str, user_input: str) -> str:
        return self._rewrite_user_input(user_id, user_input)
    
    def _build_final_prompt(self, user_profile: str, memory_context_str: str, filtered_series_objects: list[dict[str, Any]]) -> str:
        series_details_texts = []
        for series in filtered_series_objects:
            series_name = series.get("recommended_products")
            series_intro = series.get("package_brief", "暂无简介")
            package_texts = []
            if series.get("plan_items"):
                for pkg in series["plan_items"]:
                    price = pkg.get('price', 'N/A')
                    features = pkg.get('features', '无')
                    package_texts.append(f"  - {price}元档: {features}")
            package_text = "\n".join(package_texts) if package_texts else "  暂无具体套餐信息。"
            series_text = f"""**套餐系列: {series_name}**
                        *系列简介*: {series_intro}
                        *具体套餐选项*:
                        {package_text}
                        """
            series_details_texts.append(series_text)
        series_details_text = "\n".join(series_details_texts)
        # 这块目前不包含交互历史
        prompt = f"""你是一位顶级的智能电信业务推荐客服，拥有深刻的洞察力。
                
                ### 第一部分：完整的上下文
                
                1. 用户画像摘要:
                {user_profile}
                2. 详细的上下文记忆 (按重要性排序):
                {memory_context_str if memory_context_str else "无更多详细记忆。"}
                
                ### 第二部分：可选择的推荐方案
                
                {series_details_text}
                
                ### 第三部分：你的任务
                只需要你输出**生成话术**的部分。
                请严格按照以下步骤思考：
                1.  **深度理解**: 结合“用户画像摘要”和“详细的上下文记忆”，形成对用户需求的立体、精准判断。特别注意“交互反馈与决策历史”，避免重复或推荐用户已拒绝的内容。
                2.  **智能决策**: 从“待选的推荐方案”中，选择1-2个最能解决用户核心痛点或满足其显性需求的套餐进行重点推荐。
                3.  **生成话术**: 生成一段符合你角色身份、有说服力的推荐语。
                    *   先精准回应用户的核心需求（参考上下文记忆）。
                    *   结合套餐简介，清晰地解释为什么推荐的系列是最佳选择。
                    *   介绍所选系列的具体套餐选项，并可能根据用户的预算、痛点等信息，进行倾向性推荐（例如，“考虑到您上月流量超出较多，159元档位可能更具性价比”）。
                    *   最后，主动、开放地询问用户意见，引导对话继续。
                
                请开始你的推荐："""
        return prompt
