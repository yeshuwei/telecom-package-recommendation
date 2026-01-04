"""
MemoryManager: 负责与 mem0 服务进行交互，包括更新和查询用户记忆。
"""
import logging
import json
from typing import Dict, Any, List, Optional

from new_agent.memory_service import get_memory_service
from mcp_tools.mcp_client import get_mcp_client
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryManager:
    """
    封装与 mem0 服务的交互。它直接利用 mem0 内置的 LLM 和提示词来处理记忆，
    不再进行外部的二次处理。
    """

    def __init__(self):
        self.memory_service = get_memory_service()
        self.mcp_client = get_mcp_client()

    def update_memory(self, user_id: str, user_input: str) -> bool:
        """
        直接将用户输入传递给 mem0.add()。
        mem0 内部会使用预设的提示词和LLM自动提取、合并、修正和强化记忆。

        Args:
            user_id: 用户ID
            user_input: 用户的原始输入

        Returns:
            是否成功更新记忆
        """
        if not self.memory_service.available:
            logger.warning("MemoryService 不可用，跳过记忆更新。")
            return False

        try:
            metadata = {"role": "user"}
            # 直接调用 add，让 mem0 内部处理一切
            self.memory_service.add(user_id=user_id, messages=user_input, metadata=metadata)
            logger.info(f"成功将输入传递给 mem0 进行处理，用户: {user_id}")
            return True
        except Exception as e:
            logger.error(f"更新用户 {user_id} 的记忆时出错: {e}", exc_info=True)
            return False

    def add_system_memory(self, user_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        存储系统侧生成的记忆，例如AI的回复、推荐动作等。
        这个方法不会调用LLM进行二次处理，而是直接存储。
        """
        if not self.memory_service.available:
            return False

        if metadata is None:
            metadata = {}
        # 为所有系统记忆添加一个 'role' 标识
        if "role" not in metadata:
            metadata['role'] = 'assistant'
        
        try:
            self.memory_service.system_add(user_id=user_id, messages=text, metadata=metadata)
            logger.info(f"成功为用户 {user_id} 添加了一条系统回复记忆。")
            return True
        except Exception as e:
            logger.error(f"添加系统记忆时出错: {e}", exc_info=True)
            return False

    def query_memory(self, user_id: str, query: str, top_k: int = 1, filters: Optional[Dict[str, Any]] = None) \
            -> Optional[Any]:
        """
        查询用户关于某个特定方面的前top_k记忆。

        Args:
            user_id: 用户ID
            query: 相关的查询语句。
            top_k: 返回最相关的几条记忆
            filters: 过滤条件
        Returns:
            最相关的记忆值，或 None
        """
        if not self.memory_service.available:
            return None

        results = self.memory_service.search(
            user_id=user_id,
            query=query,
            top_k=top_k,
            filters=filters
        )

        return results.get("results")

    def get_recent_turns(self, user_id: str, n: int = 4) -> List[Dict[str, str]]:
        """
        获取最近的 n 轮对话历史。

        Args:
            user_id: 用户ID
            n: 要获取的对话轮数

        Returns:
            一个包含系统回复历史的列表，例如 [{'role': 'assistant', 'text': '...'}]
        """
        try:
            all_memories = self.memory_service.get_all_user_system_memory(user_id=user_id)
            if not all_memories:
                return []

            recent_turns = []
            # 假设 all_memories 是按时间倒序的
            for mem in all_memories[:n]:
                role = mem.get("role", "unknown")
                text = mem.get("memory", "")
                if role != "unknown" and text:
                    recent_turns.append({"role": role, "text": text})
            
            # 因为是从最近的开始取，所以需要反转列表，让其符合对话顺序
            return list(reversed(recent_turns))

        except Exception as e:
            logger.error(f"获取用户 {user_id} 的最近对话历史时出错: {e}", exc_info=True)
            return []

    def get_full_profile_summary(self, user_id: str,  detailed_memories: Dict[str, list]) -> Optional[Any]:
        """
        获取用户近期的所有记忆，用于构建更丰富的查询。
        """

        user_history_info = self.mcp_client.mysql_tools.query_cursor_by_phone_number(user_id)
        prompt = f"""请根据以下用户的上下文记忆信息，历史信息，生成一段简短的、第三人称的摘要，
                    总结他的核心需求和偏好,方便从推荐原则向量数据库中查询合适的套餐系列。
                    注意：如果发生冲突，请使用相关的上下文记忆信息去覆盖对应的历史信息部分，确保记忆总结能够全面的反映用户的需求，
                    历史信息只在缺失的时候作为补充。
                    上下文记忆信息：
                    - {json.dumps(detailed_memories, indent=2, ensure_ascii=False)}
                    历史信息：
                    - {user_history_info}
                    
                    返回字符串格式的摘要即可。
                    摘要："""

        try:
            summary = self.mcp_client.generate_content(prompt)
            return summary.strip()
        except Exception as e:
            logger.error(f"生成记忆摘要失败: {e}")
            return ""
