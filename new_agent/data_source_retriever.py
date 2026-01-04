"""
DataSourceRetriever: 负责从所有外部数据源（MySQL, 向量数据库）获取原始信息。
"""
import logging
from typing import Dict, Any, List, Optional
from mcp_tools.mcp_client import get_mcp_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSourceRetriever:
    """
    封装对所有外部数据源的查询逻辑。
    """

    def __init__(self):
        self.mcp_client = get_mcp_client()

    def query_rec_principles(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        根据用户画像查询推荐原则向量数据库，获取最匹配的套餐系列名称和它的推荐原则加简介。

        Args:
            query: 根据用户记忆生成的自然语言查询。
            top_k: 返回最匹配的系列数量。

        Returns:
            一个包含推荐系列名称的列表，例如 ['5G畅享融合套餐', '畅玩卡']。
        """
        try:
            # search 方法返回一个字典列表，我们需要从中提取出套餐系列名称
            results = self.mcp_client.rag_tools.search(query, top_k)

            # 假设返回的字典中有 'recommended_products' 字段
            series_total = []
            for res in results:
                series = {
                    "trigger_conditions": res.get("trigger_conditions"),
                    "recommended_products": res.get("recommended_products"),
                    "package_brief": res.get("package_brief")
                }
                series_total.append(series)
            return series_total
        except Exception as e:
            logger.error(f"查询推荐原则时出错: {e}", exc_info=True)
            return []

    def query_user_profile_db(self, user_id: str) -> Dict[str, Any]:
        """
        查询用户MySQL数据库，获取静态补充信息。

        Args:
            user_id: 用户ID。

        Returns:
            一个包含用户属性的字典。
        """
        try:
            return self.mcp_client.mysql_tools.query_cursor_by_phone_number(user_id)
        except Exception as e:
            logger.error(f"查询用户MySQL资料时出错: {e}", exc_info=True)
            return {}

    def query_package_brief_db(self, series_name: str) -> Optional[Dict[str, Any]]:
        """
        查询套餐MySQL数据库，获取指定系列的所有套餐详情和系列简介。

        Args:
            series_name: 套餐系列名称。

        Returns:
            一个字典，包含 'introduction' 和 'packages' 列表，或在失败时返回 None。
        """

        try:
            brief = self.mcp_client.mysql_tools.query_package_intro_by_name(series_name)
            return brief
        except Exception as e:
            logger.error(f"查询套餐详情时出错: {e}", exc_info=True)
            return None
