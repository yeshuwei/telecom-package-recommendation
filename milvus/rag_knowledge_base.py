"""
RAG知识库主模块
"""
import logging
from typing import List, Dict, Any, Optional
from milvus.milvus_client import MilvusClient
from milvus.embedding_service import EmbeddingService
from milvus.document_processor import DocumentProcessor
from configs.config import SOURCE_DOC_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGKnowledgeBase:
    def __init__(self):
        self.milvus_client = MilvusClient()
        self.embedding_service = EmbeddingService()
        self.document_processor = DocumentProcessor()
        self.is_initialized = False

    def connect(self) -> bool:
        try:
            if not self.milvus_client.connect():
                return False
            if not self.embedding_service.test_connection():
                return False
            if not self.milvus_client.get_existing_collection():
                return False
            if not self.milvus_client.load_collection():
                return False
            self.is_initialized = True
            logger.info("连接RAG知识库成功")
            return True

        except Exception as e:
            logger.error(f"连接知识库失败{e}")
            return False
    def initialize(self) -> bool:
        """初始化知识库"""
        try:
            # 连接Milvus
            if not self.milvus_client.connect():
                return False

            # 测试embedding服务
            if not self.embedding_service.test_connection():
                return False

            # 创建集合
            if not self.milvus_client.create_collection():
                return False

            # 加载集合
            if not self.milvus_client.load_collection():
                return False

            self.is_initialized = True
            logger.info("RAG知识库初始化成功")
            return True

        except Exception as e:
            logger.error(f"初始化知识库失败: {e}")
            return False

    def build_knowledge_base(self) -> bool:
        """构建知识库"""
        try:
            if not self.is_initialized:
                logger.error("知识库未初始化")
                return False

            # 解析文档 提取出来元素为字典的列表 每一个字典为一条推荐规则
            principles = self.document_processor.parse_document(SOURCE_DOC_PATH)
            if not principles:
                logger.error("文档解析失败")
                return False

            # 生成搜索文本
            search_texts = []
            for principle in principles:
                search_text = self.document_processor.get_search_text(principle)
                search_texts.append(search_text)

            # 生成向量
            logger.info("开始生成向量...")
            embeddings = self.embedding_service.get_embeddings_batch(search_texts)

            # 过滤掉None的embedding
            valid_data = []
            for i, (principle, embedding) in enumerate(zip(principles, embeddings)):
                if embedding is not None:
                    valid_data.append({
                        'principle_id': principle['principle_id'],
                        'title': principle['title'],
                        'content': principle['content'],
                        'trigger_conditions': principle['trigger_conditions'],
                        'recommended_products': principle['recommended_products'],
                        'package_brief': principle.get('package_brief', ''),
                        'category': principle['category'],
                        'embedding': embedding
                    })

            if not valid_data:
                logger.error("没有有效的向量数据")
                return False

            # 准备插入数据，确保数据类型正确
            insert_data = {
                'principle_id': [str(item['principle_id']) for item in valid_data],
                'title': [str(item['title']) for item in valid_data],
                'content': [str(item['content']) for item in valid_data],
                'trigger_conditions': [str(item['trigger_conditions']) for item in valid_data],
                'recommended_products': [str(item['recommended_products']) for item in valid_data],
                'package_brief': [str(item['package_brief']) for item in valid_data],
                'category': [str(item['category']) for item in valid_data],
                'embedding': [list(map(float, item['embedding'])) for item in valid_data]  # 确保embedding是float列表
            }

            # 插入数据
            if self.milvus_client.insert_data(insert_data):
                logger.info(f"成功构建知识库，包含 {len(valid_data)} 条记录")
                return True
            else:
                logger.error("插入数据失败")
                return False

        except Exception as e:
            logger.error(f"构建知识库失败: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关原则"""
        try:
            if not self.is_initialized:
                logger.error("知识库未初始化")
                return []

            # 生成查询向量
            query_embedding = self.embedding_service.get_embedding(query)
            if not query_embedding:
                logger.error("生成查询向量失败")
                return []

            # 搜索 返回结构通常只将distance id等搜索元数据暴露 其余属性都封装在entity容器当中
            results = self.milvus_client.search([query_embedding], top_k)
            if not results:
                logger.error("搜索失败")
                return []

            # 格式化结果
            formatted_results = []
            for hit in results[0]:
                result = {
                    'principle_id': hit.entity.get('principle_id'),
                    'title': hit.entity.get('title'),
                    'content': hit.entity.get('content'),
                    'trigger_conditions': hit.entity.get('trigger_conditions'),
                    'recommended_products': hit.entity.get('recommended_products'),
                    'package_brief': hit.entity.get('package_brief'),
                    'category': hit.entity.get('category'),
                    'similarity_score': hit.score
                }
                formatted_results.append(result)

            logger.info(f"搜索完成，返回 {len(formatted_results)} 条结果")
            return formatted_results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_stats(self) -> Optional[int]:
        """获取知识库统计信息"""
        # 目前只有数据条数
        if self.is_initialized:
            return self.milvus_client.get_collection_stats()
        return None
