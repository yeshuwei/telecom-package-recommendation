"""
Milvus向量数据库客户端
"""
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import logging
from configs.config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, DIMENSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(self):
        self.collection = None
        
    def connect(self):
        """连接到Milvus数据库"""
        try:
            connections.connect(
                # 指定连接的名称，通过这个名称可以确定要使用的连接实例
                alias="default",
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
            logger.info(f"成功连接到Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
            return True
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            return False
    
    def create_collection(self):
        """创建集合"""
        try:
            # 检查集合是否已存在，如果存在则删除
            if utility.has_collection(COLLECTION_NAME):
                logger.info(f"集合 {COLLECTION_NAME} 已存在，正在删除...")
                utility.drop_collection(COLLECTION_NAME)
                logger.info(f"集合 {COLLECTION_NAME} 已删除")
            
            # 定义字段schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="principle_id", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="trigger_conditions", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="recommended_products", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="package_brief", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                # 定义embedding字段的数据类型为浮点型向量，维度dim通过读取config文件得到
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
            ]
            
            # 创建集合schema
            schema = CollectionSchema(
                fields=fields,
                description="电信产品推荐原则知识库"
            )
            
            # 创建集合
            self.collection = Collection(
                name=COLLECTION_NAME,
                schema=schema
            )
            
            # 创建平面索引（适合小数据量，100%精确度）
            index_params = {
                "metric_type": "COSINE",
                "index_type": "FLAT"
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"成功创建集合: {COLLECTION_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False
    
    def get_existing_collection(self):
        """获取已存在的集合"""
        try:
            if utility.has_collection(COLLECTION_NAME):
                self.collection = Collection(COLLECTION_NAME)
                logger.info(f"成功获取已存在的集合: {COLLECTION_NAME}")
                return True
            else:
                logger.error(f"集合 {COLLECTION_NAME} 不存在")
                return False
        except Exception as e:
            logger.error(f"获取集合失败: {e}")
            return False
    
    def load_collection(self):
        """加载集合到内存"""
        try:
            if self.collection:
                self.collection.load()
                logger.info(f"集合 {COLLECTION_NAME} 已加载到内存")
                return True
            return False
        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False
    
    def insert_data(self, data):
        """插入数据"""
        try:
            if not self.collection:
                logger.error("集合未初始化")
                return False

            # 验证数据格式
            required_fields = ['principle_id', 'title', 'content', 'trigger_conditions', 'recommended_products', 'package_brief', 'category', 'embedding']
            for field in required_fields:
                if field not in data:
                    logger.error(f"缺少必需字段: {field}")
                    return False
                
                if not isinstance(data[field], list):
                    logger.error(f"字段 {field} 必须是列表类型")
                    return False
                
                if len(data[field]) == 0:
                    logger.error(f"字段 {field} 不能为空")
                    return False

            # 验证所有字段长度一致
            field_lengths = {field: len(data[field]) for field in required_fields}
            if len(set(field_lengths.values())) > 1:
                logger.error(f"所有字段长度必须一致: {field_lengths}")
                return False

            # 验证embedding维度
            first_embedding = data['embedding'][0]
            if len(first_embedding) != DIMENSION:
                logger.error(f"Embedding维度不匹配: 期望 {DIMENSION}, 实际 {len(first_embedding)}")
                return False

            logger.info(f"数据验证通过，准备插入 {len(data['principle_id'])} 条记录")

            insert_data = [
                data["principle_id"],
                data["title"],
                data["content"],
                data["trigger_conditions"],
                data["recommended_products"],
                data["package_brief"],
                data["category"],
                data["embedding"],
            ]
            # 插入数据 注意milvus不接受dict格式的数据的插入！！！
            # 之前报错是因为将dict当做插入的列，那么dict其中的数组类型不相同 -> 同一列的数据含有不同的类型 所以报错
            insert_result = self.collection.insert(insert_data)
            self.collection.flush()
            
            logger.info(f"成功插入 {len(data['principle_id'])} 条记录")
            return insert_result
        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            return False
    
    def search(self, query_vectors, top_k=5):
        """搜索相似向量"""
        try:
            if not self.collection:
                logger.error("集合未初始化")
                return None

            # 平面索引搜索参数
            search_params = {
                "metric_type": "COSINE"
            }
            
            results = self.collection.search(
                data=query_vectors,
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["principle_id", "title", "content", "trigger_conditions", "recommended_products", "package_brief", "category"]
            )
            
            return results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return None
    
    def get_collection_stats(self):
        """获取集合统计信息"""
        try:
            if not self.collection:
                return None
            
            stats = self.collection.num_entities
            logger.info(f"集合 {COLLECTION_NAME} 包含 {stats} 条记录")
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return None
