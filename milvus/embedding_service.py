"""
阿里云text-embedding-v4模型服务
"""
import dashscope
from dashscope import TextEmbedding
import logging
import time
from configs.config import DASHSCOPE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        # 设置API Key
        dashscope.api_key = DASHSCOPE_API_KEY
        self.model = "text-embedding-v4"
        
    def get_embedding(self, text):
        """获取单个文本的向量表示"""
        try:
            response = TextEmbedding.call(
                model=self.model,
                input=text
            )
            
            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                return embedding
            else:
                logger.error(f"获取embedding失败: {response.message}")
                return None
                
        except Exception as e:
            logger.error(f"调用embedding服务失败: {e}")
            return None
    
    def get_embeddings_batch(self, texts, batch_size=10):
        """批量获取文本向量表示"""
        embeddings = []

        # range(start, stop, step) step决定了i每次变化的步长
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = TextEmbedding.call(
                    model=self.model,
                    input=batch_texts
                )
                
                if response.status_code == 200:
                    batch_embeddings = [item['embedding'] for item in response.output['embeddings']]
                    embeddings.extend(batch_embeddings)
                    logger.info(f"成功处理批次 {i//batch_size + 1}, 共 {len(batch_embeddings)} 个向量")
                else:
                    logger.error(f"批次处理失败: {response.message}")
                    # 为失败的批次添加None
                    embeddings.extend([None] * len(batch_texts))
                
                # 添加延迟避免API限制
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"批次处理异常: {e}")
                embeddings.extend([None] * len(batch_texts))
        
        return embeddings
    
    def test_connection(self):
        """测试连接"""
        try:
            test_text = "测试文本"
            embedding = self.get_embedding(test_text)
            
            if embedding:
                logger.info(f"Embedding服务连接正常，向量维度: {len(embedding)}")
                return True
            else:
                logger.error("Embedding服务连接失败")
                return False
                
        except Exception as e:
            logger.error(f"测试连接失败: {e}")
            return False
