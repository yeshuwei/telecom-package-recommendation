"""
记忆服务 - 基于mem0框架的记忆管理
"""
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

from configs import config as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryService:
    """记忆服务类 - mem0适配器"""

    def __init__(self, namespace: str = "telco"):
        """
        初始化记忆服务
        
        Args:
            namespace: 命名空间
        """
        self.namespace = namespace
        self.available = False
        self.client = None

        custom_fact_extraction_prompt = """
            你是一个智能电信客服意图识别专家。请从用户的自然语言输入中提取关键信息，并将其归类到以下四个类别中。
            
            ### 信息分类定义（四大分类共11个子分类后面给出的只是举例，你也可以自由发挥，只要是属于这11个分类，和它们举的例子关联性较强即可）：
            
            1. 基础属性与资格 
               - 社会属性：年龄段、职业、居住/工作地环境是否有5G覆盖、同家庭有关的属性等等。
               - 账户属性：是否为老用户、信用等级、当前持有套餐或已订购的增值包等等。
               - 设备信息：终端型号、是否双卡等等。
            
            2. 行为特征与场景
               - 资源消耗：月均流量/语音用量、是否溢出（经常超过套餐上限/大量剩余）等等。
               - 消费习惯：付费敏感度（嫌贵/不在乎）等等。
               - 场景环境：出差漫游、游戏加速、居家办公、爱好追剧 等等。
               
            3. 显性需求与痛点 
               - 核心诉求：降费、提速、加流量、办副卡、宽带融合等等。
               - 业务痛点：流量不够、网速慢、信号差、资费贵、骚扰电话多等等。
               - 预算范围：具体的心理价位区间。
            
            4. 交互反馈与决策历史 
               - 拒绝画像：所有拒绝过的套餐及原因（关键）。
               - 意向痕迹：咨询过但未办理的业务，或者是用户明确提出的目标套餐等等。
            
            ### 输出要求：
            请仅以 JSON 格式返回结果。JSON 的格式必须严格对应下面的参考样例中的Output。"facts"列表中的每一项应该是一个概括性的短句。
            如果某个分类未提取到信息，请保留"facts"键并将它的值设为空列表 []。
            
            ### 参考样例 (Few-Shot Examples)：
            
            Input: 喂，听得到吗？
            Output: {
                "facts": []
            }
            
            Input: 我最近刚换了华为Mate60，但是感觉家里的网速好慢啊，看视频老卡。
            Output: {
                "facts": [
                "【基础属性与资格】终端型号：华为Mate60",
                "【行为特征与场景】场景环境：居家视频体验",
                "【显性需求与痛点】业务痛点：网速慢, 看视频卡顿"
                ]
            }
            
            Input: 我是送外卖的，平时电话特别多。现在的套餐只要39块钱，但是流量太少了不够用。你们上个月给我推的那个129的冰激凌套餐太贵了，有没有大概60块钱左右流量多点的？
            Output: {
                "facts": [
                "【基础属性与资格】职业：送外卖",
                "【行为特征与场景】资源消耗：语音通话多, 场景环境：送外卖的过程中",
                "【显性需求与痛点】核心诉求：增加流量, 预算范围：60元左右, 业务痛点：流量不够用,消费习惯：对价格敏感",
                "【交互反馈与决策历史】拒绝套餐：曾拒绝129元冰激凌套餐（原因：太贵）"
                ]
            }
            
            Input: 我想查一下我上个月的话费，顺便问一下我有两张卡，能不能把账单合在一起交？
            Output: {
                "facts": [
                "【基础属性与资格】设备信息：双卡用户",
                "【显性需求与痛点】核心诉求：查询话费,主副卡/账单融合"
                ]
            }
            """
        system_fact_extraction_prompt = """
            你是一个电信推荐系统的“记忆管理员”。你的任务是从对话中提取关键事实，特别是要**高度压缩**系统的回复内容。
            
            ### 压缩规则 (针对系统回复)：
            1. **忽略废话**：绝对不要提取问候语（如“您好”）、客套话（如“很高兴为您服务”）或通用的营销修辞。
            2. **提取动作**：将系统的长回复转化为以下简短格式：
               - "系统推荐了：[套餐名称] (价格：[X]元)"
               - "系统解释了：[套餐名称] 的 [某功能] (原因：[X])"
               - "系统拒绝了：[用户请求] (原因：[X])"
               - "系统询问了：[具体问题]"
            3. **保留关键数据**：必须保留金额、流量数值、合约期长度等硬性指标。
            4. **输出格式**：要将所有内容都压缩到一个字符串当中，然后将这个字符串存储在"facts"对应的列表当中。即"facts"当中最多只能有一个字符串。
                           请仅以 JSON 格式返回结果。JSON 的格式必须严格对应下面的参考样例中的Output。。
            
            ### 示例 (Few-Shot Examples)：
            
            Input: 
            Assistant: 尊敬的用户您好，根据您刚才提到的平时刷抖音比较多，我强烈向您推荐我们的“5G畅享卡”。这个套餐非常划算，每个月只要59元，包含200G的定向流量，专门覆盖抖音、快手等APP，而且现在办理还送一个月会员，您看怎么样？
            Output: 
            {"facts": ["系统推荐了：5G畅享卡 (59元/月),系统强调卖点：200G定向流量 (覆盖抖音),系统提及优惠：赠送首月会员"]}
            
            Input: 
            Assistant: 非常抱歉，因为您的号码还有未到期的合约，所以暂时不能办理这个19元的保号套餐，您需要等到明年3月份合约结束后再来申请。
            Output: 
            {"facts": ["系统拒绝办理：19元保号套餐,拒绝原因：存在未到期合约,时间节点：需明年3月后申请"]}
        """
        try:

            from mem0 import Memory

            mem0_config = {
                "llm": {
                    "provider": cfg.MODEL_PROVIDER,
                    "config": {
                        "model": cfg.MODEL_NAME,
                        "api_key": cfg.DASHSCOPE_API_KEY,
                        "openai_base_url": cfg.OPENAI_BASE_URL,
                        "temperature": 0.1,
                        "max_tokens": 2000,
                        "top_p": 1.0
                    }
                },
                "custom_fact_extraction_prompt": custom_fact_extraction_prompt,
                "embedder": {
                    # 采用 openai 兼容模式接入阿里 dashscope embedding-v4
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-v4",
                        "api_key": cfg.DASHSCOPE_API_KEY,
                        "embedding_dims": 1024,
                        "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    },
                },
                "vector_store": {
                    "provider": "milvus",
                    "config": {
                        "collection_name": cfg.MILVUS_DB_NAME,
                        "embedding_model_dims": 1024,
                        "url": "http://localhost:19530",
                        "token": "sec"
                    },
                },
                # 可选命名空间，若版本不支持会忽略
                "namespace": namespace,
            }
            mem0_config_system = {
                "llm": {
                    "provider": cfg.MODEL_PROVIDER,
                    "config": {
                        "model": cfg.MODEL_NAME,
                        "api_key": cfg.DASHSCOPE_API_KEY,
                        "openai_base_url": cfg.OPENAI_BASE_URL,
                        "temperature": 0.1,
                        "max_tokens": 2000,
                        "top_p": 1.0
                    }
                },
                "custom_fact_extraction_prompt": system_fact_extraction_prompt,
                "embedder": {
                    # 采用 openai 兼容模式接入阿里 dashscope embedding-v4
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-v4",
                        "api_key": cfg.DASHSCOPE_API_KEY,
                        "embedding_dims": 1024,
                        "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    },
                },
                "vector_store": {
                    "provider": "milvus",
                    "config": {
                        "collection_name": cfg.MILVUS_DB_NAME,
                        "embedding_model_dims": 1024,
                        "url": "http://localhost:19530",
                        "token": "sec"
                    },
                },
                # 可选命名空间，若版本不支持会忽略
                "namespace": namespace,
            }

            # openai 兼容客户端常要求环境变量
            os.environ.setdefault("OPENAI_API_KEY", cfg.DASHSCOPE_API_KEY)
            os.environ.setdefault("OPENAI_BASE_URL", cfg.OPENAI_BASE_URL)

            # 新版推荐写法
            if hasattr(Memory, "from_config"):
                self.client = Memory.from_config(mem0_config)
                self.system_client = Memory.from_config(mem0_config_system)
            self.available = True
            logger.info("mem0记忆服务初始化成功（Memory.from_config, deepseek + embedding-v4 + Milvus）")
        except ImportError:
            logger.warning("mem0未安装，使用No-Op模式")
        except Exception as e:
            self.client = None
            self.available = False
            logger.error(f"mem0初始化失败，使用No-Op模式: {e}", exc_info=True)

    def add(
            self,
            user_id: str,
            messages: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加记忆
        
        Args:
            user_id: 用户ID（电话号码）
            messages: 记忆文本
            metadata: 元数据
            
        Returns:
            是否成功
        """
        try:
            if metadata is None:
                metadata = {}

            # 添加时间戳 方便自定义时间格式，便于阅览
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.now().isoformat()

            if "role" not in metadata:
                metadata["role"] = "user"

            self.client.add(user_id=user_id, messages=messages, metadata=metadata)
            logger.info(f"记忆已添加: {user_id} - {messages[:50]}...")
            return True
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            return False

    def system_add(
            self,
            user_id: str,
            messages: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加系统回复记忆

        Args:
            user_id: 用户ID（电话号码）
            messages: 记忆文本
            metadata: 元数据

        Returns:
            是否成功
        """
        try:
            if metadata is None:
                metadata = {}

            # 添加时间戳
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.now().isoformat()

            if "role" not in metadata:
                metadata["role"] = "assistant"

            self.system_client.add(user_id=user_id, messages=messages, metadata=metadata)
            logger.info(f"记忆已添加: {user_id} - {messages[:50]}...")
            return True
        except Exception as e:
            logger.error(f"添加记忆失败: {e}")
            return False


    def search(
            self,
            user_id: str,
            query: str,
            top_k: int = 5,
            filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        搜索记忆
        
        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            记忆列表
        """
        try:
            results = self.client.search(
                user_id=user_id,
                query=query,
                limit=top_k,
                filters=filters
            )
            logger.info(f"检索到 {len(results)} 条记忆")
            return results
        except Exception as e:
            logger.error(f"搜索记忆失败: {e}")
            return {}

    def delete_user(self, user_id: str) -> bool:
        """
        删除用户所有记忆
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否成功
        """
        try:
            self.client.delete_user(user_id=user_id)
            logger.info(f"已删除用户 {user_id} 的所有记忆")
            return True
        except Exception as e:
            logger.error(f"删除记忆失败: {e}")
            return False

    def get_all_user_memory(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户画像记忆
        Args:
            user_id: 用户ID
        Returns:
            记忆列表
        """
        try:
            # 注意get_all函数现在必须指定filters
            results = self.client.get_all(filters={"role": "assistant"}, user_id=user_id)
            return results.get("results")
        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return []

    def get_all_user_system_memory(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户相关系统回复记忆
        Args:
            user_id: 用户ID
        Returns:
            记忆列表
        """
        try:
            # 注意get_all函数现在必须指定filters
            results = self.client.get_all(filters={"role": "assistant"}, user_id=user_id)
            return results.get("results")
        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return []
# 全局单例
_memory_service_instance = None


def get_memory_service(api_key: Optional[str] = None) -> MemoryService:
    """获取记忆服务单例"""
    global _memory_service_instance
    if _memory_service_instance is None:
        _memory_service_instance = MemoryService()
    return _memory_service_instance

