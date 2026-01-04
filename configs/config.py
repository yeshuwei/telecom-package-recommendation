"""
RAG知识库配置文件
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录 - 从当前文件位置向上一级
PROJECT_ROOT = Path(__file__).parent.parent

# Milvus配置
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MILVUS_DB_NAME = "telecom_recommendation"

# llm配置（使用阿里云 DashScope 的 OpenAI 兼容接口）
MODEL_NAME = "qwen-flash"  # 阿里云千问极速模型
MODEL_PROVIDER = "openai"   # 通过 OpenAI 兼容协议调用
# 阿里云 DashScope 兼容接口配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-7a2be908156d44d29071fecfd6dd4cfb")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 兼容旧字段，避免引用报错（已不再使用 DeepSeek）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 集合配置
COLLECTION_NAME = "telecom_principles"
DIMENSION = 1024  # text-embedding-v4的向量维度

# 文档配置 - 使用绝对路径
SOURCE_DOC_PATH = str(PROJECT_ROOT / "sources" / "电信产品推荐原则.md")
PRICE_SHEET_PATH = str(PROJECT_ROOT / "sources" / "副本安徽在售资费.xlsx")
SERIES_INTRO_MD_PATH = str(PROJECT_ROOT / "sources" / "套餐简介_smDkXtsMAi7TacipsvazkS.md")