"""
知识库构建脚本
"""
import os
import sys

# 添加项目根目录到sys.path，解决导入问题
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from milvus.rag_knowledge_base import RAGKnowledgeBase

def main():
    """主函数"""
    print("=" * 50)
    print("电信产品推荐原则 RAG知识库构建")
    print("=" * 50)
    
    # 检查环境变量
    # if not os.getenv("DASHSCOPE_API_KEY"):
    #     print("错误: 请设置环境变量 DASHSCOPE_API_KEY")
    #     print("示例: export DASHSCOPE_API_KEY=your_api_key")
    #     return False
    
    # 检查文档文件
    if not os.path.exists("../sources/电信产品推荐原则.md"):
        print("错误: 找不到源文档文件 sources/电信产品推荐原则.md")
        return False
    
    # 创建知识库实例
    kb = RAGKnowledgeBase()
    
    # 初始化知识库
    print("正在初始化知识库...")
    if not kb.initialize():
        print("知识库初始化失败")
        return False
    print("知识库初始化成功")
    
    # 构建知识库
    print("正在构建知识库...")
    if not kb.build_knowledge_base():
        print("知识库构建失败")
        return False
    
    # 显示统计信息
    stats = kb.get_stats()
    if stats:
        print(f"知识库构建完成，包含 {stats} 条记录")
    
    print("=" * 50)
    print("知识库构建成功！")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)