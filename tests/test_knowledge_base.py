"""
知识库测试脚本
"""
import os
import sys
from milvus.rag_knowledge_base import RAGKnowledgeBase

def test_search():
    """测试搜索功能"""
    print("=" * 50)
    print("RAG知识库搜索测试")
    print("=" * 50)
    
    # 创建知识库实例
    kb = RAGKnowledgeBase()
    kb.connect()

    # 测试查询
    test_queries = [
        "老年用户推荐什么套餐",
        "5G终端用户需要什么产品",
        "流量不够用怎么办",
        "视频用户推荐什么服务",
        "家庭网络优化方案",
        "高端客户有什么特权",
        "语音通话超出费用",
        "学生用户教育服务"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 30)
        
        results = kb.search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   分类: {result['category']}")
                print(f"   相似度: {result['similarity_score']:.4f}")
                print(f"   触发条件: {result['trigger_conditions']}")
                print(f"   推荐产品: {result['recommended_products']}")
                print()
        else:
            print("未找到相关结果")
    
    return True

def main():
    """主函数"""
    # if not os.getenv("DASHSCOPE_API_KEY"):
    #     print("错误: 请设置环境变量 DASHSCOPE_API_KEY")
    #     return False
    
    return test_search()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
