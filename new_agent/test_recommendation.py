"""
测试新的推荐流程
"""
import logging
from new_agent.recommendation_agent import RecommendationAgent

# 设置日志级别，方便查看流程
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_recommendation_flow():
    """
    测试完整的推荐流程
    """
    print("=" * 70)
    print("开始测试推荐流程")
    print("=" * 70)

    # 1. 创建推荐代理实例
    agent = RecommendationAgent()

    # 2. 模拟用户输入
    user_id = "19156511222"  # 测试用的用户ID（电话号码）
    user_input = "我现在找了个送外卖的工作啊，对通话语音需求较高，有没有新的套餐符合我呢？"

    print(f"\n【用户输入】")
    print(f"用户ID: {user_id}")
    print(f"用户说: {user_input}")
    print("\n" + "-" * 70)

    # 3. 执行推荐流程
    try:
        response = agent.run(user_id, user_input)

        print(f"\n【AI推荐回复】")
        print(response)
        print("\n" + "=" * 70)
        print("测试完成")
        print("=" * 70)

        return response
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)
        print(f"\n❌ 测试失败: {e}")
        return None


user_input = ["我现在找了个送外卖的工作啊，对通话语音需求较高，有没有新的套餐符合我呢？",
              "我平时喜欢看高清电影，流量用得比较多，预算大概100-150元左右，有什么合适的套餐推荐吗？",
              "我现在只看小说，听音乐了，不看什么电影电视剧了，预算只有不到100块，有没有新的套餐符合我呢"]


def test_multi_turn_conversation():
    """
    测试多轮对话（模拟连续交互）
    """
    print("\n" + "=" * 70)
    print("开始测试多轮对话")
    print("=" * 70)

    agent = RecommendationAgent()
    user_id = "19156511222"

    # 第一轮：用户提出需求
    print(f"\n【第一轮对话】")
    user_input_1 = user_input[1]
    print(f"用户: {user_input_1}")
    response_1 = agent.run(user_id, user_input_1)
    print(f"AI: {response_1}")

    # 第二轮：用户拒绝某个推荐
    print(f"\n【第二轮对话】")
    user_input_2 = user_input[2]
    print(f"用户: {user_input_2}")
    response_2 = agent.run(user_id, user_input_2)
    print(f"AI: {response_2}")

    # 第三轮：用户进一步明确需求
    print(f"\n【第三轮对话】")
    user_input_3 = user_input[0]
    print(f"用户: {user_input_3}")
    response_3 = agent.run(user_id, user_input_3)
    print(f"AI: {response_3}")

    print("\n" + "=" * 70)
    print("多轮对话测试完成")
    print("=" * 70)


if __name__ == "__main__":
    # 测试单次推荐
    test_recommendation_flow()

    # 如果需要测试多轮对话，取消下面的注释
    # test_multi_turn_conversation()
