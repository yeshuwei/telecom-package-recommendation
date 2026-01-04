"""
mem0 基础功能小demo：
1) 初始化 MemoryService（使用 configs/config.py 中的 Gemini + embedding-v4 + Milvus 配置）
2) 添加一条记忆
3) 检索并打印结果
运行：python -m new_agent.test
"""
import json
import os
from openai import OpenAI
from new_agent.memory_service import get_memory_service
from new_agent.recommendation_agent import RecommendationAgent
memory_service = get_memory_service()
recommendation_agent = RecommendationAgent()
response_message = ["""您好，根据我们之前的沟通和您的使用情况，我为您进行了深入的分析。
您目前使用的是239元套餐，每月流量消耗都超过35GB，主要用来看高清视频、刷短视频和直播。虽然您已经是千兆宽带用户，但手机流量的需求依然旺盛，而且您也提到希望将月费控制在100-150元左右。
结合您的这些核心需求——**“更多流量看高清视频”**和**“优化月费支出”**，我为您筛选出了一个目前非常匹配的方案。
### **为您重点推荐：徽金卡单品（权益版）**
这个系列套餐的核心优势在于：**“加量不加价，甚至还能省钱”**。它不是一个简单的手机套餐，而是将您每月的话费，转化成了“通信基础服务 + 额外流量 + 实用权益”的组合包，非常适合像您这样流量消耗大、且消费活跃的用户。
**为什么它特别适合您？**
1.  **流量大幅升级，看视频更自由**：您当前套餐流量约40GB（239元档），而徽金卡在您的预算区间内，能提供**翻倍甚至更多的流量**。例如129元档，就包含30GB国内流量+24个月每月30GB园区流量，每月可用流量达60GB，完全能满足您刷高清视频的需求，告别流量焦虑。
2.  **月费显著下降，符合您的预算**：您提到的100-150元预算，在这个系列中有**99元、129元、169元**多个档位可供选择，都能在降低您当前月费的同时，提供更充裕的流量。
3.  **权益实用，直接回馈消费**：套餐包含的翼支付商户券、话费券或电商券，您在日常购物、充话费时可以直接抵扣，相当于每月话费的一部分又“返还”给了您，进一步提升了性价比。虽然您对影视会员需求不明显，但附赠的周卡权益也可以作为额外福利。
### **具体套餐倾向性建议：**
考虑到您**月均流量35GB+**，且追求**高清视频体验**，我建议您可以重点关注 **129元档** 或 **169元档**：
*   **【129元档】**：月费 **129元**。包含 **60GB/月** 总流量（30GB国内+30GB园区），以及500分钟通话。这个档位完美契合您的预算上限，流量相比现在提升50%，是“降费增流”的性价比之选。
*   **【169元档】**：月费 **169元**。包含 **70GB/月** 总流量（40GB国内+30GB园区），以及800分钟通话。如果您希望流量更加宽裕，为未来可能增长的视频需求留足空间，这个档位体验会更无忧。
**总结一下：** 转办徽金卡系列，您可以在**每月节省70-110元话费**的基础上，获得**更多可用流量**，并额外获得消费抵扣权益，实现通信成本和体验的双重优化。
您看，对于 **129元档** 和 **169元档**，哪个档位的流量和预算更符合您的预期呢？我们可以现在就为您测算一下具体的办理流程和优惠。
""",
                    """您好！很高兴再次为您服务。结合我们之前的沟通和您的使用情况，我为您做了更深入的分析。

您目前使用的是239元套餐，月均消费241元，流量使用约35GB，并且已经是我们尊贵的千兆宽带用户。我特别注意到，您经常观看高清视频、使用抖音等应用，对流量的需求确实很大，而且您之前提到过希望月费在100-150元左右。上次推荐的纯流量卡可能不太符合您对宽带和综合权益的需求，这次我为您筛选了更匹配的方案。

基于您“大流量、千兆宽带、内容权益”这三大核心需求，我为您重点推荐 **【5G畅享融合套餐（含千兆宽带）】** 系列。这个系列完美整合了您正在使用的手机和宽带业务，不仅能将两者合账付费、管理更方便，更重要的是，它能以更优的整体价值，满足您对高速网络和充裕流量的追求。

在这个系列中，我特别建议您关注 **199元档** 和 **239元档**：
*   **199元档**：月费**199元**，包含**60GB国内流量**、1000分钟通话，并继续提供您正在使用的**1000M千兆宽带**和一部超清iTV。对比您当前约241元的月均消费，此套餐直接为您节省了约40元，而流量却从您目前实际使用的约35GB提升到了60GB，能有效避免流量焦虑，非常适合您观看高清视频的习惯。
*   **239元档**：月费**239元**，包含**80GB国内流量**、1000分钟通话，同样包含千兆宽带和iTV。这个档位与您当前消费持平，但流量供给更加充裕，让您可以更随心所欲地刷短视频、看直播。

**给您的建议：**
考虑到您明确的预算倾向（100-150元）和当前的实际消费，**199元档**的性价比尤为突出。它在价格低于您当前消费的同时，提供了翻倍的流量保障和您已依赖的千兆宽带，是“降支出、升体验”的优选。当然，如果您希望流量储备更加宽裕，**239元档**也是保持消费不变、全面升级体验的稳健之选。

您看，对于**199元**这个档位的套餐，或者整个融合套餐的模式，您觉得怎么样？我可以为您进一步计算更详细的费用对比。"""]

def test_connection():
    client = OpenAI(
        api_key="sk-7a2be908156d44d29071fecfd6dd4cfb",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    response = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "你是谁"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)


user_input = ["我现在找了个送外卖的工作啊，对通话语音需求较高，有没有新的套餐符合我呢？",
              "我平时喜欢看高清电影，流量用得比较多，预算大概100-150元左右，有什么合适的套餐推荐吗？",
              "我现在只看小说，听音乐了，不看什么电影电视剧了，预算只有不到100块，有没有新的套餐符合我呢"]

def test_mem0_storage(text: str):
    svc = get_memory_service()
    if not svc.available:
        print("mem0 不可用，可能未安装或初始化失败")
        return

    user_id = "19156511222"
    # text = "我现在只看小说，听音乐了，不看什么电影电视剧了，预算只有不到100块，有没有新的套餐符合我呢"
    metadata = {
        "role": "user"
    }

    ok = svc.add(user_id=user_id, messages=text, metadata=metadata)
    print(f"add -> {ok}")

    # all_memory = svc.get_all_user_memory(user_id)
    #
    # print(json.dumps(all_memory, ensure_ascii=False, indent=2))
    # results = svc.search(user_id=user_id, query="用户最新的【显性需求】是？", top_k=2, filters={"role": "user"})
    # print("search results:")
    # for i, r in enumerate(results.get("results")):
    #     print(f"{i + 1}. memory={r.get('memory')} meta={r.get('metadata')}")


def test_mem0_system_storage():
    # memory_service.system_add(user_id="test_user", messages="")
    result = memory_service.get_all_user_system_memory(user_id="19156511222")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def test_rewrite_user_input():
    user_input_text = "我现在只看小说，听音乐了，不看什么电影电视剧了，预算只有不到100块，有没有新的套餐符合我呢"
    rewrite_input = recommendation_agent.rewrite_user_input(user_id="19156511222", user_input=user_input_text)
    print(rewrite_input)
    return rewrite_input


def test_run_demo():
    user_input = "我平时喜欢看高清电影，流量用得比较多，预算大概100-150元左右，有什么合适的套餐推荐吗？"


if __name__ == "__main__":
    text = test_rewrite_user_input()
    # test_mem0_system_storage()
    # test_run_demo()
    test_mem0_storage(text)