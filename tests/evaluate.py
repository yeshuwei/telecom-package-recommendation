"""
评估脚本：对比 baseline / ablation_no_slot / ours_full 三种配置
运行方式：python tests/evaluate.py
"""
import json
import sys
import os
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.ranking_metrics import hit_at_k, mrr, ndcg_at_k
from metrics.business_metrics import budget_violation_rate, constraint_satisfaction_rate
from metrics.significance_test import paired_t_test

TEST_CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_test_cases():
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(config_name: str) -> dict:
    config_path = os.path.join(CONFIGS_DIR, f"{config_name}.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_workflow_result(final_state: dict) -> dict:
    """
    从工作流最终状态中提取评估所需字段。
    - pred_packages: 推荐套餐名称列表（有序，取 price_selection_results）
    - pred_price: 第一个推荐套餐的价格
    - pred_attrs: 推荐套餐的属性（用于约束满足率检验）
    """
    price_results = final_state.get("price_selection_results", [])

    pred_packages = []
    pred_price = None
    pred_attrs = {}

    for item in price_results:
        # item 结构示例：{"series": "5G畅享套餐", "plan_name": "5G畅享199", "price": 199, ...}
        name = item.get("plan_name") or item.get("series", "")
        if name:
            pred_packages.append(name)
        # 也把 series 名加入，方便与 gold_packages 对比
        series = item.get("series", "")
        if series and series not in pred_packages:
            pred_packages.append(series)

    if price_results:
        first = price_results[0]
        pred_price = first.get("price")
        # 从第一个推荐结果中提取属性，用于约束满足率
        pred_attrs = {
            "need_5g": "5G" in (first.get("series", "") + first.get("plan_name", "")),
            "need_broadband": "融合" in (first.get("series", "") + first.get("plan_name", "")),
        }

    return {
        "pred_packages": pred_packages,
        "pred_price": pred_price,
        "pred_attrs": pred_attrs,
    }


def run_system(test_case: dict, config_name: str) -> dict:
    """
    根据配置调用工作流，返回推荐结果。
    """
    # 动态加载配置，根据 config 决定是否启用各模块
    cfg = load_config(config_name)
    system_cfg = cfg.get("system", {})

    # baseline 模式：纯规则，不走 LangGraph
    if system_cfg.get("mode") == "rule":
        return _run_baseline_rule(test_case)

    # agent 模式：走 LangGraph 工作流
    from agent.workflow_graph import app
    from agent.state import AgentState

    initial_state: AgentState = {
        "phone_number": test_case["phone_number"],
        "input": test_case["input"],
        "chat_history": [],
        # 将配置注入 state，供各 agent 读取开关
        "_config": system_cfg,
    }

    final_state = app.invoke(initial_state)
    return parse_workflow_result(final_state)


def _run_baseline_rule(test_case: dict) -> dict:
    """
    Baseline：纯关键词规则推荐，不调用 LLM。
    仅作对比基线，逻辑尽量简单。
    """
    user_input = test_case["input"]
    pred_packages = []

    keyword_map = [
        (["5G", "五G"], "5G畅享套餐"),
        (["宽带", "融合"], "家庭融合套餐"),
        (["老年", "老人", "65"], "老年关爱套餐"),
        (["军人", "部队"], "军人优惠套餐"),
        (["贫困", "困难", "低保"], "惠民套餐"),
        (["流量", "不够", "溢出"], "流量王套餐"),
        (["视频", "抖音", "爱奇艺", "直播"], "视频畅享套餐"),
        (["通话", "语音", "打电话"], "畅聊套餐"),
        (["学生", "网课", "教育"], "教育优惠套餐"),
    ]

    for keywords, package in keyword_map:
        if any(kw in user_input for kw in keywords):
            pred_packages.append(package)

    if not pred_packages:
        pred_packages = ["基础套餐"]

    return {
        "pred_packages": pred_packages,
        "pred_price": None,
        "pred_attrs": {},
    }


def evaluate_config(test_cases: list, config_name: str) -> dict:
    """对单个配置跑全部测试用例，返回各指标"""
    hit1_scores, hit3_scores, mrr_scores, ndcg3_scores = [], [], [], []
    budget_samples, constraint_samples = [], []

    for tc in test_cases:
        gold = tc.get("gold_packages", [])
        if not gold:
            continue  # 无标注答案的用例跳过排名指标

        try:
            result = run_system(tc, config_name)
        except NotImplementedError:
            raise
        except Exception as e:
            print(f"[WARN] {tc['id']} 运行失败: {e}")
            continue

        preds = result.get("pred_packages", [])
        hit1_scores.append(hit_at_k(preds, gold, k=1))
        hit3_scores.append(hit_at_k(preds, gold, k=3))
        mrr_scores.append(mrr(preds, gold))
        ndcg3_scores.append(ndcg_at_k(preds, gold, k=3))

        # 业务指标
        constraints = tc.get("constraints", {})
        if "budget_max" in constraints:
            budget_samples.append({
                "budget_max": constraints["budget_max"],
                "pred_price": result.get("pred_price")
            })
        if "must" in constraints:
            constraint_samples.append({
                "must": constraints["must"],
                "pred_attrs": result.get("pred_attrs", {})
            })

    n = len(hit1_scores) or 1
    return {
        "config": config_name,
        "n_evaluated": n,
        "Hit@1": sum(hit1_scores) / n,
        "Hit@3": sum(hit3_scores) / n,
        "MRR": sum(mrr_scores) / n,
        "NDCG@3": sum(ndcg3_scores) / n,
        "BudgetViolationRate": budget_violation_rate(budget_samples),
        "ConstraintSatisfactionRate": constraint_satisfaction_rate(constraint_samples),
        # 保留逐样本分数用于显著性检验
        "_hit1_scores": hit1_scores,
        "_mrr_scores": mrr_scores,
    }


def print_results(results: list):
    print("\n{'='*60}")
    print(f"{'Config':<25} {'Hit@1':>7} {'Hit@3':>7} {'MRR':>7} {'NDCG@3':>8} {'BudgetVio':>10} {'ConstrSat':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['config']:<25} "
            f"{r['Hit@1']:>7.3f} "
            f"{r['Hit@3']:>7.3f} "
            f"{r['MRR']:>7.3f} "
            f"{r['NDCG@3']:>8.3f} "
            f"{r['BudgetViolationRate']:>10.3f} "
            f"{r['ConstraintSatisfactionRate']:>10.3f}"
        )
    print("=" * 80)


def significance_tests(results: list):
    """对 ours_full vs 其他配置做配对t检验"""
    ours = next((r for r in results if r["config"] == "ours_full"), None)
    if not ours:
        return
    print("\n显著性检验 (paired t-test, Hit@1):")
    for r in results:
        if r["config"] == "ours_full":
            continue
        a = ours["_hit1_scores"]
        b = r["_hit1_scores"]
        if len(a) != len(b) or len(a) == 0:
            print(f"  ours_full vs {r['config']}: 样本数不匹配，跳过")
            continue
        t, p = paired_t_test(a, b)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
        print(f"  ours_full vs {r['config']}: t={t:.3f}, p={p:.4f} {sig}")


def main():
    test_cases = load_test_cases()
    configs = ["baseline", "ablation_no_slot", "ours_full"]
    results = []
    for cfg in configs:
        print(f"正在评估配置: {cfg} ...")
        r = evaluate_config(test_cases, cfg)
        results.append(r)

    print_results(results)
    significance_tests(results)


if __name__ == "__main__":
    main()
