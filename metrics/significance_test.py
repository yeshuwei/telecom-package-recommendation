from typing import List, Tuple
from scipy.stats import ttest_rel

def paired_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """
    返回 (t_stat, p_value)
    a/b: 两个模型在同一批样本上的逐样本得分（如Hit@1）
    """
    t_stat, p_val = ttest_rel(a, b)
    return float(t_stat), float(p_val)