from typing import List, Dict

def hit_at_k(preds: List[str], gold: List[str], k: int) -> float:
    topk = preds[:k]
    return 1.0 if any(p in gold for p in topk) else 0.0

def mrr(preds: List[str], gold: List[str]) -> float:
    for i, p in enumerate(preds, start=1):
        if p in gold:
            return 1.0 / i
    return 0.0

def ndcg_at_k(preds: List[str], gold: List[str], k: int) -> float:
    import math
    dcg = 0.0
    for i, p in enumerate(preds[:k], start=1):
        if p in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1)) or 1.0
    return dcg / idcg