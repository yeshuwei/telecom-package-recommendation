from typing import Dict, List, Any

def budget_violation_rate(samples: List[Dict[str, Any]]) -> float:
    # samples: [{"budget_max":100, "pred_price":129}, ...]
    if not samples:
        return 0.0
    violations = 0
    valid = 0
    for s in samples:
        b = s.get("budget_max")
        p = s.get("pred_price")
        if b is None or p is None:
            continue
        valid += 1
        if p > b:
            violations += 1
    return violations / valid if valid else 0.0

def constraint_satisfaction_rate(samples: List[Dict[str, Any]]) -> float:
    # samples: [{"must":{"need_5g":True}, "pred_attrs":{"need_5g":True}}, ...]
    if not samples:
        return 0.0
    ok = 0
    for s in samples:
        must = s.get("must", {})
        pred = s.get("pred_attrs", {})
        satisfied = all(pred.get(k) == v for k, v in must.items())
        ok += int(satisfied)
    return ok / len(samples)