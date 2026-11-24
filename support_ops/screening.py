"""筛查模型、按时点回放与可信范围检查。

筛查标签只用未受邀约学员的后测，因为受邀约者是否进步被辅导本身改变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cohort import Record
from .effects import Z, ols

SKILLS = ("listen", "explain", "respond")


def feature_matrix(records: Sequence[Record]) -> np.ndarray:
    rows = []
    for r in records:
        values = [r.screened.get(skill, 60.0) for skill in SKILLS]
        rows.append(
            values
            + [
                r.screened_mean,
                r.screened_min,
                float(sum(1 for value in values if value < 60)),
                float(r.track == "general"),
                float(r.track == "community"),
            ]
        )
    return np.array(rows, float)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), x])
    beta = np.zeros(design.shape[1])
    for _ in range(iters):
        p = predict_proba(beta, design)
        weight = p * (1 - p) + 1e-9
        hessian = design.T @ (design * weight[:, None]) + l2 * np.eye(design.shape[1])
        gradient = design.T @ (y - p) - l2 * beta
        beta = beta + np.linalg.solve(hessian, gradient)
    return beta


def predict_proba(beta: np.ndarray, design: np.ndarray) -> np.ndarray:
    if design.shape[1] != beta.shape[0]:
        design = np.column_stack([np.ones(len(design)), design])
    return 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))


def auc(y: np.ndarray, score: np.ndarray) -> Optional[float]:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def confusion(y: np.ndarray, flagged: np.ndarray) -> Dict[str, object]:
    tp = int(np.sum(flagged & (y == 1)))
    fp = int(np.sum(flagged & (y == 0)))
    fn = int(np.sum(~flagged & (y == 1)))
    tn = int(np.sum(~flagged & (y == 0)))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "ppv": tp / (tp + fp) if tp + fp else None,
        "flag_rate": (tp + fp) / len(y) if len(y) else None,
    }


@dataclass
class RiskModel:
    mu: np.ndarray
    sd: np.ndarray
    beta: np.ndarray
    threshold: float = 0.5

    def predict(self, records: Sequence[Record]) -> np.ndarray:
        x = (feature_matrix(records) - self.mu) / self.sd
        return predict_proba(self.beta, x)

    def flag(self, records: Sequence[Record]) -> np.ndarray:
        return self.predict(records) >= self.threshold


def risk_model(train: Sequence[Record], threshold: float = 0.5) -> Optional[RiskModel]:
    """只用成熟对照训练的二分类模型；标签是“常规练习下后测仍低于 60”。"""
    pool = [r for r in train if not r.assigned and r.post_independent]
    if len(pool) < 20 or len({r.post_risk for r in pool}) < 2:
        return None
    x = feature_matrix(pool)
    y = np.array([1.0 if r.post_risk else 0.0 for r in pool])
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    return RiskModel(mu, sd, fit_logistic((x - mu) / sd, y), threshold)


def backtest(records: Sequence[Record], threshold: float = 0.5) -> List[Dict[str, object]]:
    """按时点回放：每轮只用此前已成熟的对照训练，再检查当轮对照。

    当轮被邀约者不进入训练或评价；他们在常规练习下会怎样不可观察。
    """
    pool = [r for r in records if not r.assigned and r.post_independent]
    output = []
    for wave in sorted({r.wave for r in pool}):
        train = [r for r in pool if r.wave < wave]
        test = [r for r in pool if r.wave == wave]
        row: Dict[str, object] = {
            "wave": wave,
            "train_n": len(train),
            "test_n": len(test),
            "events": int(sum(1 for r in test if r.post_risk)),
        }
        model = risk_model(train, threshold)
        if model is None:
            row["trained"] = False
            row["reason"] = "此前轮次对照不足，无法训练"
            output.append(row)
            continue
        row["trained"] = True
        prob = model.predict(test)
        truth = np.array([1.0 if r.post_risk else 0.0 for r in test])
        row.update({"auc": auc(truth, prob), "brier": float(np.mean((prob - truth) ** 2)),
                    "mean_pred": float(prob.mean()), "base_rate": float(truth.mean()),
                    **confusion(truth, prob >= threshold)})
        output.append(row)
    return output


def model_comparison(records: Sequence[Record]) -> List[Dict[str, object]]:
    """规则式与模型式筛查在同一批成熟对照上的比较。"""
    pool = [r for r in records if not r.assigned and r.post_independent]
    if not pool:
        return []
    truth = np.array([1.0 if r.post_risk else 0.0 for r in pool])
    rules = {
        "任一技能低于 60": lambda r: any(v < 60 for v in r.screened.values()),
        "均分低于 60": lambda r: r.screened_mean < 60,
        "最低技能低于 60": lambda r: r.screened_min < 60,
        "至少两项低于 60": lambda r: sum(1 for v in r.screened.values() if v < 60) >= 2,
    }
    output = []
    for name, rule in rules.items():
        output.append({"model": name, "n": len(pool),
                       **confusion(truth, np.array([rule(r) for r in pool]))})
    model = risk_model(pool)
    if model is not None:
        prob = model.predict(pool)
        output.append({"model": "基线调整分数模型", "n": len(pool),
                       "auc": auc(truth, prob), "brier": float(np.mean((prob - truth) ** 2)),
                       **confusion(truth, prob >= model.threshold)})
    return output


def coverage_check(records: Sequence[Record]) -> List[Dict[str, object]]:
    """按轮次扩窗：此前累计的收益区间是否覆盖当轮单独估计。"""
    pool = [r for r in records if r.post_independent and r.policy in ("P1", "P2")]
    output = []
    waves = sorted({r.wave for r in pool})

    def estimate(subset: Sequence[Record]):
        if len(subset) < 30:
            return None
        y = np.array([r.post_value - r.screened_mean for r in subset])
        assigned = np.array([1.0 if r.assigned else 0.0 for r in subset])
        if assigned.sum() < 5 or (1 - assigned).sum() < 5:
            return None
        base = np.array([r.screened_mean for r in subset])
        prob = np.array([1.0 if r.prob >= 0.5 else 0.0 for r in subset])
        w = np.array([r.wave for r in subset], float)
        cols = [base, prob] + [(w == level).astype(float) for level in sorted(set(w))[1:]]
        beta, se = ols(y, np.column_stack([assigned] + cols))
        return float(beta[1]), float(se[1])

    for wave in waves:
        prior = estimate([r for r in pool if r.wave < wave])
        current = estimate([r for r in pool if r.wave == wave])
        if prior is None:
            output.append({"wave": wave, "checked": False, "reason": "此前轮次不足以估计区间"})
            continue
        if current is None:
            output.append({"wave": wave, "checked": False, "reason": "当轮不足以单独估计"})
            continue
        low, high = prior[0] - Z * prior[1], prior[0] + Z * prior[1]
        output.append({"wave": wave, "checked": True, "prior": prior[0], "prior_low": low,
                       "prior_high": high, "current": current[0],
                       "covered": bool(low <= current[0] <= high)})
    return output


def availability_by_track(records: Sequence[Record]) -> List[Dict[str, object]]:
    """分轮转与练习能力报告筛查可用性：样本、事件数、区间宽度。"""
    output = []
    mature = [r for r in records if r.post_independent and r.policy in ("P1", "P2")]
    for track in ("surgery", "general", "community"):
        for band, (low, high) in (("低于 60", (0, 60)), ("60 及以上", (60, 101))):
            sub = [r for r in mature if r.track == track and low <= r.screened_mean < high]
            treated = [r for r in sub if r.assigned]
            control = [r for r in sub if not r.assigned]
            row: Dict[str, object] = {
                "track": track, "band": band, "n": len(sub),
                "assigned": len(treated), "control": len(control),
            }
            if len(treated) >= 10 and len(control) >= 10:
                y = np.array([r.post_value - r.screened_mean for r in sub])
                a = np.array([1.0 if r.assigned else 0.0 for r in sub])
                b = np.array([r.screened_mean for r in sub])
                p = np.array([1.0 if r.prob >= 0.5 else 0.0 for r in sub])
                w = np.array([r.wave for r in sub], float)
                cols = [b, p] + [(w == level).astype(float) for level in sorted(set(w))[1:]]
                beta, se = ols(y, np.column_stack([a] + cols))
                row.update({"itt": float(beta[1]),
                            "low": float(beta[1] - Z * se[1]),
                            "high": float(beta[1] + Z * se[1]),
                            "width": float(2 * Z * se[1])})
            else:
                row["reason"] = "该组合的邀约组或对照组不足，无法给出收益区间"
            output.append(row)
    return output
