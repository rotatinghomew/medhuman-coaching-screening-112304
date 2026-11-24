"""收益估计、后测风险与缺失处理。

三个对象要分开：

* 当时通常做法下的后测风险：未获邀约学员在常规练习下的结果。
* 邀约意向收益（ITT）：随机邀约带来的平均收益，是转介决策可直接用的量。
* 完成辅导者的额外收益（CACE）：只在存在可信对照（随机邀约）时估计。

实际领取不等于被分配；缺失后测既不当作成功，也不当作失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cohort import Record

Z = 1.959963984540054
KINDS = ("short", "full")


@dataclass
class Estimate:
    label: str
    value: Optional[float]
    low: Optional[float]
    high: Optional[float]
    n: int
    treated: int = 0
    control: int = 0
    note: str = ""

    @property
    def identified(self) -> bool:
        return self.value is not None

    @property
    def excludes_zero(self) -> bool:
        return self.identified and (self.low > 0 or self.high < 0)

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "n": self.n,
            "treated": self.treated,
            "control": self.control,
            "note": self.note,
        }


def ols(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """带常数项的 OLS，HC1 稳健标准误。"""
    design = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    n, k = design.shape
    if n <= k:
        return beta, np.full(k, np.inf)
    xtx = np.linalg.pinv(design.T @ design)
    meat = (design * (resid ** 2)[:, None]).T @ design
    cov = xtx @ meat @ xtx * (n / (n - k))
    return beta, np.sqrt(np.diag(cov))


def _design(records: Sequence[Record], extra: Sequence[np.ndarray] = ()) -> np.ndarray:
    base = np.array([r.screened_mean for r in records], float)
    base2 = (base - 60.0) ** 2 / 100.0
    prob = np.array([1.0 if r.prob >= 0.5 else 0.0 for r in records])
    waves = np.array([r.wave for r in records], float)
    levels = sorted(set(waves))
    fe = [(waves == level).astype(float) for level in levels[1:]]
    return np.column_stack([base, base2, prob] + fe + list(extra))


def randomized(records: Sequence[Record]) -> List[Record]:
    return [
        r
        for r in records
        if r.policy in ("P1", "P2") and r.post_independent
    ]


def _estimate(
    label: str, y: np.ndarray, indicator: np.ndarray, controls: np.ndarray,
    treated: int, control: int, note: str
) -> Estimate:
    if treated < 10 or control < 10:
        return Estimate(label, None, None, None, treated + control, treated, control,
                        "邀约组或对照组不足 10 人，不能给出区间")
    beta, se = ols(y, np.column_stack([indicator, controls]))
    return Estimate(
        label, float(beta[1]), float(beta[1] - Z * se[1]), float(beta[1] + Z * se[1]),
        treated + control, treated, control, note,
    )


def intent_to_treat(records: Sequence[Record]) -> Estimate:
    """随机邀约的平均收益；未完成邀约者仍留在邀约组。"""
    rows = randomized(records)
    if not rows:
        return Estimate("邀约意向收益", None, None, None, 0, 0, 0, "没有成熟的随机邀约轮次")
    y = np.array([r.post_value - r.screened_mean for r in rows])
    assigned = np.array([1.0 if r.assigned else 0.0 for r in rows])
    return _estimate(
        "邀约意向收益（ITT）", y, assigned, _design(rows),
        int(assigned.sum()), int((1 - assigned).sum()),
        "基线调整、轮次固定效应；对照组保留常规练习",
    )


def intent_to_treat_clustered(records: Sequence[Record]) -> Estimate:
    """邀约意向收益，按教练聚类稳健标准误（对照组单独成一簇）。"""
    rows = randomized(records)
    if not rows:
        return Estimate("邀约意向收益（教练聚类）", None, None, None, 0,
                        note="没有成熟的随机邀约轮次")
    y = np.array([r.post_value - r.screened_mean for r in rows])
    assigned = np.array([1.0 if r.assigned else 0.0 for r in rows])
    design = np.column_stack([np.ones(len(rows)), assigned, _design(rows)])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    n, k = design.shape
    clusters = [r.coach or "__control__" for r in rows]
    meat = np.zeros((k, k))
    for cluster in sorted(set(clusters)):
        index = [i for i, value in enumerate(clusters) if value == cluster]
        score = resid[index] @ design[index]
        meat += np.outer(score, score)
    groups = len(set(clusters))
    xtx = np.linalg.pinv(design.T @ design)
    cov = xtx @ meat @ xtx * (groups / (groups - 1)) * ((n - 1) / (n - k))
    se = float(np.sqrt(cov[1, 1]))
    value = float(beta[1])
    return Estimate(
        "邀约意向收益（教练聚类）", value, value - Z * se, value + Z * se, len(rows),
        int(assigned.sum()), int((1 - assigned).sum()),
        f"按 {groups - 1} 位教练聚类；同一教练关联的结果不当独立观察",
    )


def cace(records: Sequence[Record]) -> Tuple[Estimate, Estimate]:
    """以随机邀约为工具变量估计完成辅导者的收益，并附一阶段。"""
    rows = randomized(records)
    if not rows:
        empty = Estimate("完成辅导者的额外收益", None, None, None, 0)
        return empty, empty
    y = np.array([r.post_value - r.screened_mean for r in rows])
    assigned = np.array([1.0 if r.assigned else 0.0 for r in rows])
    completed = np.array([1.0 if r.completed else 0.0 for r in rows])
    controls = _design(rows)
    first_beta, first_se = ols(completed, np.column_stack([assigned, controls]))
    first = Estimate(
        "邀约对完成的作用（一阶段）", float(first_beta[1]),
        float(first_beta[1] - Z * first_se[1]), float(first_beta[1] + Z * first_se[1]),
        len(rows), int(assigned.sum()), int((1 - assigned).sum()),
    )
    design_z = np.column_stack([np.ones(len(rows)), controls, assigned])
    design_x = np.column_stack([np.ones(len(rows)), controls, completed])
    pz = design_z @ np.linalg.pinv(design_z.T @ design_z) @ design_z.T
    beta = np.linalg.pinv(design_x.T @ pz @ design_x) @ (design_x.T @ pz @ y)
    resid = y - design_x @ beta
    n, k = design_x.shape
    xtx = np.linalg.pinv(design_x.T @ pz @ design_x)
    meat = (design_x * (resid ** 2)[:, None]).T @ design_x
    cov = xtx @ meat @ xtx * (n / (n - k))
    se = float(np.sqrt(cov[-1, -1]))
    estimate = Estimate(
        "完成辅导者的额外收益（CACE）", float(beta[-1]),
        float(beta[-1] - Z * se), float(beta[-1] + Z * se), len(rows),
        int(assigned.sum()), int((1 - assigned).sum()),
        "以随机邀约为工具变量；只识别实际完成者，未完成者不计入",
    )
    return estimate, first


def by_modality(records: Sequence[Record]) -> Dict[str, Estimate]:
    """按邀约方式对照未受邀约者的意向收益。

    邀约方式在邀约时指定，并没有按方式随机分配；这里只作描述，
    不解释为两种方式之间的因果优劣。
    """
    rows = [r for r in randomized(records)]
    if not rows:
        return {}
    output: Dict[str, Estimate] = {}
    for kind in KINDS:
        sub = [r for r in rows if r.offered_kind == kind or not r.assigned]
        treated = [r for r in sub if r.assigned and r.offered_kind == kind]
        control = [r for r in sub if not r.assigned]
        y = np.array([r.post_value - r.screened_mean for r in sub])
        indicator = np.array([1.0 if r.assigned and r.offered_kind == kind else 0.0 for r in sub])
        output[kind] = _estimate(
            f"{'低负担答疑' if kind == 'short' else '整段训练'}的邀约收益",
            y, indicator, _design(sub), len(treated), len(control),
            "方式在邀约时指定、并未随机分配；只估计该方式的邀约收益，"
            "不是方式间的因果对比",
        )
    return output


def completion_rates(records: Sequence[Record]) -> Dict[str, Dict[str, float]]:
    rows = [r for r in records if r.policy in ("P1", "P2") and r.assigned]
    output: Dict[str, Dict[str, float]] = {}
    for kind in KINDS:
        sub = [r for r in rows if r.offered_kind == kind]
        done = sum(1 for r in sub if r.completed)
        output[kind] = {
            "assigned": len(sub),
            "completed": done,
            "rate": done / len(sub) if sub else float("nan"),
        }
    return output


def balance_table(records: Sequence[Record]) -> List[Dict[str, object]]:
    """邀约组与对照组在轮次内是否可比；不平衡的层要单独说明。"""
    rows = [r for r in records if r.policy in ("P1", "P2")]
    output = []
    for policy in ("P1", "P2"):
        for prob in sorted({r.prob for r in rows if r.policy == policy}):
            sub = [r for r in rows if r.policy == policy and r.prob == prob]
            treated = [r for r in sub if r.assigned]
            control = [r for r in sub if not r.assigned]
            if not treated or not control:
                output.append(
                    {
                        "policy": policy,
                        "prob": prob,
                        "treated": len(treated),
                        "control": len(control),
                        "comparable": False,
                        "note": "该层只有邀约组或只有对照组，无法在层内比较",
                    }
                )
                continue
            a = np.array([r.screened_mean for r in treated])
            b = np.array([r.screened_mean for r in control])
            pooled = np.sqrt(
                a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
            ) if len(a) > 1 and len(b) > 1 else float("nan")
            diff = float(a.mean() - b.mean())
            t = float(diff / pooled) if pooled and pooled == pooled and pooled > 0 else None
            output.append(
                {
                    "policy": policy,
                    "prob": prob,
                    "treated": len(treated),
                    "control": len(control),
                    "treated_mean": float(a.mean()),
                    "control_mean": float(b.mean()),
                    "diff": diff,
                    "t": t,
                    "comparable": bool(t is None or abs(t) <= 2.5),
                    "note": ""
                    if t is None or abs(t) <= 2.5
                    else "该层基线不平衡，邀约可能按分数决定；只能作为调整后的意向估计",
                }
            )
    return output


def itt_without_imbalanced(records: Sequence[Record]) -> Estimate:
    """去掉基线不平衡的层，作为敏感性检查。"""
    rows = randomized(records)
    if not rows:
        return Estimate("稳健性检查", None, None, None, 0)
    imbalanced = {
        (row["policy"], row["prob"])
        for row in balance_table(rows)
        if not row["comparable"]
    }
    keep = [r for r in rows if (r.policy, r.prob) not in imbalanced]
    if len(keep) < 40:
        return Estimate(
            "去掉不平衡层后的邀约收益", None, None, None, len(keep),
            note="去掉不平衡层后样本过少，不能单独估计",
        )
    y = np.array([r.post_value - r.screened_mean for r in keep])
    assigned = np.array([1.0 if r.assigned else 0.0 for r in keep])
    return _estimate(
        "去掉不平衡层后的邀约收益", y, assigned, _design(keep),
        int(assigned.sum()), int((1 - assigned).sum()),
        "只保留轮次内邀约组与对照组基线可比的层",
    )


def usual_practice_risk(records: Sequence[Record]) -> Dict[str, object]:
    """当时通常做法下的后测风险：只用未受邀约者。"""
    pool = [r for r in records if not r.assigned and r.post_independent]
    if not pool:
        return {"identified": False, "reason": "没有未受邀约的成熟后测", "n": 0}
    events = sum(1 for r in pool if r.post_risk)
    return {
        "identified": True,
        "n": len(pool),
        "events": events,
        "risk": events / len(pool),
        "note": "未受邀约学员的常规练习结果，不含辅导带来的改善",
    }


def missing_table(
    records: Sequence[Record], due_by: Optional[datetime] = None
) -> List[Dict[str, object]]:
    """区分“已到期但缺失”和“尚未到期”；两者都不当作成功或失败。"""
    table = []
    for policy in ("P0", "P1", "P2"):
        for assigned in (False, True):
            sub = [r for r in records if r.policy == policy and r.assigned == assigned]
            if not sub:
                continue
            observed = [r for r in sub if r.post_independent]
            same_coach = [r for r in sub if r.post is not None and not r.post_independent]
            pending = [r for r in sub if r.post is None and due_by and r.post_due > due_by]
            missing = [
                r for r in sub if r.post is None and not (due_by and r.post_due > due_by)
            ]
            table.append(
                {
                    "policy": policy,
                    "assigned": assigned,
                    "n": len(sub),
                    "observed": len(observed),
                    "same_coach": len(same_coach),
                    "pending": len(pending),
                    "missing": len(missing),
                    "observed_gain": float(np.mean([r.gain for r in observed]))
                    if observed else None,
                }
            )
    return table


def naive_receipt(records: Sequence[Record]) -> Dict[str, object]:
    """实际领取与未领取的朴素比较，用来说明为什么不能当因果。"""
    rows = [r for r in randomized(records)]
    if not rows:
        return {"available": False}
    done = [r for r in rows if r.completed]
    rest = [r for r in rows if not r.completed]
    return {
        "available": True,
        "completed": len(done),
        "not_completed": len(rest),
        "completed_gain": float(np.mean([r.gain for r in done])),
        "not_completed_gain": float(np.mean([r.gain for r in rest]))
        if rest else None,
        "difference": float(np.mean([r.gain for r in done]) - np.mean([r.gain for r in rest]))
        if rest else None,
        "mean_screened_completed": float(np.mean([r.screened_mean for r in done])),
        "mean_screened_not": float(np.mean([r.screened_mean for r in rest])) if rest else None,
        "note": "领取者基线更低；该差值混合了选择效应，不能解释为辅导收益",
    }


def missing_bounds(
    records: Sequence[Record], due_by: Optional[datetime] = None
) -> Dict[str, object]:
    """极端情形下缺失会把收益区间推到哪里；缺失既不记成功也不记失败。"""
    rows = [
        r for r in records
        if r.policy in ("P1", "P2")
        and not (r.post is None and due_by and r.post_due > due_by)
    ]
    if not rows:
        return {"available": False}
    output = {}
    for label, group in (("assigned", [r for r in rows if r.assigned]),
                         ("control", [r for r in rows if not r.assigned])):
        observed = [r for r in group if r.post_independent]
        missing = len(group) - len(observed)
        if not observed:
            output[label] = {"observed": 0, "missing": missing, "identified": False}
            continue
        gain = np.mean([r.gain for r in observed])
        low = (sum(r.gain for r in observed) + missing * -100) / len(group)
        high = (sum(r.gain for r in observed) + missing * 100) / len(group)
        output[label] = {
            "n": len(group),
            "observed": len(observed),
            "missing": missing,
            "observed_gain": float(gain),
            "low": float(low),
            "high": float(high),
        }
    if "assigned" in output and "control" in output and "low" in output["assigned"] \
            and "low" in output["control"]:
        output["difference_low"] = output["assigned"]["low"] - output["control"]["high"]
        output["difference_high"] = output["assigned"]["high"] - output["control"]["low"]
    output["note"] = (
        "边界假设缺失后测落在 0 至 100 的极端；用于说明缺失规模，不能替代缺失机制"
    )
    return output


def same_coach(records: Sequence[Record]) -> Dict[str, object]:
    """同一教练复评的后测不当作独立观察。"""
    all_rows = list(records)
    same = [r for r in all_rows if r.post is not None and not r.post_independent]
    independent = [r for r in all_rows if r.post_independent]
    return {
        "same_coach_n": len(same),
        "independent_n": len(independent),
        "same_coach_students": [r.id for r in same],
        "note": "同一位教练既带教又复评，结果不当成独立观察；主分析只用独立后测。",
        "primary_analysis_uses": "independent",
    }


def never_eligible(records: Sequence[Record]) -> List[Dict[str, object]]:
    """从未被允许选择辅导的区间：这些区间的结果不能作方式比较。"""
    output = []
    for policy in ("P0", "P1", "P2"):
        rows = [r for r in records if r.policy == policy]
        if not rows:
            continue
        never = [r for r in rows if not r.assigned]
        offered_low = [r for r in rows if r.assigned and r.screened_min < 60]
        offered_high = [r for r in rows if r.assigned and r.screened_min >= 60]
        output.append(
            {
                "policy": policy,
                "n": len(rows),
                "never_assigned": len(never),
                "assigned_low": len(offered_low),
                "assigned_high": len(offered_high),
                "note": "P0 为旧策略：只对低于门槛者按规则安排，未做随机分配；"
                        "其未受邀约区间没有随机对照"
                if policy == "P0"
                else "邀约概率由教学组给定，不是接受概率；未受邀约区间无辅导，"
                     "只能作对照，不能当方式选择",
            }
        )
    return output


def coach_summary(records: Sequence[Record]) -> List[Dict[str, object]]:
    """各教练的带教量与结果；相同教练关联的结果不当成独立观察。"""
    rows = [r for r in records if r.policy in ("P1", "P2") and r.assigned]
    output = []
    for coach in sorted({r.coach for r in rows if r.coach}):
        sub = [r for r in rows if r.coach == coach]
        completed = [r for r in sub if r.completed]
        observed = [r for r in sub if r.post_independent]
        output.append(
            {
                "coach": coach,
                "assigned": len(sub),
                "completed": len(completed),
                "completion_rate": len(completed) / len(sub) if sub else None,
                "observed": len(observed),
                "mean_screened": float(np.mean([r.screened_mean for r in sub])) if sub else None,
                "mean_gain": float(np.mean([r.gain for r in observed])) if observed else None,
                "practices_both_kinds": len({r.offered_kind for r in sub}) > 1,
            }
        )
    return output


def improved_by_help(records: Sequence[Record], effect: Optional[float] = None) -> Dict[str, object]:
    """按估计的额外收益回推：有多少“后测达标”其实依赖辅导。

    这些学员若被标成“原本不需要”，下一届会漏掉他们。
    """
    rows = randomized(records)
    if not rows:
        return {"available": False, "reason": "没有成熟的随机邀约轮次"}
    if effect is None:
        effect = cace(records)[0].value
    if effect is None:
        return {"available": False, "reason": "没有可信对照，无法回推反事实"}
    treated = [r for r in rows if r.completed and r.post_independent]
    if not treated:
        return {"available": False, "reason": "没有完成辅导且已复测的学员"}
    helped = [
        r for r in treated
        if r.post_value >= 60 and r.post_value - effect < 60
    ]
    remain = [r for r in treated if r.post_value < 60]
    return {
        "available": True,
        "completed": len(treated),
        "post_pass": sum(1 for r in treated if r.post_value >= 60),
        "helped_over_line": len(helped),
        "still_below": len(remain),
        "effect_used": float(effect),
        "students": [r.id for r in helped],
        "note": "回推使用 CACE 点估计；若缺少可信对照就不能这样归类",
    }


def label_bias(records: Sequence[Record], wave: Optional[int] = None) -> List[Dict[str, object]]:
    """把辅导组后测混进标签会漏掉“得到帮助才进步”的学员。

    对每个成熟轮次，用此前轮次的对照训练、用当轮对照评价；
    两种标签做法并排：只用对照，或把受邀约者也纳入。
    """
    from .screening import fit_logistic, predict_proba, feature_matrix, auc

    mature = [r for r in records if r.post_independent and r.policy in ("P1", "P2")]
    output = []
    for target in sorted({r.wave for r in mature}):
        if wave is not None and target != wave:
            continue
        prior = [r for r in mature if r.wave < target]
        test = [r for r in mature if r.wave == target and not r.assigned]
        if not prior or len(test) < 10:
            continue
        for label, train in (("只用对照", [r for r in prior if not r.assigned]), ("全部已观察", prior)):
            if len(train) < 20 or len({r.post_risk for r in train}) < 2:
                continue
            x = feature_matrix(train)
            y = np.array([1.0 if r.post_risk else 0.0 for r in train])
            mu, sd = x.mean(axis=0), x.std(axis=0)
            sd[sd == 0] = 1.0
            beta = fit_logistic((x - mu) / sd, y)
            xt = (feature_matrix(test) - mu) / sd
            prob = predict_proba(beta, xt)
            truth = np.array([1.0 if r.post_risk else 0.0 for r in test])
            flagged = prob >= 0.5
            tp = int(np.sum(flagged & (truth == 1)))
            fp = int(np.sum(flagged & (truth == 0)))
            fn = int(np.sum(~flagged & (truth == 1)))
            output.append(
                {
                    "wave": target,
                    "label": label,
                    "train_n": len(train),
                    "train_events": int(y.sum()),
                    "test_n": len(test),
                    "test_events": int(truth.sum()),
                    "auc": auc(truth, prob),
                    "brier": float(np.mean((prob - truth) ** 2)),
                    "flagged": tp + fp,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "sensitivity": tp / (tp + fn) if tp + fn else None,
                    "ppv": tp / (tp + fp) if tp + fp else None,
                    "missed": fn,
                }
            )
    return output
