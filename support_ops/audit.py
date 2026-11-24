"""对照实际辅导与未成熟后测的轮次回看。

明确标出：哪些阶段的选择发生了变化、哪些证据不足的部位要继续保留可对照的安排。
"""

from __future__ import annotations

from typing import Dict

from .bundle import Bundle
from .cohort import build_cohort
from .effects import cace, intent_to_treat, by_modality
from .screening import availability_by_track, coverage_check, model_comparison

Z = 1.959963984540054


def _availability(records):
    return availability_by_track(records)


def wave_audit(bundle: Bundle, wave: int, as_of=None) -> Dict[str, object]:
    """回看某一轮：名单对照、实际辅导、后测成熟度与阶段变化。"""
    records = build_cohort(bundle, at=as_of)
    target = [r for r in records if r.wave == wave]
    if not target:
        return {"wave": wave, "available": False, "reason": "该轮次不在资料中"}

    rows = []
    for record in sorted(target, key=lambda r: r.id):
        rows.append(
            {
                "student": record.id,
                "track": record.track,
                "screened_min": record.screened_min,
                "screened_mean": record.screened_mean,
                "assigned": record.assigned,
                "offered_kind": record.offered_kind,
                "completed": record.completed,
                "coach": record.coach,
                "post": record.post_value,
                "post_independent": record.post_independent,
                "post_mature": record.post is not None,
                "gain": record.gain,
                "recheck_after_decision": record.corrected_after_decision(),
                "recheck_detail": [
                    {
                        "skill": row.skill,
                        "value": row.value,
                        "received": row.received.isoformat(),
                        "supersedes": row.supersedes,
                    }
                    for row in record.verified
                ],
            }
        )

    assigned = [r for r in target if r.assigned]
    completed = [r for r in assigned if r.completed]
    immature = [r for r in target if r.post is None]
    same_coach = [r for r in target if r.post is not None and not r.post_independent]
    stage_changes = [
        {
            "student": r.id,
            "skill": row.skill,
            "value": row.value,
            "received": row.received.isoformat(),
            "supersedes": row.supersedes,
            "after_invitation": True,
        }
        for r in target
        for row in r.verified
        if row.received > r.invitation_at
    ]

    return {
        "wave": wave,
        "available": True,
        "policy": target[0].policy,
        "n": len(target),
        "screened_risk": sum(1 for r in target if r.screened_min < 60),
        "assigned": len(assigned),
        "completed": len(completed),
        "completed_share": len(completed) / len(assigned) if assigned else None,
        "post_mature": sum(1 for r in target if r.post is not None),
        "post_independent": sum(1 for r in target if r.post_independent),
        "immature": [r.id for r in immature],
        "same_coach": [r.id for r in same_coach],
        "stage_changes": stage_changes,
        "students": rows,
    }


def carry_forward(bundle: Bundle) -> Dict[str, object]:
    """下一轮能带走的证据状态：哪些轮次成熟、哪些部位仍缺对照。"""
    records = build_cohort(bundle)
    by_wave = {}
    for wave in sorted({r.wave for r in records}):
        rows = [r for r in records if r.wave == wave]
        by_wave[wave] = {
            "n": len(rows),
            "mature": sum(1 for r in rows if r.post_independent),
            "immature": sum(1 for r in rows if r.post is None),
            "same_coach": sum(1 for r in rows if r.post is not None and not r.post_independent),
            "assigned": sum(1 for r in rows if r.assigned),
            "completed": sum(1 for r in rows if r.completed),
        }
    mature = [r for r in records if r.post_independent and r.policy in ("P1", "P2")]
    gaps = []
    for track in ("surgery", "general", "community"):
        for band, (low, high) in (("低于 60", (0, 60)), ("60 及以上", (60, 101))):
            sub = [r for r in mature if r.track == track and low <= r.screened_mean < high]
            treated = sum(1 for r in sub if r.assigned)
            control = sum(1 for r in sub if not r.assigned)
            covered = treated >= 10 and control >= 10
            gaps.append(
                {
                    "track": track,
                    "band": band,
                    "n": len(sub),
                    "assigned": treated,
                    "control": control,
                    "covered": covered,
                    "note": "" if covered
                    else f"对照组只有 {control} 人（需至少 10 人），"
                         "下一轮应保留未受邀约的可比安排，收益优先级暂不可给",
                }
            )
    return {
        "waves": by_wave,
        "evidence_gaps": gaps,
        "itt": intent_to_treat(records).as_dict(),
        "cace": cace(records)[0].as_dict(),
        "modalities": {k: v.as_dict() for k, v in by_modality(records).items()},
        "coverage": coverage_check(records),
        "availability": _availability(records),
        "screening_models": model_comparison(records),
        "unresolved_controls": [
            r.id for r in records if r.policy == "P2" and r.post is None
        ],
    }
