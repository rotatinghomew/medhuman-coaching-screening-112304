"""按可见时点组织的学员记录。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

from .bundle import Appointment, Bundle, FirstScore, Invitation, PostScore, Student, slot_start

MATURE = "mature"
IMMATURE = "immature"
LOW_BURDEN = "short"
FULL_BLOCK = "full"
RECHECK_STAGE = "recheck"


def mean(values) -> float:
    values = list(values.values()) if isinstance(values, dict) else list(values)
    if not values:
        raise ValueError("没有可平均的数值")
    return sum(values) / len(values)


@dataclass
class Record:
    """一名学员在某一轮次决策时点上的可见记录。

    ``screened`` 是决策时点可见的最终版初评（复核已生效）；
    ``initial`` 是首次评分，只用于描述当时通常做法下的观察，不进入筛查。
    """

    student: Student
    invitation: Invitation
    decision_at: datetime
    screened: Dict[str, float]
    initial: Dict[str, float]
    verified: List[FirstScore]
    post: Optional[PostScore]
    appointments: List[Appointment]
    post_due: datetime
    review_due: datetime = datetime.min

    @property
    def id(self) -> str:
        return self.student.id

    @property
    def wave(self) -> int:
        return self.student.wave

    @property
    def track(self) -> str:
        return self.student.track

    @property
    def policy(self) -> str:
        return self.invitation.policy

    @property
    def rotation_end(self) -> datetime:
        return self.student.rotation_end

    @property
    def assigned(self) -> bool:
        return self.invitation.assigned

    @property
    def completed(self) -> bool:
        return self.invitation.completed

    @property
    def offered_kind(self) -> str:
        return self.invitation.offered_kind

    @property
    def prob(self) -> float:
        return self.invitation.prob

    @property
    def coach(self) -> Optional[str]:
        return self.invitation.coach

    @property
    def has_post(self) -> bool:
        return self.post is not None

    @property
    def post_independent(self) -> bool:
        return self.post is not None and self.post.status == "independent"

    @property
    def post_value(self) -> Optional[float]:
        return None if self.post is None else self.post.value

    @property
    def screened_mean(self) -> float:
        return mean(self.screened)

    @property
    def screened_min(self) -> float:
        return min(self.screened.values())

    @property
    def initial_mean(self) -> float:
        return mean(self.initial)

    @property
    def gain(self) -> Optional[float]:
        """后测减决策时点可见的初评均分。"""
        if self.post is None:
            return None
        return self.post.value - self.screened_mean

    @property
    def post_risk(self) -> Optional[bool]:
        """后测低于 60 即需继续支持。"""
        return None if self.post is None else self.post.value < 60

    @property
    def invitation_at(self) -> datetime:
        return self.invitation.sent

    @property
    def stage(self) -> str:
        """本阶段是否已吸收复核改判。"""
        return RECHECK_STAGE if self.recheck_rows else "initial"

    @property
    def recheck_rows(self) -> List[FirstScore]:
        """邀约之后、本阶段筛查时点之前生效的复核记录。"""
        return [
            row
            for row in self.verified
            if self.invitation_at < row.received <= self.decision_at
        ]

    def corrected_after_decision(self) -> bool:
        """本阶段筛查时点之后还会出现复核改判（不可见）。"""
        return any(row.received > self.decision_at for row in self.verified)


def build_cohort(bundle: Bundle, at: Optional[datetime] = None) -> List[Record]:
    """构造队列。

    ``at`` 是做出选择的时点：初评与复核只取该时点前可见的记录。
    省略 ``at`` 时按各轮次自己的邀约发出时刻截取，避免把之后的复核带回决策。
    """
    records: List[Record] = []
    for student_id, student in bundle.students.items():
        invitation = bundle.invitations[student_id]
        screen_at = at if at is not None else invitation.sent
        verified = [
            row for row in bundle.score_events(student_id) if row.status == "verified"
        ]
        records.append(
            Record(
                student=student,
                invitation=invitation,
                decision_at=screen_at,
                screened=bundle.scores_at(student_id, screen_at),
                initial=bundle.initial_scores(student_id),
                verified=verified,
                post=bundle.post_at(student_id, at),
                appointments=bundle.confirmed_appointments(student_id, at),
                post_due=bundle.post_due(student_id),
                review_due=bundle.review_due(student_id),
            )
        )
    records.sort(key=lambda record: (record.wave, record.id))
    return records


def select(records: Iterable[Record], wave: Optional[int] = None,
           waves: Optional[Sequence[int]] = None, policy: Optional[str] = None,
           mature_only: bool = False) -> List[Record]:
    out = list(records)
    if wave is not None:
        out = [r for r in out if r.wave == wave]
    if waves is not None:
        out = [r for r in out if r.wave in set(waves)]
    if policy is not None:
        out = [r for r in out if r.policy == policy]
    if mature_only:
        out = [r for r in out if r.post_independent]
    return out


def controls(records: Iterable[Record]) -> List[Record]:
    """未收到辅导邀约的学员：描述“当时通常做法”下的风险。"""
    return [r for r in records if not r.assigned]


def full_block_pairs(
    bundle: Bundle, student_id: str, coach: Optional[str] = None,
    exclude_slots: Sequence[str] = (), as_of: Optional[datetime] = None,
) -> List[tuple]:
    """列出转科前能完成的整段训练两次场次组合。

    两次须同一教练、相隔 1 至 3 日、都在转科截止前，且不能复用同一场次。
    """
    student = bundle.students[student_id]
    pool = [
        slot
        for slot in bundle.slots.values()
        if slot.kind == FULL_BLOCK
        and slot.id not in set(exclude_slots)
        and slot_start(slot) < student.rotation_end
        and (as_of is None or slot_start(slot) > as_of)
        and (coach is None or slot.coach == coach)
    ]
    pairs = []
    for index, first in enumerate(pool):
        for second in pool[index + 1:]:
            if first.coach != second.coach:
                continue
            gap = abs((slot_start(second).date() - slot_start(first).date()).days)
            if 1 <= gap <= 3:
                pairs.append((first, second))
    return pairs


def feasibility(bundle: Bundle, student_id: str, as_of: datetime) -> Dict[str, object]:
    """转科前能完成的连续练习次数与整段训练可行性。"""
    student = bundle.students[student_id]
    upcoming = [slot for slot in bundle.slots.values() if as_of < slot_start(slot) < student.rotation_end]
    days = sorted({slot.day for slot in upcoming})
    full_days = sorted({slot.day for slot in upcoming if slot.kind == FULL_BLOCK})
    return {
        "student": student_id,
        "rotation_end": student.rotation_end.isoformat(),
        "practice_days_before_rotation": len(days),
        "full_days_before_rotation": len(full_days),
        "days": days,
        "can_complete_full_block": bool(full_block_pairs(bundle, student_id, as_of=as_of)),
    }


def confirmed_commitments(
    bundle: Bundle, student_id: str, at: Optional[datetime] = None
) -> tuple:
    """把已确认预约整理成一段整段训练加若干答疑，同一段承诺只算一次。"""
    items = [
        bundle.slots[item.slot]
        for item in bundle.confirmed_appointments(student_id, at)
        if item.slot in bundle.slots
    ]
    items.sort(key=slot_start)
    full = [slot for slot in items if slot.kind == FULL_BLOCK]
    block = None
    for index in range(len(full) - 1):
        first, second = full[index], full[index + 1]
        gap = abs((slot_start(second).date() - slot_start(first).date()).days)
        if first.coach == second.coach and 1 <= gap <= 3 and first.id != second.id:
            block = (first, second)
            break
    shorts = tuple(slot for slot in items if slot.kind == LOW_BURDEN)
    return block, shorts
