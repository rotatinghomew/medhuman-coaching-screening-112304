"""按筛查、最低保障、转科前可完成次数与导师分项资质排班。

口径与资料约定一致：

* ``short``：每次最多一席每人，按分项答疑；不同场次是多次承诺。
* ``full``：每人两次、间隔 1 至 3 日、同一导师主持、都在当前轮转截止前；
  两次合起来算一段承诺，不能拆成两段。
* 已确认场次继续占用；``requested`` 只是申请，不占座，转为确认需教练与学员明确同意。
* 已确认的答疑与整段训练可并行，但彼此不能替代；同一次课不能同时充当两段承诺。

每条安排都写明依据：是来自有可靠收益证据的档位，还是只能遵循最低保障规则。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .bundle import Bundle, Slot, slot_start
from .cohort import FULL_BLOCK, LOW_BURDEN, Record, build_cohort, confirmed_commitments

BAND_LOW = "低于 60"
BAND_HIGH = "60 及以上"


@dataclass
class Need:
    record: Record
    deficits: Tuple[str, ...]
    confirmed_block: Optional[Tuple[Slot, Slot]]
    confirmed_short: Tuple[Slot, ...]
    stage: str
    practice_days: int
    full_need_score: float
    recheck_applied: bool = False

    @property
    def student(self) -> str:
        return self.record.id

    @property
    def confirmed_skills(self) -> Set[str]:
        covered: Set[str] = set()
        if self.confirmed_block is not None:
            covered.update(self.confirmed_block[0].skills)
            covered.update(self.confirmed_block[1].skills)
        for slot in self.confirmed_short:
            covered.update(slot.skills)
        return covered & set(self.deficits)

    @property
    def residual(self) -> Tuple[str, ...]:
        return tuple(skill for skill in self.deficits if skill not in self.confirmed_skills)

    @property
    def has_confirmed(self) -> bool:
        return self.confirmed_block is not None or bool(self.confirmed_short)

    @property
    def band(self) -> str:
        return BAND_LOW if self.record.screened_mean < 60 else BAND_HIGH


@dataclass
class Placement:
    student: str
    stage: str
    tier: str
    sessions: List[str]
    covered: Tuple[str, ...]
    uncovered: Tuple[str, ...]
    reason: str
    basis: str
    practice_days: int
    confirmed: bool = False
    pending: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "student": self.student,
            "stage": self.stage,
            "tier": self.tier,
            "sessions": list(self.sessions),
            "covered": list(self.covered),
            "uncovered": list(self.uncovered),
            "reason": self.reason,
            "basis": self.basis,
            "practice_days": self.practice_days,
            "confirmed": self.confirmed,
            "pending": list(self.pending),
        }


@dataclass
class Plan:
    scenario: str
    stage: str
    placements: List[Placement]
    unserved: List[Dict[str, object]]
    seats_left: Dict[str, int]
    minimum_short: int
    minimum_short_fraction: float
    notes: List[str]

    def summary(self) -> Dict[str, object]:
        served = [p for p in self.placements if p.sessions]
        short = sum(1 for p in served if LOW_BURDEN in p.tier)
        return {
            "scenario": self.scenario,
            "stage": self.stage,
            "screened_risk": len(self.placements),
            "served": len(served),
            "full_tier": sum(1 for p in served if FULL_BLOCK in p.tier),
            "short_tier": short,
            "minimum_short": self.minimum_short,
            "minimum_short_met": short >= self.minimum_short,
            "fully_covered": sum(1 for p in served if not p.uncovered),
            "unserved": len(self.unserved),
            "by_rule_only": sum(1 for p in served if p.basis == "guarantee"),
            "by_evidence": sum(1 for p in served if p.basis == "evidence"),
            "seats_left": dict(self.seats_left),
        }


def deficits(record: Record, threshold: float = 60.0) -> Tuple[str, ...]:
    return tuple(sorted(skill for skill, value in record.screened.items() if value < threshold))


def practice_days(bundle: Bundle, student_id: str, as_of: datetime) -> int:
    """转科前还能安排的练习日数（有场次的日期数）。"""
    student = bundle.students[student_id]
    return len(
        {
            slot.day
            for slot in bundle.slots.values()
            if as_of < slot_start(slot) < student.rotation_end
        }
    )


def build_need(
    bundle: Bundle, record: Record, stage: str, threshold: float = 60.0,
    at: Optional[datetime] = None,
) -> Need:
    block, shorts = confirmed_commitments(bundle, record.id, at)
    policy = bundle.policies[record.student.policy]
    return Need(
        record=record,
        deficits=deficits(record, threshold),
        confirmed_block=block,
        confirmed_short=shorts,
        stage=stage,
        practice_days=practice_days(bundle, record.id, at or record.decision_at),
        full_need_score=policy.full_need_score,
        recheck_applied=bool(record.recheck_rows),
    )


def _full_pairs(
    bundle: Bundle, need: Need, capacity: Dict[str, int], used_slots: Dict[str, Set[str]],
    used_days: Dict[str, Set[str]], as_of: datetime, priority_skills: Sequence[str],
) -> List[Tuple[Slot, Slot, Set[str]]]:
    student = bundle.students[need.student]
    pool = [
        slot
        for slot in bundle.slots.values()
        if slot.kind == FULL_BLOCK and as_of < slot_start(slot) < student.rotation_end
    ]
    options = []
    for index, first in enumerate(pool):
        for second in pool[index + 1:]:
            if first.coach != second.coach:
                continue
            gap = abs((slot_start(second).date() - slot_start(first).date()).days)
            if not 1 <= gap <= 3:
                continue
            if capacity.get(first.id, 0) <= 0 or capacity.get(second.id, 0) <= 0:
                continue
            taken = used_slots.get(need.student, set())
            if first.id in taken or second.id in taken:
                continue
            busy_days = used_days.get(need.student, set())
            if first.day in busy_days or second.day in busy_days:
                continue
            covered = (set(first.skills) | set(second.skills)) & set(priority_skills)
            options.append((first, second, covered))
    options.sort(key=lambda item: (-len(item[2]), slot_start(item[0]), item[0].id, item[1].id))
    return options


def _short_options(
    bundle: Bundle, student_id: str, skill: str, capacity: Dict[str, int],
    used_slots: Dict[str, Set[str]], used_days: Dict[str, Set[str]], as_of: datetime,
) -> List[Slot]:
    """低负担答疑：每位学员每日最多一次，且只用于解释分项。"""
    student = bundle.students[student_id]
    taken = used_slots.get(student_id, set())
    busy_days = used_days.get(student_id, set())
    options = [
        slot
        for slot in bundle.slots.values()
        if slot.kind == LOW_BURDEN
        and skill in slot.skills
        and as_of < slot_start(slot) < student.rotation_end
        and capacity.get(slot.id, 0) > 0
        and slot.id not in taken
        and slot.day not in busy_days
    ]
    options.sort(key=lambda slot: (slot.day, slot.begin, slot.id))
    return options


def _take(
    capacity: Dict[str, int], slot: Slot, used_slots: Dict[str, Set[str]],
    used_days: Dict[str, Set[str]], student: str,
) -> None:
    capacity[slot.id] -= 1
    used_slots.setdefault(student, set()).add(slot.id)
    used_days.setdefault(student, set()).add(slot.day)


def _seats_left(bundle: Bundle, capacity: Dict[str, int]) -> Dict[str, int]:
    return {
        "full": sum(value for sid, value in capacity.items() if bundle.slots[sid].kind == FULL_BLOCK),
        "short": sum(value for sid, value in capacity.items() if bundle.slots[sid].kind == LOW_BURDEN),
    }


def _tier(bundle: Bundle, sessions: Sequence[str]) -> str:
    kinds = {bundle.slots[sid].kind for sid in sessions}
    if kinds == {FULL_BLOCK, LOW_BURDEN}:
        return "full+short"
    if kinds == {FULL_BLOCK}:
        return FULL_BLOCK
    return LOW_BURDEN


def _basis(need: Need, sessions: Sequence[str], evidence: Dict[Tuple[str, str], bool]) -> str:
    """依据可靠收益证据，还是只能遵循保障规则；与是否已确认无关。"""
    if not sessions:
        return "none"
    return "evidence" if evidence.get((need.record.track, need.band), False) else "guarantee"


def _finish(
    bundle: Bundle, need: Need, sessions: List[str], covered: Set[str],
    uncovered: Sequence[str], evidence: Dict[Tuple[str, str], bool],
) -> Placement:
    basis = _basis(need, sessions, evidence)
    parts: List[str] = []
    if need.has_confirmed:
        parts.append("保留已确认场次，与新增安排分开计数")
    if need.recheck_applied:
        parts.append("本阶段已吸收复核改判；改判不挤掉已确认场次")
    if not sessions:
        parts.append("转科前排不出可执行场次，列入等待名单")
    elif covered and len(covered) >= len(need.deficits):
        parts.append("导师分项资质覆盖全部分项")
    elif covered:
        parts.append("导师分项资质只覆盖部分分项，其余按保障规则补足")
    if basis == "guarantee":
        parts.append("该轮转与练习能力段没有可靠收益证据，只能遵循保障规则")
    return Placement(
        student=need.student,
        stage=need.stage,
        tier=_tier(bundle, sessions) if sessions else "none",
        sessions=sessions,
        covered=tuple(sorted(covered)),
        uncovered=tuple(sorted(uncovered)),
        reason="；".join(parts),
        basis=basis,
        practice_days=need.practice_days,
        confirmed=need.has_confirmed,
    )


def _schedule(
    bundle: Bundle, needs: Sequence[Need], capacity: Dict[str, int],
    used_slots: Dict[str, Set[str]], used_days: Dict[str, Set[str]], as_of: datetime,
    evidence: Dict[Tuple[str, str], bool], guarantee_first: bool, full_min_deficits: int,
) -> Tuple[List[Placement], List[Dict[str, object]]]:
    placements: List[Placement] = []
    unserved: List[Dict[str, object]] = []

    reserved: Dict[str, List[str]] = {}
    if guarantee_first:
        for need in needs:
            if need.has_confirmed:
                continue
            chosen = []
            for skill in need.deficits:
                options = _short_options(
                    bundle, need.student, skill, capacity, used_slots, used_days, as_of
                )
                if options:
                    slot = options[0]
                    _take(capacity, slot, used_slots, used_days, need.student)
                    chosen = [slot.id]
                    break
            reserved[need.student] = chosen

    for need in needs:
        sessions: List[str] = []
        covered: Set[str] = set(need.confirmed_skills)
        if need.confirmed_block is not None:
            sessions = [need.confirmed_block[0].id, need.confirmed_block[1].id]
            sessions += [slot.id for slot in need.confirmed_short]
        else:
            sessions = [slot.id for slot in need.confirmed_short]
        sessions += reserved.get(need.student, [])
        for sid in sessions:
            covered.update(set(bundle.slots[sid].skills) & set(need.deficits))

        residual = [skill for skill in need.deficits if skill not in covered]
        # 政策声明的整段训练门槛与“至少缺两项”同时满足才排整段训练
        can_full = (
            need.confirmed_block is None
            and len(residual) >= full_min_deficits
            and need.record.screened_min < need.full_need_score
        )
        if can_full:
            options = _full_pairs(
                bundle, need, capacity, used_slots, used_days, as_of, residual
            )
            if options:
                first, second, block_covered = options[0]
                _take(capacity, first, used_slots, used_days, need.student)
                _take(capacity, second, used_slots, used_days, need.student)
                sessions.extend([first.id, second.id])
                covered.update(block_covered)

        residual = [skill for skill in need.deficits if skill not in covered]
        uncovered: List[str] = []
        for skill in residual:
            options = _short_options(
                bundle, need.student, skill, capacity, used_slots, used_days, as_of
            )
            if not options:
                uncovered.append(skill)
                continue
            slot = options[0]
            _take(capacity, slot, used_slots, used_days, need.student)
            sessions.append(slot.id)
            covered.add(skill)

        placements.append(_finish(bundle, need, sessions, covered, uncovered, evidence))
        if not sessions:
            unserved.append(
                {
                    "student": need.student,
                    "deficits": list(need.deficits),
                    "practice_days": need.practice_days,
                    "reason": "转科前没有可用场次；保留在等待名单，不记为已辅导",
                }
            )

    return placements, unserved


def build_plan(
    bundle: Bundle,
    records: Sequence[Record],
    wave: int,
    scenario: str = "quota",
    as_of: Optional[datetime] = None,
    threshold: float = 60.0,
    probabilities: Optional[Dict[str, float]] = None,
    evidence: Optional[Dict[Tuple[str, str], bool]] = None,
    full_min_deficits: int = 2,
) -> Plan:
    """为某一轮次排班。

    ``as_of`` 之后收到的评分与预约不可见；复查改判由此进入下一阶段。
    ``scenario='quota'``：全体名额，优先整段训练。
    ``scenario='guarantee'``：最低保障，先给每位筛查高风险学员一次低负担答疑，
    并满足本轮政策规定的最低答疑份额。
    """
    if scenario not in ("quota", "guarantee"):
        raise ValueError("scenario 只能是 quota 或 guarantee")
    wave_decisions = [r.decision_at for r in records if r.wave == wave]
    if not wave_decisions:
        raise ValueError(f"轮次 {wave} 不在资料中")
    if as_of is None:
        as_of = max(wave_decisions)
    snapshot = build_cohort(bundle, at=as_of)
    all_wave_records = [r for r in snapshot if r.wave == wave]
    wave_records = [r for r in all_wave_records if min(r.screened.values()) < threshold]
    stage = "recheck" if any(r.recheck_rows for r in wave_records) else "initial"
    policy = bundle.policies.get(
        bundle.students[wave_records[0].id].policy if wave_records else "P2"
    )
    fraction = policy.minimum_short_fraction if policy else 0.0

    def priority(record: Record):
        # 先看转科前还能安排几天，再看筛查风险与邀约优先级：
        # 练习日最少的学员先占位，避免他们把有限场次让给时间宽裕的人。
        probability = (probabilities or {}).get(record.id, 0.5)
        days = practice_days(bundle, record.id, as_of)
        return (days, -probability, min(record.screened.values()), record.rotation_end, record.id)

    needs = [
        build_need(bundle, r, stage, threshold, as_of)
        for r in sorted(wave_records, key=priority)
    ]
    capacity = {slot.id: slot.capacity for slot in bundle.slots.values()}
    used_slots: Dict[str, Set[str]] = {}
    used_days: Dict[str, Set[str]] = {}
    for need in needs:
        committed = list(need.confirmed_short)
        if need.confirmed_block is not None:
            committed += list(need.confirmed_block)
        for slot in committed:
            capacity[slot.id] -= 1
            used_slots.setdefault(need.student, set()).add(slot.id)
            used_days.setdefault(need.student, set()).add(slot.day)

    minimum_short = (
        math.ceil(fraction * len(needs)) if scenario == "guarantee" else 0
    )
    placements, unserved = _schedule(
        bundle, needs, capacity, used_slots, used_days, as_of, evidence or {},
        guarantee_first=(scenario == "guarantee"),
        full_min_deficits=full_min_deficits,
    )
    served = sum(1 for p in placements if p.sessions)
    minimum_short = min(minimum_short, served)

    notes = [
        "已确认的答疑与整段训练并行保留，互不替代；整段训练的两次不拆分计数。",
        "筛查只使用决策时点可见的最终版初评；复核改判只追加下一阶段安排，不挤掉已确认场次。",
        "实际领取不等于被分配；未完成或未复测不记为成功。",
    ]
    if scenario == "guarantee":
        notes.append(
            f"最低保障：{served} 人中获得低负担答疑至少 {minimum_short} 人"
            f"（政策最低份额 {fraction:.2f}）。"
        )
    for record in sorted(all_wave_records, key=lambda r: r.id):
        for item in bundle.requested_appointments(record.id):
            slot = bundle.slots.get(item.slot)
            if slot is None or slot_start(slot) <= as_of:
                continue
            notes.append(
                f"{record.id} 的申请 {item.slot}（{slot.kind}，{slot.day}）尚未确认，不占座；"
                "转为确认需教练与学员明确同意"
            )
    return Plan(
        scenario=scenario,
        stage=stage,
        placements=placements,
        unserved=unserved,
        seats_left=_seats_left(bundle, capacity),
        minimum_short=minimum_short,
        minimum_short_fraction=fraction,
        notes=notes,
    )


def two_stage_plan(
    bundle: Bundle,
    records: Sequence[Record],
    wave: int,
    initial_at: datetime,
    recheck_at: datetime,
    scenario: str = "quota",
    threshold: float = 60.0,
    probabilities: Optional[Dict[str, float]] = None,
    evidence: Optional[Dict[Tuple[str, str], bool]] = None,
) -> Dict[str, object]:
    """复核改判前后的两个阶段。

    第一阶段只看见 ``initial_at`` 前的记录；第二阶段看见 ``recheck_at`` 前的复核。
    已确认场次在两个阶段都保留、只占一次座位；同一次课不会被当成两段承诺，
    复核改判也不能挤掉第一阶段已确认的安排。
    """
    before = build_plan(
        bundle, records, wave, scenario, as_of=initial_at, threshold=threshold,
        probabilities=probabilities, evidence=evidence,
    )
    after = build_plan(
        bundle, records, wave, scenario, as_of=recheck_at, threshold=threshold,
        probabilities=probabilities, evidence=evidence,
    )
    before_by_student = {p.student: p for p in before.placements}
    after_by_student = {p.student: p for p in after.placements}
    changed = []
    for student_id in sorted(set(before_by_student) | set(after_by_student)):
        old = before_by_student.get(student_id)
        new = after_by_student.get(student_id)
        old_sessions = list(old.sessions) if old else []
        new_sessions = list(new.sessions) if new else []
        sessions_changed = old_sessions != new_sessions
        if not sessions_changed and not (old and new and old.stage != new.stage):
            continue
        # 只有“第一阶段已确认的场次”被真正改动时才需要重新确认
        confirmed_session_touched = bool(
            old
            and old.confirmed
            and (
                not new
                or not new.confirmed
                or set(old_sessions) != set(new_sessions)
            )
        )
        # 两份确认书在阶段之间才可见：第二阶段新增的已确认场次不算挤占
        confirmed_newly_visible = bool(
            new and new.confirmed and not (old and old.confirmed)
        )
        changed.append(
            {
                "student": student_id,
                "before": old_sessions,
                "after": new_sessions,
                "stage_before": old.stage if old else None,
                "stage_after": new.stage if new else None,
                "confirmed_kept": bool(new and new.confirmed),
                "lost_confirmed": bool(old and old.confirmed and not (new and new.confirmed)),
                "dropped_unconfirmed": sorted(set(old_sessions) - set(new_sessions)),
                "same_slot_reused": sorted(set(old_sessions) & set(new_sessions)),
                "sessions_changed": sessions_changed,
                "confirmed_newly_visible": confirmed_newly_visible,
                "needs_explicit_confirmation": confirmed_session_touched,
            }
        )
    squeezed = [
        item for item in changed
        if item["before"] and not item["after"]
    ]
    return {
        "before": before,
        "after": after,
        "stage_changed": after.stage != before.stage,
        "changed_students": changed,
        "squeezed_students": squeezed,
        "confirmed_retained": [
            {"student": p.student, "sessions": p.sessions}
            for p in after.placements
            if p.confirmed
        ],
        "confirmation_needed": [
            item["student"] for item in changed if item["needs_explicit_confirmation"]
        ],
        "notes": [
            "已确认场次在两个阶段都保留，只占一次座位；同一段整段训练不拆成两段承诺。",
            "复核改判只追加或调整未确认名额；涉及已确认场次的调整须教练与学员明确确认。",
            "第二阶段重排时，第一阶段未确认的提案可能被换到别的场次；"
            "变化名单逐人列出，供协调员核对。",
        ],
    }


def compare_scenarios(
    bundle: Bundle,
    records: Sequence[Record],
    wave: int,
    as_of: Optional[datetime] = None,
    probabilities: Optional[Dict[str, float]] = None,
    evidence: Optional[Dict[Tuple[str, str], bool]] = None,
) -> Dict[str, object]:
    """协调员比较全体名额与最低覆盖。"""
    quota = build_plan(bundle, records, wave, "quota", as_of, probabilities=probabilities,
                       evidence=evidence)
    guarantee = build_plan(bundle, records, wave, "guarantee", as_of, probabilities=probabilities,
                           evidence=evidence)
    return {
        "quota": {"summary": quota.summary(), "plan": quota},
        "guarantee": {"summary": guarantee.summary(), "plan": guarantee},
        "notes": [
            "全体名额方案优先整段训练，覆盖更多分项；最低保障方案先保证每人至少一次答疑。",
            "两者都保留已确认场次；已确认场次不计入新增承诺。",
        ],
    }
