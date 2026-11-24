"""资料交换包的读取与字段校验。

资料均为合成教学资料；不含真实学生、患者或诊疗记录。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SKILLS = ("listen", "explain", "respond")
TZ = timezone(timedelta(hours=8))


def parse_time(value: str) -> datetime:
    """把 ISO 时间读成带时区的 datetime。"""
    return datetime.fromisoformat(value.strip())


@dataclass(frozen=True)
class Student:
    id: str
    wave: int
    track: str
    rotation_end: datetime
    policy: str


@dataclass(frozen=True)
class FirstScore:
    id: str
    student: str
    skill: str
    value: float
    received: datetime
    status: str
    supersedes: str


@dataclass(frozen=True)
class Invitation:
    student: str
    policy: str
    assigned: bool
    offered_kind: str
    prob: float
    completed: bool
    coach: Optional[str]
    sent: datetime


@dataclass(frozen=True)
class PostScore:
    student: str
    book: str
    observed: datetime
    received: datetime
    value: float
    status: str

    @property
    def independent(self) -> bool:
        return self.status == "independent"


@dataclass(frozen=True)
class Policy:
    id: str
    tracks: Tuple[str, ...]
    kind: str
    review_at_day: int
    minimum_short_fraction: float
    full_need_score: float


@dataclass(frozen=True)
class Slot:
    id: str
    coach: str
    day: str
    begin: int
    capacity: int
    kind: str
    skills: Tuple[str, ...]


@dataclass(frozen=True)
class Appointment:
    id: str
    student: str
    slot: str
    state: str
    received: datetime

    @property
    def confirmed(self) -> bool:
        return self.state == "confirmed"


def slot_start(slot: Slot) -> datetime:
    """场次开始时刻；场次日期按资料约定使用 +08:00。"""
    date = datetime.fromisoformat(slot.day)
    hour, minute = divmod(slot.begin, 60)
    return date.replace(hour=hour, minute=minute, tzinfo=TZ)


def _read_table(root: Path, filename: str, fields: Sequence[str]) -> List[Dict[str, str]]:
    with (root / filename).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if list(reader.fieldnames or []) != list(fields):
            raise ValueError(f"{filename} 的表头与资料清单不符")
        rows = []
        for line, row in enumerate(reader, start=2):
            if len(row) != len(fields) or any(row[field] is None for field in fields):
                raise ValueError(f"{filename} 第 {line} 行的字段数量不符")
            rows.append(row)
        return rows


class Bundle:
    """一个可按时点回放的教学支持交换包。"""

    FIELDS = {
        "student.csv": ("id", "wave", "track", "rotation_end", "policy"),
        "first_score.csv": ("id", "student", "skill", "value", "received", "status", "supersedes"),
        "invitation.csv": (
            "student",
            "policy",
            "assigned",
            "offered_kind",
            "prob",
            "completed",
            "coach",
            "sent",
        ),
        "post_score.csv": ("student", "book", "observed", "received", "value", "status"),
        "policy.csv": (
            "id",
            "tracks",
            "kind",
            "review_at_day",
            "minimum_short_fraction",
            "full_need_score",
        ),
        "slot.csv": ("id", "coach", "day", "begin", "capacity", "kind", "skills"),
        "appointment.csv": ("id", "student", "slot", "state", "received"),
    }

    def __init__(
        self,
        students: Dict[str, Student],
        first_scores: List[FirstScore],
        invitations: Dict[str, Invitation],
        post_scores: Dict[str, PostScore],
        policies: Dict[str, Policy],
        slots: Dict[str, Slot],
        appointments: List[Appointment],
        root: Optional[Path] = None,
    ) -> None:
        self.students = students
        self.first_scores = first_scores
        self.invitations = invitations
        self.post_scores = post_scores
        self.policies = policies
        self.slots = slots
        self.appointments = appointments
        self.root = root
        self._first_by_student: Dict[str, List[FirstScore]] = {}
        for row in first_scores:
            self._first_by_student.setdefault(row.student, []).append(row)

    # ---------------------------------------------------------------- 读取
    @classmethod
    def load(cls, root: Path | str) -> "Bundle":
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if set(manifest) != set(cls.FIELDS):
            raise ValueError("清单与读取器登记的表不一致")
        tables = {name: _read_table(root, name, fields) for name, fields in cls.FIELDS.items()}

        students = {
            row["id"]: Student(
                id=row["id"],
                wave=int(row["wave"]),
                track=row["track"],
                rotation_end=parse_time(row["rotation_end"]),
                policy=row["policy"],
            )
            for row in tables["student.csv"]
        }
        first_scores = [
            FirstScore(
                id=row["id"],
                student=row["student"],
                skill=row["skill"],
                value=float(row["value"]),
                received=parse_time(row["received"]),
                status=row["status"],
                supersedes=row["supersedes"],
            )
            for row in tables["first_score.csv"]
        ]
        invitations = {
            row["student"]: Invitation(
                student=row["student"],
                policy=row["policy"],
                assigned=row["assigned"] == "1",
                offered_kind=row["offered_kind"],
                prob=float(row["prob"]),
                completed=row["completed"] == "1",
                coach=row["coach"] or None,
                sent=parse_time(row["sent"]),
            )
            for row in tables["invitation.csv"]
        }
        post_scores = {
            row["student"]: PostScore(
                student=row["student"],
                book=row["book"],
                observed=parse_time(row["observed"]),
                received=parse_time(row["received"]),
                value=float(row["value"]),
                status=row["status"],
            )
            for row in tables["post_score.csv"]
        }
        policies = {
            row["id"]: Policy(
                id=row["id"],
                tracks=tuple(part for part in row["tracks"].split(",") if part),
                kind=row["kind"],
                review_at_day=int(row["review_at_day"]),
                minimum_short_fraction=float(row["minimum_short_fraction"]),
                full_need_score=float(row["full_need_score"]),
            )
            for row in tables["policy.csv"]
        }
        slots = {
            row["id"]: Slot(
                id=row["id"],
                coach=row["coach"],
                day=row["day"],
                begin=int(row["begin"]),
                capacity=int(row["capacity"]),
                kind=row["kind"],
                skills=tuple(part for part in row["skills"].split(",") if part),
            )
            for row in tables["slot.csv"]
        }
        appointments = [
            Appointment(
                id=row["id"],
                student=row["student"],
                slot=row["slot"],
                state=row["state"],
                received=parse_time(row["received"]),
            )
            for row in tables["appointment.csv"]
        ]
        return cls(
            students,
            first_scores,
            invitations,
            post_scores,
            policies,
            slots,
            appointments,
            root,
        )

    # ---------------------------------------------------------------- 时点视图
    def scores_at(self, student_id: str, at: datetime) -> Dict[str, float]:
        """返回时点 ``at`` 可见的最终版初测分。

        同一技能后到的记录若 ``supersedes`` 指定了更早的记录，则旧记录失效；
        时点之后才收到的记录一律不可见。同一技能并列时取 id 较大者，保证可回放。
        """
        visible = [row for row in self._first_by_student.get(student_id, []) if row.received <= at]
        visible_ids = {row.id for row in visible}
        replaced = {
            row.supersedes
            for row in visible
            if row.supersedes and row.supersedes in visible_ids
        }
        latest: Dict[str, FirstScore] = {}
        for row in visible:
            if row.id in replaced:
                continue
            current = latest.get(row.skill)
            if current is None or (row.received, row.id) > (current.received, current.id):
                latest[row.skill] = row
        return {skill: row.value for skill, row in sorted(latest.items())}

    def score_events(self, student_id: str, at: Optional[datetime] = None) -> List[FirstScore]:
        """按收到时间列出初测与复核记录。"""
        rows = sorted(self._first_by_student.get(student_id, []), key=lambda r: (r.received, r.id))
        if at is not None:
            rows = [row for row in rows if row.received <= at]
        return rows

    def initial_scores(self, student_id: str) -> Dict[str, float]:
        """首次评分（不含复核），用于描述当时通常做法下的观察。"""
        latest: Dict[str, FirstScore] = {}
        for row in self.score_events(student_id):
            if row.status == "initial" and row.skill not in latest:
                latest[row.skill] = row
        return {skill: row.value for skill, row in sorted(latest.items())}

    def post_at(self, student_id: str, at: Optional[datetime] = None) -> Optional[PostScore]:
        post = self.post_scores.get(student_id)
        if post is None or (at is not None and post.received > at):
            return None
        return post

    def invitations_at(self, at: datetime) -> Dict[str, Invitation]:
        return {sid: row for sid, row in self.invitations.items() if row.sent <= at}

    def appointments_at(self, at: datetime) -> List[Appointment]:
        return [row for row in self.appointments if row.received <= at]

    def confirmed_appointments(
        self, student_id: Optional[str] = None, at: Optional[datetime] = None
    ) -> List[Appointment]:
        return [
            item
            for item in self.appointments
            if item.confirmed
            and (student_id is None or item.student == student_id)
            and (at is None or item.received <= at)
        ]

    def requested_appointments(self, student_id: Optional[str] = None) -> List[Appointment]:
        return [
            item
            for item in self.appointments
            if not item.confirmed and (student_id is None or item.student == student_id)
        ]

    def slots_before(self, student_id: str, at: datetime) -> List[Slot]:
        """学员转科截止前可用的场次。"""
        student = self.students[student_id]
        return [
            slot
            for slot in self.slots.values()
            if slot_start(slot) < student.rotation_end and slot_start(slot) > at
        ]

    def review_due(self, student_id: str) -> datetime:
        """评分者复核的应到时刻：初评后由本轮政策 ``review_at_day`` 指定。"""
        policy = self.policies[self.students[student_id].policy]
        initials = [row for row in self.score_events(student_id) if row.status == "initial"]
        return min(row.received for row in initials) + timedelta(days=policy.review_at_day)

    def post_due(self, student_id: str) -> datetime:
        """独立后测到期日：初测后第八日。"""
        initials = [row for row in self.score_events(student_id) if row.status == "initial"]
        return min(row.received for row in initials) + timedelta(days=8)

    def decision_time(self, student_id: str) -> datetime:
        """该轮决策时点：邀约发出时刻。"""
        return self.invitations[student_id].sent

    def data_cutoff(self) -> datetime:
        """资料中最后一条收到记录的时刻；此后发生的事都不可见。"""
        times = [row.received for row in self.post_scores.values()]
        times += [row.received for row in self.first_scores]
        times += [row.sent for row in self.invitations.values()]
        times += [row.received for row in self.appointments]
        return max(times)
