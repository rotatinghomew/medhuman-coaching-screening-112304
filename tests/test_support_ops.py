"""核验时点、因果估计与排班口径。"""

import unittest
from datetime import datetime
from pathlib import Path

from support_ops import (
    Bundle,
    availability_by_track,
    build_cohort,
    build_plan,
    cace,
    compare_scenarios,
    intent_to_treat,
    label_bias,
    missing_bounds,
    missing_table,
    naive_receipt,
    usual_practice_risk,
    wave_audit,
)
from support_ops.effects import (
    balance_table,
    coach_summary,
    improved_by_help,
    intent_to_treat_clustered,
    itt_without_imbalanced,
    never_eligible,
    same_coach,
)
from support_ops.schedule import two_stage_plan

DATA = Path(__file__).resolve().parents[1] / "data"
TZ = "+08:00"


def at(text):
    return datetime.fromisoformat(text + TZ)


class BundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Bundle.load(DATA)
        cls.records = build_cohort(cls.bundle)

    def test_recheck_only_visible_after_receipt(self):
        before = self.bundle.scores_at("W9R016", at("2026-11-13T12:00:00"))
        after = self.bundle.scores_at("W9R016", at("2026-11-14T09:00:00"))
        self.assertEqual(before["explain"], 57.2)
        self.assertEqual(after["explain"], 39.0)
        self.assertEqual(set(after), {"listen", "explain", "respond"})

    def test_superseded_record_not_double_counted(self):
        rows = self.bundle.score_events("W9R016")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1].supersedes, "W9R016-1")
        self.assertEqual(len(self.bundle.scores_at("W9R016", at("2026-11-14T09:00:00"))), 3)

    def test_post_score_status_is_respected(self):
        same_coach = [r for r in self.records if r.post is not None and not r.post_independent]
        self.assertEqual(len(same_coach), 40)
        self.assertTrue(all(r.post.status == "same_coach" for r in same_coach))

    def test_wave_nine_is_immature(self):
        wave_nine = [r for r in self.records if r.wave == 9]
        self.assertEqual(len(wave_nine), 64)
        self.assertFalse(any(r.has_post for r in wave_nine))


class EstimationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Bundle.load(DATA)
        cls.records = build_cohort(cls.bundle)

    def test_itt_and_cace_are_positive_and_ordered(self):
        itt = intent_to_treat(self.records)
        cace_est, first = cace(self.records)
        self.assertGreater(itt.value, 0)
        self.assertGreater(cace_est.value, itt.value)
        self.assertGreater(first.value, 0.5)
        self.assertLess(first.value, 1)

    def test_usual_practice_risk_uses_controls_only(self):
        risk = usual_practice_risk(self.records)
        pool = [r for r in self.records if not r.assigned and r.post_independent]
        self.assertEqual(risk["n"], len(pool))
        self.assertAlmostEqual(risk["risk"], risk["events"] / risk["n"])

    def test_naive_receipt_is_flagged_non_causal(self):
        naive = naive_receipt(self.records)
        self.assertGreater(naive["difference"], 5)
        self.assertIn("不能解释为辅导收益", naive["note"])

    def test_missing_not_counted_as_success(self):
        cutoff = self.bundle.data_cutoff()
        table = missing_table(self.records, cutoff)
        for row in table:
            self.assertEqual(
                row["n"],
                row["observed"] + row["same_coach"] + row["pending"] + row["missing"],
            )
        bounds = missing_bounds(self.records, cutoff)
        self.assertLess(bounds["difference_low"], bounds["difference_high"])
        self.assertGreaterEqual(bounds["assigned"]["low"], -100)

    def test_helped_over_line_is_not_labelled_low_risk(self):
        helped = improved_by_help(self.records)
        self.assertTrue(helped["available"])
        self.assertGreater(helped["helped_over_line"], 0)
        self.assertLessEqual(helped["helped_over_line"], helped["post_pass"])

    def test_same_coach_excluded_from_primary_analysis(self):
        info = same_coach(self.records)
        self.assertEqual(info["same_coach_n"], 40)
        self.assertEqual(info["independent_n"], 432)

    def test_balance_flags_imbalanced_stratum(self):
        rows = balance_table(self.records)
        flagged = [row for row in rows if not row["comparable"]]
        self.assertTrue(flagged)
        self.assertIn("不平衡", flagged[0]["note"])
        sensitivity = itt_without_imbalanced(self.records)
        self.assertTrue(sensitivity.identified)
        self.assertLess(sensitivity.value, intent_to_treat(self.records).value)

    def test_coach_cluster_and_summary(self):
        clustered = intent_to_treat_clustered(self.records)
        plain = intent_to_treat(self.records)
        self.assertAlmostEqual(clustered.value, plain.value)
        self.assertGreater(clustered.high - clustered.low, 0)
        coaches = coach_summary(self.records)
        self.assertEqual(len(coaches), 7)
        self.assertTrue(all(row["assigned"] > 0 for row in coaches))
        self.assertGreater(sum(row["completed"] for row in coaches), 0)

    def test_never_eligible_intervals_described(self):
        rows = never_eligible(self.records)
        self.assertEqual([row["policy"] for row in rows], ["P0", "P1", "P2"])
        p0 = next(row for row in rows if row["policy"] == "P0")
        self.assertEqual(p0["assigned_high"], 0)
        self.assertIn("没有随机对照", p0["note"])


class ScreeningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Bundle.load(DATA)
        cls.records = build_cohort(cls.bundle)

    def test_label_bias_misses_more_with_treated_outcomes(self):
        rows = label_bias(self.records)
        by_wave = {}
        for row in rows:
            by_wave.setdefault(row["wave"], {})[row["label"]] = row
        any_worse = False
        for wave, labels in by_wave.items():
            clean = labels["只用对照"]["missed"]
            contaminated = labels["全部已观察"]["missed"]
            self.assertGreaterEqual(contaminated, clean)
            any_worse = any_worse or contaminated > clean
        self.assertTrue(any_worse)

    def test_availability_marks_unsupported_cells(self):
        rows = availability_by_track(self.records)
        unsupported = [row for row in rows if row.get("itt") is None]
        self.assertTrue(unsupported)
        for row in unsupported:
            self.assertIn("不足", row["reason"])


class SchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = Bundle.load(DATA)
        cls.records = build_cohort(cls.bundle)
        cls.initial_at = at("2026-11-13T12:00:00")
        cls.recheck_at = at("2026-11-14T09:00:00")

    def _check_plan(self, plan, as_of):
        from support_ops import slot_start

        capacity = {slot.id: slot.capacity for slot in self.bundle.slots.values()}
        usage = {}
        for placement in plan.placements:
            student = self.bundle.students[placement.student]
            confirmed = {
                item.slot
                for item in self.bundle.confirmed_appointments(placement.student, as_of)
            }
            full = []
            for sid in placement.sessions:
                slot = self.bundle.slots[sid]
                if sid not in confirmed:
                    usage[sid] = usage.get(sid, 0) + 1
                self.assertLess(slot_start(slot), student.rotation_end)
                if slot.kind == "full":
                    full.append(slot)
            self.assertEqual(len(placement.sessions), len(set(placement.sessions)))
            days = [self.bundle.slots[sid].day for sid in placement.sessions]
            self.assertEqual(len(days), len(set(days)), "同一学员同日只能有一次课")
            full.sort(key=slot_start)
            self.assertEqual(len(full) % 2, 0)
            for index in range(0, len(full), 2):
                first, second = full[index], full[index + 1]
                self.assertEqual(first.coach, second.coach)
                gap = abs((slot_start(second).date() - slot_start(first).date()).days)
                self.assertGreaterEqual(gap, 1)
                self.assertLessEqual(gap, 3)
        for sid, count in usage.items():
            self.assertLessEqual(count, capacity[sid], f"{sid} 超容量")

    def test_quota_plan_is_feasible(self):
        plan = build_plan(self.bundle, self.records, 9, "quota", as_of=self.initial_at)
        self._check_plan(plan, self.initial_at)
        self.assertGreater(plan.summary()["served"], 0)
        self.assertEqual(plan.summary()["unserved"], sum(1 for p in plan.unserved))

    def test_guarantee_plan_meets_minimum_short_fraction(self):
        plan = build_plan(self.bundle, self.records, 9, "guarantee", as_of=self.initial_at)
        self._check_plan(plan, self.initial_at)
        summary = plan.summary()
        self.assertGreaterEqual(summary["short_tier"], summary["minimum_short"])
        self.assertAlmostEqual(plan.minimum_short_fraction, 0.25)

    def test_two_stage_keeps_confirmed_and_flags_changes(self):
        stages = two_stage_plan(
            self.bundle, self.records, 9, self.initial_at, self.recheck_at
        )
        self.assertTrue(stages["stage_changed"])
        self.assertEqual(stages["after"].stage, "recheck")
        self.assertEqual(stages["before"].stage, "initial")
        # W9R016 的已确认答疑保留，且第一阶段已确认场次没被改动
        retained = {row["student"]: row["sessions"] for row in stages["confirmed_retained"]}
        self.assertIn("Z741", retained["W9R016"])
        for item in stages["changed_students"]:
            self.assertFalse(item["lost_confirmed"])
            if item["student"] == "W9R016":
                self.assertIn("Z741", item["same_slot_reused"])
                self.assertFalse(item["sessions_changed"])
                self.assertFalse(item["needs_explicit_confirmation"])
            if item["student"] == "W9R021":
                # 确认书在阶段之间才可见，第一阶段提案被取代，不算挤掉已确认场次
                self.assertTrue(item["confirmed_newly_visible"])

    def test_recheck_is_not_double_counted(self):
        stages = two_stage_plan(
            self.bundle, self.records, 9, self.initial_at, self.recheck_at
        )
        before = sum(len(p.sessions) for p in stages["before"].placements)
        after = sum(len(p.sessions) for p in stages["after"].placements)
        self.assertLessEqual(abs(after - before), 6)

    def test_requested_slot_not_reserved(self):
        plan = build_plan(self.bundle, self.records, 9, "quota", as_of=self.initial_at)
        self.assertTrue(any("W9R024" in note and "不占座" in note for note in plan.notes))

    def test_compare_scenarios_reports_rule_only_served(self):
        result = compare_scenarios(
            self.bundle, self.records, 9, as_of=self.initial_at
        )
        self.assertIn("quota", result)
        self.assertIn("guarantee", result)
        self.assertGreaterEqual(
            result["quota"]["summary"]["by_rule_only"], 0
        )


class AuditTest(unittest.TestCase):
    def test_wave_audit_marks_immature_posts(self):
        bundle = Bundle.load(DATA)
        audit = wave_audit(bundle, 9)
        self.assertEqual(audit["post_mature"], 0)
        self.assertEqual(len(audit["immature"]), 64)
        self.assertEqual(len(audit["stage_changes"]), 1)
        self.assertEqual(audit["stage_changes"][0]["student"], "W9R016")


if __name__ == "__main__":
    unittest.main()
