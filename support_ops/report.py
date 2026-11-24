"""把分析结果整理成教务与协调员可读的报告。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .audit import carry_forward, wave_audit
from .bundle import Bundle
from .cohort import Record, build_cohort, feasibility
from .effects import (
    balance_table,
    by_modality,
    cace,
    coach_summary,
    completion_rates,
    intent_to_treat,
    intent_to_treat_clustered,
    improved_by_help,
    itt_without_imbalanced,
    label_bias,
    missing_bounds,
    missing_table,
    naive_receipt,
    never_eligible,
    same_coach,
    usual_practice_risk,
)
from .schedule import compare_scenarios, two_stage_plan
from .screening import (
    availability_by_track,
    backtest,
    coverage_check,
    model_comparison,
)


def _fmt(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


class _EstimateView:
    """把估计字典包装成带属性的对象，复用区间格式。"""

    def __init__(self, row: Dict[str, object]):
        self.value = row.get("value")
        self.low = row.get("low")
        self.high = row.get("high")

    @property
    def identified(self) -> bool:
        return self.value is not None


def _interval(row: Dict[str, object]) -> str:
    est = _EstimateView(row)
    if not est.identified:
        return "无法给出"
    return f"{_fmt(est.value)}（95% {_fmt(est.low)} 至 {_fmt(est.high)}）"


def evidence_by_track(records: Sequence[Record]) -> Dict[tuple, bool]:
    """哪一档「轮转 × 练习能力」有可用的收益证据。

    要同时满足：有可估计的收益区间，且邀约组与对照组各至少 10 人。
    """
    return {
        (row["track"], row["band"]): bool(
            row.get("itt") is not None
            and row.get("assigned", 0) >= 10
            and row.get("control", 0) >= 10
        )
        for row in availability_by_track(records)
    }


def build_report(bundle: Bundle, wave: int = 9, initial_at: Optional[str] = None,
                 recheck_at: Optional[str] = None) -> Dict[str, object]:
    waves = {student.wave for student in bundle.students.values()}
    if wave not in waves:
        raise ValueError(f"轮次 {wave} 不在资料中（可用：{min(waves)} 至 {max(waves)}）")
    records = build_cohort(bundle)
    cutoff = bundle.data_cutoff()
    evidence = evidence_by_track(records)
    cace_estimate, first_stage = cace(records)
    report: Dict[str, object] = {
        "wave": wave,
        "inventory": {
            "students": len(bundle.students),
            "waves": sorted({r.wave for r in bundle.students.values()}),
            "first_scores": len(bundle.first_scores),
            "invitations": len(bundle.invitations),
            "post_scores": len(bundle.post_scores),
            "slots": len(bundle.slots),
            "appointments": len(bundle.appointments),
        },
        "usual_practice_risk": usual_practice_risk(records),
        "itt": intent_to_treat(records).as_dict(),
        "cace": cace_estimate.as_dict(),
        "first_stage": first_stage.as_dict(),
        "by_modality": {key: est.as_dict() for key, est in by_modality(records).items()},
        "balance": balance_table(records),
        "coach_summary": coach_summary(records),
        "sensitivity": itt_without_imbalanced(records).as_dict(),
        "clustered": intent_to_treat_clustered(records).as_dict(),
        "completion": completion_rates(records),
        "missing": missing_table(records, cutoff),
        "missing_bounds": missing_bounds(records, cutoff),
        "same_coach": same_coach(records),
        "never_eligible": never_eligible(records),
        "naive_receipt": naive_receipt(records),
        "availability": availability_by_track(records),
        "backtest": backtest(records),
        "model_comparison": model_comparison(records),
        "coverage": coverage_check(records),
        "label_bias": label_bias(records),
        "improved_by_help": improved_by_help(records),
        "carry_forward": carry_forward(bundle),
        "wave_audit": wave_audit(bundle, wave),
    }
    if initial_at and recheck_at:
        stages = two_stage_plan(
            bundle, records, wave,
            datetime.fromisoformat(initial_at), datetime.fromisoformat(recheck_at),
            evidence=evidence,
        )
        scenarios = compare_scenarios(
            bundle, records, wave,
            as_of=datetime.fromisoformat(recheck_at), evidence=evidence,
        )
        report["stages"] = {
            "before": stages["before"].summary(),
            "after": stages["after"].summary(),
            "stage_changed": stages["stage_changed"],
            "changed_students": stages["changed_students"],
            "squeezed_students": stages["squeezed_students"],
            "confirmation_needed": stages["confirmation_needed"],
            "notes": stages["notes"],
        }
        report["scenarios"] = {
            key: value["summary"] for key, value in scenarios.items() if key != "notes"
        }
        report["scenario_notes"] = scenarios["notes"]
        report["plan_details"] = _plan_details(
            bundle, scenarios["quota"]["plan"], datetime.fromisoformat(recheck_at)
        )
    else:
        scenarios = compare_scenarios(bundle, records, wave, evidence=evidence)
        report["scenarios"] = {
            key: value["summary"] for key, value in scenarios.items() if key != "notes"
        }
        report["scenario_notes"] = scenarios["notes"]
        default_at = max(
            (r.decision_at for r in records if r.wave == wave), default=None
        )
        report["plan_details"] = _plan_details(
            bundle, scenarios["quota"]["plan"], default_at
        )
    report["evidence_by_track"] = {
        f"{track} × {band}": bool(supported) for (track, band), supported in evidence.items()
    }
    return report


def _plan_details(bundle: Bundle, plan, as_of: Optional[datetime]) -> List[Dict[str, object]]:
    """排班学员的转科前练习日与整段训练可行性（从筛查时点起算）。"""
    rows = []
    for placement in plan.placements:
        if as_of is None:
            continue
        info = feasibility(bundle, placement.student, as_of)
        rows.append(
            {
                "student": placement.student,
                "rotation_end": info["rotation_end"],
                "practice_days_before_rotation": info["practice_days_before_rotation"],
                "can_complete_full_block": info["can_complete_full_block"],
                "sessions": placement.sessions,
            }
        )
    return rows


def render_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    inv = report["inventory"]
    lines.append(f"# 第 {report['wave']} 轮教学支持报告")
    lines.append("")
    lines.append(
        "资料为合成教学数据，不含真实学员、患者或诊疗记录；"
        "以下结果只用于学习支持的筛查与排班。"
    )
    lines.append("")
    lines.append("## 资料清点")
    lines.append("")
    lines.append(f"- 学员 {inv['students']} 人，轮次 {len(inv['waves'])} 轮")
    lines.append(
        f"- 初评 {inv['first_scores']} 条、邀约 {inv['invitations']} 条、"
        f"后测 {inv['post_scores']} 条、场次 {inv['slots']} 个、预约 {inv['appointments']} 条"
    )
    lines.append("")

    risk = report["usual_practice_risk"]
    lines.append("## 当时通常做法下的后测风险")
    lines.append("")
    if risk.get("identified"):
        lines.append(
            f"- 成熟轮次里未获邀约学员 {risk['n']} 人中有 {risk['events']} 人后测仍低于 60，"
            f"风险 {risk['risk'] * 100:.1f}%。"
        )
        lines.append(
            f"- 口径：{risk['note']}；它描述风险，不是额外辅导收益。"
        )
    else:
        lines.append(f"- 无法给出：{risk.get('reason') or '没有可用对照'}。")
    lines.append("")

    lines.append("## 额外辅导收益（只在有可信对照时给出）")
    lines.append("")
    itt = report["itt"]
    if itt["value"] is None:
        lines.append("- 没有成熟的随机邀约轮次，不能给出额外辅导收益优先级。")
    else:
        lines.append(f"- 邀约意向收益（ITT）：{_interval(itt)}；{itt['note']}。")
        lines.append(
            f"- 完成辅导者的额外收益（CACE）：{_interval(report['cace'])}；"
            "只识别实际完成者。"
        )
        lines.append(
            f"- 一阶段（邀约→完成）：{_interval(report['first_stage'])}；"
            "说明实际领取不等于被分配。"
        )
        sensitivity = report["sensitivity"]
        lines.append(
            f"- 敏感性（去掉基线不平衡的层）：{_interval(sensitivity)}；"
            f"{sensitivity['note']}。"
        )
        clustered = report["clustered"]
        lines.append(
            f"- 敏感性（按教练聚类标准误）：{_interval(clustered)}；{clustered['note']}。"
        )
    lines.append("")
    lines.append("### 轮次内邀约组与对照组是否可比")
    lines.append("")
    lines.append("| 政策 | 邀约概率 | 邀约组 | 对照组 | 基线差 | t | 可比 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report["balance"]:
        lines.append(
            f"| {row['policy']} | {row['prob']:.2f} | {row['treated']} | {row['control']} | "
            f"{_fmt(row.get('diff'), 2)} | {_fmt(row.get('t'), 2)} | "
            f"{'是' if row['comparable'] else '否'} |"
        )
    lines.append("")
    for row in report["balance"]:
        if row.get("note"):
            lines.append(f"- {row['policy']}（概率 {row['prob']:.2f}）：{row['note']}。")
    lines.append("")
    for row in report["balance"]:
        if row["treated"] < 10 or row["control"] < 10:
            lines.append(
                f"- {row['policy']}（概率 {row['prob']:.2f}）：邀约组 {row['treated']} 人、"
                f"对照组 {row['control']} 人，该层单独估计会很不稳定。"
            )
    lines.append("")
    lines.append("### 按邀约方式（描述性）")
    lines.append("")
    for key, est in report["by_modality"].items():
        name = "低负担答疑" if key == "short" else "整段训练"
        completeness = report["completion"].get(key, {})
        lines.append(
            f"- {name}：{_interval(est)}；"
            f"完成 {completeness.get('completed', 0)}/{completeness.get('assigned', 0)}。"
            f"{est['note']}。"
        )
    lines.append("")

    naive = report["naive_receipt"]
    lines.append("### 各教练的带教量与结果")
    lines.append("")
    lines.append("| 教练 | 邀约 | 完成 | 完成率 | 已复测 | 初评均分 | 已观察增益 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report["coach_summary"]:
        lines.append(
            f"| {row['coach']} | {row['assigned']} | {row['completed']} | "
            f"{_fmt(row['completion_rate'], 2)} | {row['observed']} | "
            f"{_fmt(row['mean_screened'])} | {_fmt(row['mean_gain'])} |"
        )
    lines.append("")
    lines.append(
        "教练之间的人数与结果差异只作描述；同一教练关联的结果不按独立观察处理，"
        "主估计只用独立后测，并另给按教练聚类的标准误作敏感性检查。"
    )
    lines.append("")

    lines.append("### 为什么实际领取者与未领取者的差值不能当收益")
    lines.append("")
    if naive.get("available"):
        lines.append(
            f"- 完成者 {naive['completed']} 人增益 {_fmt(naive['completed_gain'])}，"
            f"未完成者 {naive['not_completed']} 人增益 {_fmt(naive['not_completed_gain'])}，"
            f"差 {_fmt(naive['difference'])}。"
        )
        lines.append(
            f"- 完成者初评均分 {_fmt(naive['mean_screened_completed'])}，"
            f"未完成者 {_fmt(naive['mean_screened_not'])}；{naive['note']}。"
        )
    lines.append("")

    lines.append("## 缺失与未成熟后测")
    lines.append("")
    lines.append("| 政策 | 邀约 | 人数 | 独立后测 | 同教练复评 | 未到期 | 已到期缺失 | 已观察增益 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["missing"]:
        lines.append(
            f"| {row['policy']} | {'是' if row['assigned'] else '否'} | {row['n']} | "
            f"{row['observed']} | {row['same_coach']} | {row['pending']} | "
            f"{row['missing']} | {_fmt(row['observed_gain'])} |"
        )
    lines.append("")
    lines.append("缺失后测既不当作成功也不当作失败；只报告已观察部分，缺失部分保留为未知。")
    bounds = report["missing_bounds"]
    if bounds.get("assigned") and "low" in bounds["assigned"]:
        lines.append(
            f"- 极端边界：邀约组收益在 {_fmt(bounds['assigned']['low'])} 至 "
            f"{_fmt(bounds['assigned']['high'])}，对照组在 {_fmt(bounds['control']['low'])} 至 "
            f"{_fmt(bounds['control']['high'])}；差值边界 "
            f"{_fmt(bounds['difference_low'])} 至 {_fmt(bounds['difference_high'])}。"
        )
        lines.append(f"- {bounds['note']}。")
    lines.append("")
    coach = report["same_coach"]
    lines.append(
        f"- 同一教练复评的后测 {coach['same_coach_n']} 条、独立后测 "
        f"{coach['independent_n']} 条；主分析只用独立后测，{coach['note']}"
    )
    lines.append("")

    lines.append("## 筛查回放与模型比较")
    lines.append("")
    lines.append("| 轮次 | 训练人数 | 检查人数 | 事件 | AUC | Brier | 灵敏度 | PPV |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["backtest"]:
        if not row.get("trained"):
            lines.append(
                f"| {row['wave']} | {row['train_n']} | {row['test_n']} | {row['events']} | "
                f"— | — | — | 此前轮次不足 |"
            )
            continue
        lines.append(
            f"| {row['wave']} | {row['train_n']} | {row['test_n']} | {row['events']} | "
            f"{_fmt(row.get('auc'), 3)} | {_fmt(row.get('brier'), 3)} | "
            f"{_fmt(row.get('sensitivity'), 2)} | {_fmt(row.get('ppv'), 2)} |"
        )
    lines.append("")
    lines.append("| 模型 | 灵敏度 | 特异度 | PPV | 标记率 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report["model_comparison"]:
        lines.append(
            f"| {row['model']} | {_fmt(row.get('sensitivity'), 2)} | "
            f"{_fmt(row.get('specificity'), 2)} | {_fmt(row.get('ppv'), 2)} | "
            f"{_fmt(row.get('flag_rate'), 2)} |"
        )
    lines.append("")

    lines.append("### 从未被允许选择的区间")
    lines.append("")
    lines.append("| 政策 | 人数 | 从未邀约 | 低分邀约 | 高分邀约 | 口径 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in report["never_eligible"]:
        lines.append(
            f"| {row['policy']} | {row['n']} | {row['never_assigned']} | "
            f"{row['assigned_low']} | {row['assigned_high']} | {row['note']} |"
        )
    lines.append("")

    helped = report["improved_by_help"]
    lines.append("### 得到帮助才过线的学员")
    lines.append("")
    if helped.get("available"):
        lines.append(
            f"- 完成辅导且已复测 {helped['completed']} 人，后测达标 "
            f"{helped['post_pass']} 人；按额外收益回推，其中 {helped['helped_over_line']} 人"
            f"若未获辅导本会低于 60，另 {helped['still_below']} 人后测仍低于 60。"
        )
        lines.append(f"- {helped['note']}。")
        lines.append(
            "- 这些“得到帮助才过线”的学员不能被标成低风险；否则下一届筛查会漏掉同类学员。"
        )
    else:
        lines.append(f"- 无法回推：{helped.get('reason') or '没有可信对照'}。")
    lines.append("")

    lines.append("### 标签必须只用可信对照")
    lines.append("")
    lines.append(
        "把获得辅导后进步的人当成“原本不需要”，下一届会漏掉他们。"
        "下面并排两种训练标签方式（都用此前轮次训练、当轮对照评价）："
    )
    lines.append("")
    lines.append("| 轮次 | 标签 | 训练人数 | 漏判 | 灵敏度 | Brier |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in report["label_bias"]:
        lines.append(
            f"| {row['wave']} | {row['label']} | {row['train_n']} | {row['missed']} | "
            f"{_fmt(row['sensitivity'], 2)} | {_fmt(row['brier'], 3)} |"
        )
    lines.append("")

    lines.append("### 可信范围检查（扩窗回放）")
    lines.append("")
    lines.append("| 轮次 | 此前区间 | 当轮估计 | 是否覆盖 |")
    lines.append("| --- | --- | --- | --- |")
    for row in report["coverage"]:
        if not row.get("checked"):
            lines.append(f"| {row['wave']} | — | — | {row['reason']} |")
            continue
        lines.append(
            f"| {row['wave']} | {_fmt(row['prior_low'])} 至 {_fmt(row['prior_high'])} | "
            f"{_fmt(row['current'])} | {'覆盖' if row['covered'] else '未覆盖'} |"
        )
    lines.append("")

    lines.append("## 不同轮转与练习能力的可用性")
    lines.append("")
    lines.append("| 轮转 | 练习能力 | 人数 | 邀约 | 对照 | 收益区间 | 区间宽度 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report["availability"]:
        if row.get("itt") is None:
            lines.append(
                f"| {row['track']} | {row['band']} | {row['n']} | {row['assigned']} | "
                f"{row['control']} | 无法给出 | {row.get('reason', '')} |"
            )
            continue
        lines.append(
            f"| {row['track']} | {row['band']} | {row['n']} | {row['assigned']} | "
            f"{row['control']} | {_fmt(row['low'])} 至 {_fmt(row['high'])} | "
            f"{_fmt(row['width'])} |"
        )
    lines.append("")
    lines.append(
        "区间为空或过宽的组合无法支撑收益优先级；这些组合在排班中只能按保障规则处理。"
    )
    lines.append("")

    lines.append("## 转科前可完成的练习次数")
    lines.append("")
    plans = report.get("plan_details")
    if plans:
        lines.append("| 学员 | 轮转截止 | 可练习日 | 整段训练可完成 | 新增安排 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in plans:
            lines.append(
                f"| {row['student']} | {row['rotation_end'][:10]} | "
                f"{row['practice_days_before_rotation']} | "
                f"{'是' if row['can_complete_full_block'] else '否'} | "
                f"{', '.join(row['sessions']) or '无'} |"
            )
    else:
        lines.append("- 没有需要排班的筛查高风险学员。")
    lines.append("")

    lines.append("## 本轮排班比较")
    lines.append("")
    lines.append(
        "| 方案 | 筛查风险 | 已排 | 整段训练 | 低负担 | 全分项覆盖 | 未排 | 仅保障规则 | 剩余整段座位 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, label in (("quota", "全体名额"), ("guarantee", "最低覆盖")):
        row = report["scenarios"].get(key, {})
        lines.append(
            f"| {label} | {row.get('screened_risk', '—')} | {row.get('served', '—')} | "
            f"{row.get('full_tier', '—')} | {row.get('short_tier', '—')} | "
            f"{row.get('fully_covered', '—')} | {row.get('unserved', '—')} | "
            f"{row.get('by_rule_only', '—')} | {row.get('seats_left', {}).get('full', '—')} |"
        )
    lines.append("")
    lines.append(
        "「已排」按人头计；整段训练与低负担按学员计，同一学员可同时有两种安排。"
    )
    lines.append("")
    for note in report.get("scenario_notes", []):
        lines.append(f"- {note}")
    lines.append("")

    if "stages" in report:
        stages = report["stages"]
        lines.append("## 复核改判前后的两个阶段")
        lines.append("")
        lines.append(
            f"- 第一阶段已排 {stages['before']['served']} 人，"
            f"第二阶段已排 {stages['after']['served']} 人；"
            f"阶段是否变化：{'是' if stages['stage_changed'] else '否'}。"
        )
        newly_visible = [
            item["student"] for item in stages["changed_students"]
            if item["confirmed_newly_visible"] and item["sessions_changed"]
        ]
        if newly_visible:
            lines.append(
                f"- {'、'.join(newly_visible)} 的确认书在阶段之间才可见，"
                "第一阶段是提案，被替代不算挤掉已确认场次。"
            )
        for item in stages["changed_students"]:
            if item["confirmed_kept"] or item["lost_confirmed"]:
                lines.append(
                    f"- {item['student']}：第一阶段 {item['before']}，"
                    f"第二阶段 {item['after']}；已确认场次"
                    f"{'保留' if item['confirmed_kept'] else '丢失，需人工核对'}；"
                    f"{'沿用' + '、'.join(item['same_slot_reused']) if item['same_slot_reused'] else '未沿用第一阶段场次'}。"
                )
            elif item["before"] and not item["after"]:
                lines.append(
                    f"- {item['student']}：第一阶段 {item['before']}，第二阶段无场次；"
                    "转入等待名单。"
                )
            elif item["after"] and not item["before"]:
                lines.append(
                    f"- {item['student']}：第二阶段新增 {item['after']}。"
                )
        if stages["changed_students"]:
            lines.append(
                f"- 两个阶段之间共 {len(stages['changed_students'])} 人安排发生变化；"
                "未确认提案可调整，已确认场次只在明确确认后才动。"
            )
            lines.append("")
            lines.append("| 学员 | 第一阶段 | 第二阶段 | 已确认 | 需明确确认 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for item in stages["changed_students"]:
                lines.append(
                    f"| {item['student']} | {', '.join(item['before']) or '无'} | "
                    f"{', '.join(item['after']) or '无'} | "
                    f"{'是' if item['confirmed_kept'] else '否'} | "
                    f"{'是' if item['needs_explicit_confirmation'] else '否'} |"
                )
            lines.append("")
        if stages["squeezed_students"]:
            names = "、".join(item["student"] for item in stages["squeezed_students"])
            lines.append(
                f"- 复核阶段被挤出全部名额的学员：{names}；进入等待名单，"
                "不记为已辅导，也不与已确认场次混算。"
            )
        if stages["confirmation_needed"]:
            lines.append(
                f"- 第一阶段的已确认场次被改动的学员："
                f"{'、'.join(stages['confirmation_needed'])}；"
                "调整须教练与学员明确确认。"
            )
        else:
            lines.append("- 第一阶段已确认的场次都没有被改动；无需追加确认。")
        for note in stages["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    audit = report["wave_audit"]
    lines.append("## 本轮回看")
    lines.append("")
    if audit.get("available"):
        lines.append(
            f"- 名单 {audit['n']} 人，筛查高风险 {audit['screened_risk']} 人；"
            f"实际邀约 {audit['assigned']} 人，完成 {audit['completed']} 人。"
        )
        lines.append(
            f"- 后测成熟 {audit['post_mature']} 人，独立后测 {audit['post_independent']} 人；"
            f"未成熟 {len(audit['immature'])} 人，同一教练复评 {len(audit['same_coach'])} 人。"
        )
        if audit["immature"]:
            lines.append(
                f"- 未成熟名单：{'、'.join(audit['immature'][:12])}"
                f"{' 等' if len(audit['immature']) > 12 else ''}；这些结果不作为成功或失败。"
            )
        for change in audit["stage_changes"]:
            lines.append(
                f"- 阶段变化：{change['student']} 的 {change['skill']} 在 "
                f"{change['received']} 复核为 {change['value']}（取代 {change['supersedes']}）。"
            )
    lines.append("")

    carry = report["carry_forward"]
    lines.append("## 下一轮可带走的证据状态")
    lines.append("")
    lines.append("| 轮次 | 人数 | 成熟 | 未成熟 | 同教练复评 | 邀约 | 完成 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for wave, row in carry["waves"].items():
        lines.append(
            f"| {wave} | {row['n']} | {row['mature']} | {row['immature']} | "
            f"{row['same_coach']} | {row['assigned']} | {row['completed']} |"
        )
    lines.append("")
    for gap in carry["evidence_gaps"]:
        if not gap["covered"]:
            lines.append(f"- {gap['track']} × {gap['band']}：{gap['note']}")
    if not any(not gap["covered"] for gap in carry["evidence_gaps"]):
        lines.append("- 各轮转与练习能力段均有可用对照；下一轮可继续沿用同一套估计。")
    lines.append("")
    lines.append(
        "下一轮继续把对照安排留在证据不足的部位，不要用历史名单直接替代新轮次筛查。"
    )
    lines.append("")
    return "\n".join(lines)


def dump_json(report: Dict[str, object]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)
