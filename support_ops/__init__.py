"""模拟会谈教学支持的筛查、因果核验与排班工具。

所有资料均为合成教学数据，不含真实学生、患者或诊疗记录。
"""

from .audit import carry_forward, wave_audit
from .bundle import Bundle, Slot, slot_start
from .cohort import (
    Record,
    build_cohort,
    confirmed_commitments,
    controls,
    feasibility,
    full_block_pairs,
    mean,
    select,
)
from .effects import (
    balance_table,
    by_modality,
    cace,
    coach_summary,
    completion_rates,
    improved_by_help,
    intent_to_treat,
    intent_to_treat_clustered,
    itt_without_imbalanced,
    label_bias,
    missing_bounds,
    missing_table,
    naive_receipt,
    never_eligible,
    same_coach,
    usual_practice_risk,
)
from .schedule import (
    build_need,
    build_plan,
    compare_scenarios,
    deficits,
    two_stage_plan,
)
from .screening import (
    availability_by_track,
    backtest,
    coverage_check,
    model_comparison,
    risk_model,
)

__all__ = [
    "Bundle",
    "Slot",
    "slot_start",
    "Record",
    "build_cohort",
    "select",
    "controls",
    "mean",
    "feasibility",
    "full_block_pairs",
    "confirmed_commitments",
    "intent_to_treat",
    "intent_to_treat_clustered",
    "cace",
    "by_modality",
    "coach_summary",
    "balance_table",
    "itt_without_imbalanced",
    "completion_rates",
    "usual_practice_risk",
    "missing_table",
    "missing_bounds",
    "naive_receipt",
    "label_bias",
    "improved_by_help",
    "never_eligible",
    "same_coach",
    "risk_model",
    "backtest",
    "model_comparison",
    "coverage_check",
    "availability_by_track",
    "build_need",
    "build_plan",
    "compare_scenarios",
    "two_stage_plan",
    "deficits",
    "wave_audit",
    "carry_forward",
]
