"""生成教务与协调员报告的命令行入口。

示例：

    python3 -m support_ops.cli data --wave 9 \
        --initial-at 2026-11-13T12:00:00+08:00 \
        --recheck-at 2026-11-14T09:00:00+08:00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bundle import Bundle
from .report import build_report, dump_json, render_markdown


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="教学支持筛查、收益核验与排班报告")
    parser.add_argument("bundle", type=Path, help="含清单及数据表的目录")
    parser.add_argument("--wave", type=int, default=9, help="要排班的轮次")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--initial-at", help="第一阶段筛查时点（ISO 时间）")
    parser.add_argument("--recheck-at", help="复核后第二阶段筛查时点（ISO 时间）")
    args = parser.parse_args(argv)
    if bool(args.initial_at) != bool(args.recheck_at):
        parser.error("--initial-at 与 --recheck-at 需要同时给出")

    try:
        bundle = Bundle.load(args.bundle)
        report = build_report(
            bundle, wave=args.wave,
            initial_at=args.initial_at, recheck_at=args.recheck_at,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"资料读取失败：{error}\n")
    if args.format == "json":
        print(dump_json(report))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
