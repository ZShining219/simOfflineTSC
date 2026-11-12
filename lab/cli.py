"""``python -m lab`` 命令行：台账与产物的查询、核查。

    python -m lab audit [--campaign C] [--json OUT]   各 campaign 证据完整度体检
    python -m lab show RUN_ID                         单个 run 的全链路溯源
    python -m lab find [--campaign C] [--arm A] ...   按条件列出 run
    python -m lab check --staged | --all | --commit-msg FILE   提交闸门（git hook 调用）
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict

from . import checks, evidence, ledger
from .provenance import REPO_ROOT


def _fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def _pad(text, width):
    return str(text) + " " * (width - _width(text))


def _table(headers, rows):
    widths = [max([_width(h)] + [_width(r[i]) for r in rows]) for i, h in enumerate(headers)]
    line = "  ".join(_pad(h, w) for h, w in zip(headers, widths))
    print(line)
    print("-" * _width(line))
    for row in rows:
        print("  ".join(_pad(c, w) for c, w in zip(row, widths)))


def _inspect(row):
    """定位 run 并判定代码身份，返回 (位置, 本机路径, 代码等级, 提交号, manifest)。"""
    where, path = evidence.locate(row.get("run_dir"))
    manifest = evidence.read_json(path / "run_manifest.json") if path else None
    level, commit = ledger.code_level(row, manifest)
    return where, path, level, commit, manifest


def cmd_audit(args):
    duplicates = []
    rows = [r for r in ledger.load(duplicates=duplicates).values()
            if not args.campaign or r.get("campaign") == args.campaign]
    stats = defaultdict(lambda: {"n": 0, "done": 0, "metrics": 0,
                                 "where": Counter(), "code": Counter(),
                                 "missing_unflagged": 0, "config_mismatch": 0})
    for row in rows:
        where, _, level, _, manifest = _inspect(row)
        s = stats[row.get("campaign")]
        s["config_mismatch"] += ledger.config_mismatch(row, manifest)
        s["n"] += 1
        s["done"] += row.get("status") == "DONE"
        s["metrics"] += ledger.has_metrics(row)
        s["where"][where] += 1
        s["code"][level] += 1
        if where == evidence.MISSING and not row.get("evidence_lost"):
            s["missing_unflagged"] += 1
    total = {"n": 0, "done": 0, "metrics": 0, "where": Counter(),
             "code": Counter(), "missing_unflagged": 0, "config_mismatch": 0}
    table = []
    for campaign, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        for key in ("n", "done", "metrics", "missing_unflagged", "config_mismatch"):
            total[key] += s[key]
        total["where"].update(s["where"])
        total["code"].update(s["code"])
        table.append(_audit_row(campaign, s))
    table.append(_audit_row("【合计】", total))
    print("证据核查（本机视角；missing = 本机与本仓工作树中都找不到 run_dir）")
    print(f"代码身份：verified=运行时记录 HEAD；registered=仅登记时提交号；"
          f"constant=写死常量 {ledger.CONSTANT_BASELINE_COMMIT[:8]}；none=无\n")
    _table(["campaign", "行数", "DONE", "有指标", "本机在盘", "工作树", "缺失(未标记)",
            "无run_dir", "verified", "registered", "constant", "none", "配置哈希不符"], table)
    if duplicates:
        print(f"\n重复登记的 run_id：{len(duplicates)} 处（统计取最后一行；需人工裁决哪行为准）")
        for run_id, first, again in duplicates[:10]:
            print(f"   {run_id}  行 {first} 与行 {again}")
        if len(duplicates) > 10:
            print(f"   …其余 {len(duplicates) - 10} 处见 --json 输出")
    if args.json:
        payload = {c: {**{k: v for k, v in s.items() if not isinstance(v, Counter)},
                       "where": dict(s["where"]), "code": dict(s["code"])}
                   for c, s in stats.items()}
        payload = {"campaigns": payload,
                   "duplicates": [list(d) for d in duplicates]}
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")


def _audit_row(name, s):
    w, c = s["where"], s["code"]
    on_disk = w[evidence.HERE] + w[evidence.ABSOLUTE]
    return [name, s["n"], s["done"], s["metrics"], on_disk, w[evidence.WORKTREE],
            s["missing_unflagged"], w[evidence.NO_RUN_DIR],
            c[ledger.CODE_VERIFIED], c[ledger.CODE_REGISTERED],
            c[ledger.CODE_CONSTANT], c[ledger.CODE_NONE], s["config_mismatch"]]


def _resolve(rows, query):
    if query in rows:
        return rows[query]
    hits = [run_id for run_id in rows if query in run_id]
    if len(hits) == 1:
        return rows[hits[0]]
    if not hits:
        sys.exit(f"台账中没有 run_id 包含「{query}」的记录")
    print(f"「{query}」匹配到 {len(hits)} 条，请给出更完整的 run_id：")
    for run_id in hits[:20]:
        print("  ", run_id)
    sys.exit(2)


def cmd_show(args):
    row = _resolve(ledger.load(), args.run_id)
    where, path, level, commit, manifest = _inspect(row)
    files = evidence.run_files(path)
    print(f"run_id      {row['run_id']}")
    print(f"台账行号    ledger/runs.jsonl:{row['_line_no']}")
    for key in ("campaign", "line", "arm", "net", "seed", "run_kind", "status",
                "plan_id", "eval_mode", "behavior_source", "alias_of",
                "superseded_by", "queue_file", "config_sha"):
        if row.get(key) not in (None, ""):
            print(f"{key:<11} {row[key]}")
    print(f"\n[产物] 位置={where}  run_dir={row.get('run_dir') or '-'}")
    if path and str(path) != row.get("run_dir"):
        print(f"       本机路径={path}")
    if where == evidence.MISSING:
        print("       本机找不到：可能在另一台机器，也可能已丢失"
              + ("（台账已标 evidence_lost）" if row.get("evidence_lost") else ""))
    print(f"\n[代码] 可信等级={level}  提交={commit or '-'}")
    code = (manifest or {}).get("code") or {}
    if code:
        print(f"       分支={code.get('branch')}  dirty={code.get('dirty')}"
              f"  改动文件数={code.get('dirty_file_count')}")
    if manifest:
        print(f"       机器={manifest.get('machine') or '-'}  config_hash={manifest.get('config_hash')}")
    if ledger.config_mismatch(row, manifest):
        print(f"       ⚠ 台账 config_sha={row['config_sha'][:16]}… 与 manifest config_hash 不一致，"
              "以 run 目录内配置快照为准")
    hashes = (files["config/config_hashes.json"] or {}).get("files") or {}
    if hashes:
        print("\n[配置快照] " + (str(path / 'config') if path else ""))
        for name, digest in hashes.items():
            print(f"       {name:<26} {digest[:16]}")
    status = files["run_status.json"]
    if status:
        print(f"\n[运行状态] {status.get('status')}  {status.get('started_at_utc')} → "
              f"{status.get('finished_at_utc')}  exit={status.get('exit_code')}")
    summary = files["evaluation/summary.json"]
    metrics = row.get("metrics") or {}
    if summary or metrics:
        print("\n[指标]  台账值与评估摘要并列；台账未注明取值口径，引用前先核对")
        for key in ledger.METRIC_KEYS:
            if key in metrics:
                print(f"       台账 {key:<20} {_fmt(metrics[key])}")
        if summary:
            for key in ("final_episode", "final_travel_time", "best_episode",
                        "best_travel_time", "best_checkpoint"):
                if key in summary:
                    print(f"       摘要 {key:<20} {_fmt(summary[key])}")
    if row.get("notes"):
        print(f"\n[备注] {row['notes']}")


def cmd_find(args):
    rows = []
    for row in ledger.load().values():
        if args.campaign and row.get("campaign") != args.campaign:
            continue
        if args.arm and row.get("arm") != args.arm:
            continue
        if args.net and args.net not in str(row.get("net")):
            continue
        if args.status and row.get("status") != args.status:
            continue
        if args.kind and row.get("run_kind") != args.kind:
            continue
        if args.seed is not None and row.get("seed") != args.seed:
            continue
        rows.append(row)
    shown = rows[:args.limit]
    table = []
    for row in shown:
        where = evidence.locate(row.get("run_dir"))[0] if args.check_disk else ""
        metrics = row.get("metrics") or {}
        table.append([row["run_id"], row.get("status"), row.get("run_kind"),
                      row.get("arm"), row.get("net"), _fmt(row.get("seed")),
                      _fmt(metrics.get("travel_time"), 2), where])
    _table(["run_id", "状态", "类型", "arm", "net", "seed", "travel_time",
            "位置" if args.check_disk else ""], table)
    if len(rows) > len(shown):
        print(f"\n共 {len(rows)} 条，仅显示前 {len(shown)} 条（--limit 调整）")


def cmd_check(args):
    if args.commit_msg:
        with open(args.commit_msg, encoding="utf-8") as handle:
            code = checks.report(checks.check_commit_message(handle.read()), "提交信息检查")
    elif args.all:
        code = checks.report(checks.check_all(REPO_ROOT), "全量结构检查")
    else:
        code = checks.report(checks.check_staged(REPO_ROOT), "提交前检查")
    sys.exit(code)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m lab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="各 campaign 证据完整度体检")
    audit.add_argument("--campaign")
    audit.add_argument("--json", help="同时写出机读结果到该路径")
    audit.set_defaults(func=cmd_audit)

    show = sub.add_parser("show", help="单个 run 的全链路溯源")
    show.add_argument("run_id", help="完整 run_id 或唯一子串")
    show.set_defaults(func=cmd_show)

    find = sub.add_parser("find", help="按条件列出 run")
    find.add_argument("--campaign")
    find.add_argument("--arm")
    find.add_argument("--net", help="路网名子串")
    find.add_argument("--status")
    find.add_argument("--kind", help="run_kind，如 train/eval/smoke")
    find.add_argument("--seed", type=int)
    find.add_argument("--limit", type=int, default=50)
    find.add_argument("--check-disk", action="store_true", help="同时检查本机是否在盘")
    find.set_defaults(func=cmd_find)

    check = sub.add_parser("check", help="提交闸门（git hook 调用）")
    mode = check.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true", help="检查暂存区（默认）")
    mode.add_argument("--all", action="store_true", help="全量结构检查（CI 用）")
    mode.add_argument("--commit-msg", metavar="FILE", help="检查提交信息文件")
    check.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
