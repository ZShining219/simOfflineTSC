# dev/ledger — 台账构建器（收仓件）

来源：治理工作区 govern_1007/scratch/ 一次性盘点脚本，2026-10-07 裁决收仓长期维护。

- inventory.py   扫主树 artifacts/+data/output_data/ → inventory.json（盘点快照）
- build_ledger.py inventory.json → ledger/runs.jsonl（分类/状态映射/别名/隔离桶语义）
- gen_experiments.py runs.jsonl → EXPERIMENTS.md（人读台账；META 段含手维护结论，改动前先读）

注意：脚本内含主树/治理树绝对路径常量；台账追加/更正的常规操作是改 runs.jsonl 后重跑
gen_experiments.py，inventory 级重建仅在全量盘点时需要。
