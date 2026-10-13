# dev/takeover_probe/ — 已升格至 scene_attention/

本目录内容已于 2026-10-13 升格为方法包 `scene_attention/`（GOVERNANCE §3.1
统一包形状），历史由 git rename 保留。

- 主线入口：`python -m scene_attention.train` / `python -m scene_attention.eval`
- 方法卡与面表：`scene_attention/README.md`
- router 否决代际：`scene_attention/_legacy/router/`
- 产物桶沿用 `output_data/dev_takeover_probe/`（证据路径不改名）

本目录仅作指路，不再存放代码；新工作直接进 `scene_attention/`。
