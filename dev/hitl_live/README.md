# dev/hitl_live/ — 已升格（指路）

本目录全部内容已于 2026-10-13 升格至 **`hitl/`**：

| 原 | 现 |
|---|---|
| `dev/hitl_live/{package,segments,build_pairs,build_demos,filter_pool,dump_lane_topology,train_reward}.py` | `hitl/pipeline/` |
| `dev/hitl_live/build_*_plan.py`、`dispatch_r3.sh` | `hitl/plans/` |
| `dev/hitl_live/{paired_*,calibrate_*,rm_env_corr}.py` | `hitl/probe/` |
| `dev/hitl_live/{*_fig,*_table}.py` | `hitl/figs/` |
| `dev/hitl_live/topology_*.json` | `hitl/topology/` |
| `dev/action_ranking_audit/` | `hitl/audit/` |
| `tools/sumo_live_broadcast/` | `hitl/live/` |

详见 `hitl/README.md` 与 `dev/INDEX.md`。
