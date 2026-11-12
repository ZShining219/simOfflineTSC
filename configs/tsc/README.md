# configs/tsc 索引

> 由 `python -m lab configs --write` 生成，勿手改；`tests/test_architecture.py` 校验其与实际文件一致。

- **智能体默认配置**：run.py 按 `configs/tsc/<智能体名>.yml` 自动加载，必须留在本目录根部。
- **实验 overlay**：经 `--experiment-config` 显式传入。根目录的是历史存量（只许减少）；**新 overlay 一律放 `configs/tsc/<campaign>/`**。
- run 实际使用的配置以 run 目录 `config/` 下的快照为准（`python -m lab show <run_id>`）。

## 子目录（按 campaign）

| 目录 | yml 数 |
|---|---|
| `att_entity_004/` | 144 |
| `hz1x1_sanity_v1/` | 5 |
| `signal_control_v1/` | 2 |
| `xiasha1_online/` | 11 |
| `xiasha1_online_v2/` | 8 |
| `xiasha1_online_v3/` | 8 |

## 根目录文件

| 文件 | 类型 | 仍在引用它的代码/文档（至多 3 处） |
|---|---|---|
| `base.yml` | 智能体默认配置 | `common/paper_experiment.py`、`configs/tsc/colight.yml`、`configs/tsc/dqn.yml` 等 27 处 |
| `colight.yml` | 智能体默认配置 | `configs/tsc/colight_event_200.yml`、`configs/tsc/colight_event_smoke.yml`、`configs/tsc/colight_hz4x4_rl.yml` 等 10 处 |
| `colight_event150.yml` | 实验 overlay（存量） | `tests/test_paper_experiment.py`、`tests/test_paper_preflight.py`、`tools/README.md` 等 4 处 |
| `colight_event_200.yml` | 实验 overlay（存量） | 无 |
| `colight_event_extension500.yml` | 实验 overlay（存量） | 无 |
| `colight_event_random_200.yml` | 实验 overlay（存量） | `dev/att_entity_003/build_flx_stage2_queue.py` |
| `colight_event_smoke.yml` | 实验 overlay（存量） | 无 |
| `colight_hz4x4_rl.yml` | 实验 overlay（存量） | 无 |
| `colight_normal_200.yml` | 实验 overlay（存量） | 无 |
| `colight_strong150.yml` | 实验 overlay（存量） | `tests/test_strong_training.py` |
| `colight_strong150_recovery_s147.yml` | 实验 overlay（存量） | 无 |
| `colight_strong_calibration.yml` | 实验 overlay（存量） | `tests/test_strong_calibration.py` |
| `concat_colight.yml` | 智能体默认配置 | 无 |
| `concat_colight_text_event_200.yml` | 实验 overlay（存量） | 无 |
| `concat_colight_text_event_smoke.yml` | 实验 overlay（存量） | `configs/tsc/concat_colight_text_normal_smoke.yml` |
| `concat_colight_text_normal_200.yml` | 实验 overlay（存量） | 无 |
| `concat_colight_text_normal_smoke.yml` | 实验 overlay（存量） | 无 |
| `concat_counterfactual_eval.yml` | 实验 overlay（存量） | 无 |
| `dqn.yml` | 智能体默认配置 | `configs/tsc/xiasha1_online/dqn_full.yml`、`configs/tsc/xiasha1_online/dqn_slow.yml`、`dev/ledger/register_xiasha1_explore.py` 等 10 处 |
| `dqn_backup.yml` | 实验 overlay（存量） | `configs/tsc/tarl_attention.yml`、`configs/tsc/tarl_concat.yml`、`configs/tsc/tarl_crossq.yml` 等 8 处 |
| `dqn_gy5.yml` | 实验 overlay（存量） | `AGENTS.md` |
| `dqn_gy5_smoke.yml` | 实验 overlay（存量） | 无 |
| `dqn_smoke.yml` | 实验 overlay（存量） | `bench/golden/online_dqn_hz1x1.json` |
| `dqn_smoke_d3qn.yml` | 实验 overlay（存量） | 无 |
| `dqn_smoke_ddqn.yml` | 实验 overlay（存量） | 无 |
| `dqn_smoke_dueling.yml` | 实验 overlay（存量） | 无 |
| `eval_baseline_checkpoint.yml` | 实验 overlay（存量） | `dev/ledger/backfill_xiasha1_cross_eval.py`、`dev/ledger/gen_experiments.py`、`dev/ledger/register_xiasha1_cross_eval.py` 等 7 处 |
| `eval_baseline_checkpoint_d3qn.yml` | 实验 overlay（存量） | `dev/ledger/backfill_xiasha1_cross_eval.py`、`dev/ledger/register_xiasha1_cross_eval.py`、`dev/ledger/register_xiasha1_v3_cross_eval.py` 等 5 处 |
| `eval_baseline_checkpoint_ddqn.yml` | 实验 overlay（存量） | `dev/ledger/backfill_xiasha1_cross_eval.py`、`dev/ledger/register_xiasha1_cross_eval.py`、`dev/ledger/register_xiasha1_v3_cross_eval.py` 等 5 处 |
| `eval_baseline_checkpoint_dueling.yml` | 实验 overlay（存量） | `dev/ledger/backfill_xiasha1_cross_eval.py`、`dev/ledger/register_xiasha1_cross_eval.py`、`dev/ledger/register_xiasha1_v3_cross_eval.py` 等 5 处 |
| `eval_baseline_checkpoint_gy5.yml` | 实验 overlay（存量） | `tools/eval_baseline_checkpoint.py` |
| `eval_baseline_checkpoint_ts.yml` | 实验 overlay（存量） | `configs/benchmarks/xiasha1_v2_bestckpt_ts.yaml` |
| `eval_baseline_checkpoint_ts_d3qn.yml` | 实验 overlay（存量） | `configs/benchmarks/xiasha1_v2_bestckpt_ts.yaml` |
| `eval_baseline_checkpoint_ts_ddqn.yml` | 实验 overlay（存量） | `configs/benchmarks/xiasha1_v2_bestckpt_ts.yaml` |
| `eval_baseline_checkpoint_ts_dueling.yml` | 实验 overlay（存量） | `configs/benchmarks/xiasha1_v2_bestckpt_ts.yaml` |
| `fixedtime.yml` | 智能体默认配置 | 无 |
| `frap.yml` | 智能体默认配置 | `README.md`、`configs/tsc/paper_frap.yml` |
| `frap_smoke.yml` | 实验 overlay（存量） | 无 |
| `hz4x4_paper_smoke.yml` | 实验 overlay（存量） | `tests/test_paper_agents_sumo.py`、`tools/README.md` |
| `maddpg.yml` | 智能体默认配置 | 无 |
| `maddpg_v2.yml` | 智能体默认配置 | 无 |
| `magd.yml` | 智能体默认配置 | 无 |
| `maxpressure.yml` | 智能体默认配置 | `docs/plan1_support_and_pilot.md` |
| `mplight.yml` | 智能体默认配置 | `README.md`、`configs/tsc/mplight_event_smoke.yml`、`configs/tsc/mplight_hz4x4_rl.yml` 等 5 处 |
| `mplight_event_random_200.yml` | 实验 overlay（存量） | 无 |
| `mplight_event_smoke.yml` | 实验 overlay（存量） | 无 |
| `mplight_hz4x4_rl.yml` | 实验 overlay（存量） | `dev/if_mask_eval/eval_mplight_mask.py` |
| `mplight_hz4x4_rl_smoke.yml` | 实验 overlay（存量） | 无 |
| `mplight_smoke.yml` | 实验 overlay（存量） | 无 |
| `paper_colight.yml` | 智能体默认配置 | 无 |
| `paper_experiment_base.yml` | 实验 overlay（存量） | `common/paper_experiment.py`、`tools/README.md`、`world/sumo_events/README.md` |
| `paper_experiment_smoke.yml` | 实验 overlay（存量） | `tests/test_paper_experiment.py`、`tests/test_paper_preflight.py`、`tools/README.md` |
| `paper_frap.yml` | 智能体默认配置 | 无 |
| `paper_monitor_smoke.yml` | 实验 overlay（存量） | `tools/README.md`、`tools/run_paper_baseline.py` |
| `paper_p0.yml` | 实验 overlay（存量） | `tests/test_paper_p0.py`、`tools/README.md` |
| `paper_presslight.yml` | 智能体默认配置 | 无 |
| `ppo.yml` | 智能体默认配置 | 无 |
| `ppo_pfrl.yml` | 智能体默认配置 | 无 |
| `presslight.yml` | 智能体默认配置 | `configs/tsc/paper_presslight.yml` |
| `presslight_smoke.yml` | 实验 overlay（存量） | 无 |
| `sac.yml` | 智能体默认配置 | 无 |
| `sac_smoke.yml` | 实验 overlay（存量） | 无 |
| `sga_colight.yml` | 智能体默认配置 | `configs/tsc/sga_colight_text_event_smoke.yml` |
| `sga_colight_event_smoke.yml` | 实验 overlay（存量） | 无 |
| `sga_colight_text_event_150.yml` | 实验 overlay（存量） | 无 |
| `sga_colight_text_event_150_module.yml` | 实验 overlay（存量） | 无 |
| `sga_colight_text_event_200.yml` | 实验 overlay（存量） | 无 |
| `sga_colight_text_event_random_200.yml` | 实验 overlay（存量） | `dev/att_entity_003/build_flx_stage2_queue.py` |
| `sga_colight_text_event_smoke.yml` | 实验 overlay（存量） | `configs/tsc/concat_colight_text_event_200.yml`、`configs/tsc/concat_colight_text_event_smoke.yml`、`configs/tsc/sga_colight_text_event_150.yml` 等 7 处 |
| `sga_colight_text_normal_200.yml` | 实验 overlay（存量） | 无 |
| `sga_colight_text_normal_smoke.yml` | 实验 overlay（存量） | 无 |
| `sga_counterfactual_eval.yml` | 实验 overlay（存量） | 无 |
| `sga_counterfactual_eval_window.yml` | 实验 overlay（存量） | 无 |
| `sga_event_evalfix_smoke.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_align_probe.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_align_trans_meta_probe.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_align_trans_probe.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_cf_probe.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight.yml` | 智能体默认配置 | `configs/tsc/sga_flx_colight_text_event_align_smoke.yml`、`configs/tsc/sga_flx_colight_text_event_align_trans_meta_smoke.yml`、`configs/tsc/sga_flx_colight_text_event_align_trans_smoke.yml` 等 4 处 |
| `sga_flx_colight_text_event_200.yml` | 实验 overlay（存量） | `dev/att_entity_003/build_flx_stage2_queue.py` |
| `sga_flx_colight_text_event_align_smoke.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_align_trans_meta_smoke.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_align_trans_smoke.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_diag60.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_livefire.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_random_200.yml` | 实验 overlay（存量） | `dev/att_entity_003/build_flx_stage2_queue.py` |
| `sga_flx_colight_text_event_random_align_l0p1.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_random_align_l1p0.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_random_align_trans_l0p1.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_random_align_trans_l1p0.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_random_align_trans_meta_l0p1.yml` | 实验 overlay（存量） | 无 |
| `sga_flx_colight_text_event_smoke.yml` | 实验 overlay（存量） | `configs/tsc/sga_flx_colight_text_event_200.yml` |
| `sga_flx_colight_text_normal_200.yml` | 实验 overlay（存量） | `dev/att_entity_003/build_flx_stage2_queue.py` |
| `sga_mplight.yml` | 智能体默认配置 | `configs/tsc/sga_mplight_event_smoke.yml` |
| `sga_mplight_event_smoke.yml` | 实验 overlay（存量） | 无 |
| `shared_dqn.yml` | 智能体默认配置 | `docs/arterial_1x6_handoff.md` |
| `sotl.yml` | 智能体默认配置 | 无 |
| `tarl_attention.yml` | 智能体默认配置 | 无 |
| `tarl_concat.yml` | 智能体默认配置 | 无 |
| `tarl_crossq.yml` | 智能体默认配置 | 无 |
| `tarl_gat.yml` | 智能体默认配置 | 无 |
| `tarl_gating.yml` | 智能体默认配置 | 无 |
| `tarl_paper_attention_rande_200.yml` | 实验 overlay（存量） | 无 |
| `tarl_paper_attention_smoke.yml` | 实验 overlay（存量） | `dev/att_entity_003/run_tarl_text_cf.py` |
| `tarl_paper_cfprobe.yml` | 实验 overlay（存量） | `dev/att_entity_003/run_tarl_text_utility.py`、`dev/att_entity_003/watch_stage5_eval.py`、`tools/run_tarl_repr_probe.py` |
| `tarl_paper_gat_rande_200.yml` | 实验 overlay（存量） | 无 |
| `tarl_paper_gating_rande_200.yml` | 实验 overlay（存量） | 无 |
| `tarl_paper_sensor_rande_200.yml` | 实验 overlay（存量） | 无 |
| `tarl_selfattn.yml` | 智能体默认配置 | 无 |
| `tarl_sensor.yml` | 智能体默认配置 | 无 |
| `xiasha1_timeseries.yml` | 实验 overlay（存量） | `dev/demand_packs/build_xiasha_compound_probe.py`、`dev/ledger/register_xiasha1_compound_probe.py`、`tools/pack_benchmark/README.md` 等 4 处 |
