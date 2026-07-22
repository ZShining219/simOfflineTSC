# 功能 2：运行身份与状态验证

- 日期：2026-07-22
- 自动检查：`python -m unittest discover -s tests -p 'test_milestone0_*.py' -v`，6 项通过；覆盖完成、失败、非法状态、非法转换和环境值脱敏。
- 完成 smoke：FixedTime / `sumohz1x1` / training_seed 7 / 1 episode / 100 simulation steps；命令 `SUMO_HOME=/home/dev/miniforge3/envs/colight/lib/python3.10/site-packages/sumo /home/dev/miniforge3/envs/colight/bin/python3.10 run.py -w sumo -a fixedtime -n sumohz1x1 --prefix m0f2_complete_20260722 --seed 7 --interface libsumo`；退出码 0；travel time 79.5、reward -27.165、queue 10、delay 1.9417、throughput 4。
- 完成产物：`data/output_data/tsc/sumo_fixedtime/sumohz1x1/m0f2_complete_20260722/run_manifest.json` 与 `run_status.json`；状态为“已完成”，exit_code=0，开始和结束时间非空。
- 失败 smoke：同一 agent/network/seed，通过临时内存任务对象抛出 `RuntimeError('controlled failure marker')`；进程退出码 1。
- 失败产物：`data/output_data/tsc/sumo_fixedtime/sumohz1x1/m0f2_failure_20260722/run_status.json`；状态为“失败”，exit_code=1，error_type=`RuntimeError`，错误信息为受控标记。
- 字段校验：run_id、task、agent、world、network、prefix、training_seed、sumo_seed_mode、baseline_commit、created/started/finished、status、exit_code、error_type、error_message、config_hash 均由两文件提供。
- 最终审计补充：目录已独占创建但 config archive 尚未形成时若初始化失败，仍原子写 manifest/status；该失败 manifest 的 `config_hash=null` 表示 resolved_config 尚不存在，status 为“失败”、started_at_utc=null、非零退出码。重复 prefix 在目录预留前失败，不会改写既有运行。
- 清理校验：临时 FixedTime YAML 已恢复；`configs/sim/sumohz1x1.cfg` SHA-256 仍为 `314f1915c21344269dfb096f71d48741b5d16205b49b0c242a0f962796bb9dbd`。
- 行为边界：仅增加运行身份和状态生命周期，不改变算法或仿真数值语义。
