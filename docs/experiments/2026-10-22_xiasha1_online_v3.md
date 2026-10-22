# xiasha1_online_v3（几何修复版重跑）

> 状态：进行中
> 发起日期：2026-10-22　｜　负责人/Agent：Devin
> 最近更新：2026-10-22

## 关联

- campaign：`xiasha1_online_v3`（台账 288 行 REGISTERED）
- run_id / 队列文件：`xi3re_<arm>_<pack>_s<seed>` ；
  `artifacts/xiasha1_online_v3/run_queue.json`
- code_commit：本次修复提交（见 ledger code_commit）
- 产物目录：`data/output_data/tsc/sumo_<agent>/xiasha1_<pack>/xi3re_*`
- TB 投影：`artifacts/xiasha1_online_v3/tb`（127.0.0.1:6006）

## 1. 改动意图

xiasha1 全部历史实验（v1/explore/v2/cross_eval/bestckpt_ts/compound/
offline，台账 1246 行已标 `errata=geometry_truncated_v1v2`）在**截断
几何**上运行：进口车道仅 106.4m，`departPos≈80` 使车辆在距停车线
~22-27m 处以 10m/s 凭空插入。该缺陷源自
`tools/xiasha_sumo/converter.py` 的相机视野重建语义，并经
`build_xiasha_demands.py` 传播到全部 14 包。

后果：每车道可观测排队上限 ~3 辆；queue 分位、waiting_time 长尾、
uninserted、TT 绝对值全部失真——v2 期 D3QN 问题挖掘中
"N2 常驻 2 辆"正是这一截断的指纹。v3 是**修复性质**的正式重跑，
在修正几何上复现 v2 全量设计，重建可比的正式对比口径。

## 2. 改动模块

仅数据 + 一处观测上限代码 + 生成器/守卫：

- 16 个 `*_signal.net.xml`（14 包 + `xiasha1_sumo_signal` +
  `xiasha1_sumo_lane_aligned_base`）：进口车道 106.40→280.00，
  shape 外延端点同步平移，junction/location 边界一致化；
- 15 个 `*.rou.xml`：删除全部 `departPos`（车道起点插入，
  `departSpeed=10.00` 保留）；
- `world/world_sumo.py`：`max_distance` 由硬编码 200 改为
  `max(lane.getLength)` 动态取值（防同类截断复发，hz 路网一并受益）；
- `dev/demand_packs/build_xiasha_demands.py`：`departPos` 输出改 `0.00`；
- 新增守卫 `dev/checks/check_xiasha1_geometry.py`（16 net ×280 +
  15 rou 无 departPos + max_distance 动态，已跑通）。

## 3. 实验设计

与 `xiasha1_online_v2` 逐位一致（修复前后唯一变量是几何）：

- 场景：xiasha1 ×12 需求包；SUMO/libsumo，legacy `pseudo_step`；
- 臂：dqn / ddqn / dueling / d3qn / presslight / frap / mplight / sac；
- seed 0/1/2 × 200 episodes × 3600s，action interval 10s；
- overlay：`configs/tsc/xiasha1_online_v3/<arm>.yml`（=v2 overlay，
  ε 1.0→0.05 decay 0.99995，learning_start 5000，全 episode 贪心评估，
  resumable ckpt {0,25,50,100,150,200}）；
- 指标口径：travel_time / throughput / unfinished_vehicles 三指标 +
  queue、waiting_time；**v2 与 v3 指标不可直接比较**（几何变更）。

## 4. 实验结果

进行中（队列 288 run 派发后回填）。冒烟证据：

- `dev/checks/check_xiasha1_geometry.py`：16 nets ×280、15 rous
  起点插入、max_distance 动态 —— OK；
- libsumo 加载 `xiasha1_asym_nonly_1h`：车道 280m，车辆 pos≈0 插入；
- `xi3_smoke_fixedtime`（1 ep 全周期）与 `xi3smk_d3qn`
  （smoke overlay，train+test）：均正常完成，queue ~52-62
  （修复前物理上限 ~3，真实排队现已可观测）。

## 5. 结论与后续

待队列完成：收割 final-20 三指标，与 v2 做臂间相对结构对照
（饥饿/抖动/长尾机制是否随几何复原而变化），重建正式排名口径。
