# xiasha1_online_v3（几何修复版重跑）

> 状态：已完成
> 发起日期：2026-10-22　｜　负责人/Agent：Devin
> 最近更新：2026-10-22

## 关联

- campaign：`xiasha1_online_v3`（台账 288 行 REGISTERED）
- run_id / 队列文件：`xi3re_<arm>_<pack>_s<seed>` ；
  `artifacts/xiasha1_online_v3/run_queue.json`
- code_commit：本次修复提交（见 ledger code_commit）
- 产物目录：`data/output_data/tsc/sumo_<agent>/xiasha1_<pack>/xi3re_*`
- TB 投影：`artifacts/xiasha1_online_v3/tb`（127.0.0.1:6008）

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

288/288 DONE（32 并行 ~11.5h），台账已回填。臂级 final-20 均值：

| arm | TT | throughput | unfinished | queue | waiting |
|---|---:|---:|---:|---:|---:|
| ddqn | 39.12 | 1638.5 | 16.9 | 5.48 | 5.4 |
| d3qn | 39.57 | 1637.9 | 17.4 | 5.83 | 5.7 |
| sac | 39.69 | 1637.6 | 17.7 | 5.89 | 8.4 |
| dueling | 40.01 | 1638.3 | 17.0 | 5.86 | 6.0 |
| dqn | 40.35 | 1638.0 | 17.2 | 6.02 | 5.2 |
| frap | 42.07 | 1637.0 | 18.4 | 6.86 | 7.4 |
| mplight | 42.17 | 1636.4 | 18.4 | 7.02 | 7.8 |
| presslight | 42.50 | 1637.3 | 17.6 | 6.97 | 7.2 |

关键发现：**修复几何后 uninserted 机制基本消失**——各臂 throughput
≈1637（包均需求 ~1655，unfinished ~17），隐含 uninserted ≈1，
而 v2 同口径为 42–151（PressLight 最差）。v2 期"D3QN 靠饿死
左转换 TT"的叙事中，**uninserted 通道主要是截断几何伪影**；
饥饿机制是否仍在网内以排队/等待形式存在，需基于 v3 per-lane/
per-second 数据重新挖掘（v2 的 N2"常驻 2 辆"观测本身即截断产物）。

冒烟证据（修复当时）：几何守卫 OK；libsumo 280m+pos≈0 插入；
fixedtime/D3QN 冒烟通过，queue ~52-62（截断期上限 ~3）。

## 5. 结论与后续

修复口径建立：**v3 起 xiasha1 指标为正式口径，v1/v2 仅作机制参考**
（相对臂间结论在同几何内有效，绝对值不可比）。下一步：
- 基于 v3 数据重跑 D3QN 问题挖掘（长尾/饥饿/抖动是否在真实几何上
  复现），重点 asym_nonly 相位3 分配与 waiting 长尾；
- cross_eval / bestckpt_ts 若需正式口径，按 v3 checkpoint 重评。
