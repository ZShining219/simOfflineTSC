# 场景定义与覆盖率口径（scenario coverage v1）

> 目的：把用户对「场景」的主观定义落成可计算对象，供后续所有覆盖率类实验复用。
> 本文件是口径来源；具体场景集在实验卡里冻结，跑后不许改。

## 1. 场景 = 一个可指定的仿真单元

场景 c 用五轴元组定义：

```text
c = (network, demand, event, protocol, seed_bundle)
```

- `network`：路网拓扑，取 `configs/sim/<name>.cfg` 的 `network` 字段（现成：hz1x1 / hz4x4 / xiasha1 / arterial1x6 …）
- `demand`：需求体制。同一路网下不同 flow 文件，可再细分四类旋钮：
  流量水平（sparse / dense）、转向结构（左转占比、直行主导）、方向结构（东西/南北失衡、单臂集中）、时变形态（脉冲、爬坡、正弦波动）
- `event`：事件，取 `configs/events/plans/<plan>.yml`，kind ∈ `lane_blockage` / `road_closure` / `global_rain`，带 begin/end 与事件窗（现行口径 900–1200s）
- `protocol`：仿真时长与决策间隔（steps=3600、action_interval=10）与评估 sumo seed 集合
- `seed_bundle`：评估侧 sumo seeds（评估随机性来源，不入场景身份）

**「同场景」= 前四项一致；「不同场景」= 至少一项不同。**

用户原话「场景不一定指一个具体仿真包，可以在仿真包基础上再细粒度划分」——本仓已在单一 hz1x1 仿真包内备有 15 个语义标注需求包与 19 个需求 cfg（config2/3/4、sparse_low/ultra、dense_high/extreme、lt_high/extreme/wsing、asym_ns/we/wonly、st_dom、fluct_burst/ramp/recede/sin），可直接切到 8~16 个细粒度场景，**不需要新造仿真包**。

## 2. 场景簇与覆盖率

- 场景空间 `S = {c_1..c_K}`，**跑之前冻结**，写进实验卡，跑后不许增删改。
- 控制器 π 在 c 上的三指标向量 `m(π,c) = (travel_time, throughput, unfinished_vehicles)`（GOVERNANCE §5.1 强制同报）。
- 参考基线 `b(c)`：本卡取 maxpressure 在 c 上的冻结评估三指标。
- 单场景覆盖判定：

```text
cov(π,c) = 1[ ATT(π,c) ≤ (1+δ)·ATT_b(c) 且 unfinished(π,c) ≤ (1+η)·unfinished_b(c) ]
δ = 0.05，η = 0.10   （事先冻结）
```

- **二值覆盖率**：`Coverage(π) = (1/K) Σ_k cov(π,c_k)`，取值 [0,1]
- **覆盖裕度**（防二值饱和）：`Margin(π) = (1/K) Σ_k clip((τ_k − ATT(π,c_k))/τ_k, −1, +1)`，`τ_k = (1+δ)·ATT_b(c_k)`
- 分组报告：L1 训练见过 / L2 训练未见 / L3 事件场景（本卡只跑 L1+L2）

## 3. 为什么不用均值

均值会被「在 1 个场景很强、在 3 个场景崩」掩盖；覆盖率直接数「能用的场景类型个数」，正是用户问的东西。但二值覆盖率会饱和（全 1 或全 0），所以**二值覆盖率与覆盖裕度必须同时报**。

## 4. 口径纪律

1. 三指标强制同报；只报 travel_time 不算覆盖
2. 同条件对比：同场景集、同 train seeds、同 eval sumo seeds、同 episode 预算；**唯一变量是被检验机制**
3. 阈值 τ 事先冻结；跑完调 τ 就是新口径，须按 GOVERNANCE §5 重开 campaign 并重打基线
4. 允许负结果：半离线不更高就如实写「不支持」，不许挑 seed / 挑 cell / 换阈值重算
