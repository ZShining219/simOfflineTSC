"""sig 事件训练常量——本包唯一常量源（原散落于 _legacy/router/train_router.py）。

事件采样空间：sumohz1x1 走廊路网 4 条可训练 inbound 边及其
target/feeder/discharge/distractor 拓扑映射、周期/时长/时刻网格。
活代码一律从本模块取常量；_legacy/router/ 内的同名定义为历史快照，
冻结不动、不作事实源。
"""

TRAIN_EDGES = ['road_0_1_0', 'road_1_0_1', 'road_2_1_2', 'road_1_2_3']
EDGE_TARGETS = {
    'road_0_1_0': 'road_1_1_0', 'road_1_0_1': 'road_1_1_2',
    'road_2_1_2': 'road_1_2_1', 'road_1_2_3': 'road_1_3_2',
}
# Upstream feeder of each trainable inbound edge on the sumohz1x1
# corridor net — parsed from the net's lane connections
# (dispatch_state.FeatureBuilder._coord_topology derives the same map
# live; kept here for inject sampling where no world exists yet).
FEEDER_OF = {
    'road_0_1_0': 'road_1_1_2', 'road_1_0_1': 'road_1_1_3',
    'road_2_1_2': 'road_1_1_0', 'road_1_2_3': 'road_1_1_1',
}
# First (of two) discharge links each inbound edge drains into —
# the corridor's downstream leg, used to place downstream-bottleneck
# pressure for corridor-load scenarios.
DISCHARGE_OF = {
    'road_0_1_0': 'road_1_1_0', 'road_1_0_1': 'road_1_1_1',
    'road_2_1_2': 'road_1_1_2', 'road_1_2_3': 'road_1_1_3',
}
# One inbound edge OUTSIDE each event edge's 2-hop watch set —
# used to inject a physically real but report-irrelevant distractor
# surge that tests report-vs-salience competition.
DISTRACTOR_OF = {
    'road_0_1_0': 'road_1_2_3', 'road_1_0_1': 'road_0_1_0',
    'road_2_1_2': 'road_1_0_1', 'road_2_1_2': 'road_2_1_2',
}
TRAIN_PERIODS = [2, 4]
TRAIN_DURS = [300, 600, 900]
TRAIN_ONSETS = [300, 600, 900, 1200]
