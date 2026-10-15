"""``python -m hitl`` — 通道包门面：打印面表。各面用法见 hitl/README.md。"""

_FACES = (
    ('serve', 'python -m hitl.serve', '实时转播+专家席+问询门+会话录制（常驻服务）'),
    ('pipeline', 'python -m hitl.pipeline <stage>', '数据面 ETL：package/segments/build_pairs/build_demos/filter_pool/train_reward/dump_lane_topology'),
    ('plans', 'python -m hitl.plans build --round <r>', '正式 plan 生成器（轮次参数化）'),
    ('probe', 'python -m hitl.probe.<x>', '配对探针与归一化校准'),
    ('audit', 'python -m hitl.audit.<x>', "B′判官审计件"),
)


def main():
    print('hitl 通道包面表（训练态走 sequential，见 README）：')
    for name, usage, desc in _FACES:
        print(f'  {name:<8} {usage:<46} {desc}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
