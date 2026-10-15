"""``python -m scene_attention`` — 方法包门面：打印面表。详见 README.md。"""

_FACES = (
    ('train', 'python -m scene_attention.train', 'BC 蒸馏→原装 reward RL 微调（sig9→sig15 链）'),
    ('eval', 'python -m scene_attention.eval', '事件场景冻结评测（eval_episode/spec_list）'),
    ('probe', 'python -m scene_attention.probe.<x>', '探针：takeover(接管)/attn_text(文本门)/dispatch_eval(调度门)'),
    ('figs', 'scene_attention/figs/*.py', '出图件（产物件，非入口）'),
    ('events', 'scene_attention/events.py', 'sig 事件常量唯一源'),
)


def main():
    print('scene_attention 方法包面表：')
    for name, usage, desc in _FACES:
        print(f'  {name:<7} {usage:<46} {desc}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
