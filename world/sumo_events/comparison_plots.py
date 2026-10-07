"""Standalone PNG/PDF evidence figures from compare.py's saved measurements."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sumolib

from .compare import CONDITIONS, LABELS


COLORS = {'normal': '#44566C', 'lane_blockage': '#D58A17', 'road_closure': '#CA5357',
          'global_rain': '#2977B8', 'combined': '#7956A5'}


def load_timeline(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def render(output):
    output = Path(output)
    summary = json.loads((output / 'summary.json').read_text())
    manifest = json.loads((output / 'manifest.json').read_text())
    # Plot styling may change without repeating simulations. The collector,
    # event operators, inputs and controller must still match the experiment.
    for file, expected in manifest['source_sha256'].items():
        if Path(file).name != 'comparison_plots.py' and hashlib.sha256(Path(file).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Experiment source/input changed: {file}')
    seeds = summary['seeds']
    begin, end = summary['event_window']
    timelines = {(seed, condition): load_timeline(output / f'seed_{seed}' / condition / 'timeline.csv')
                 for seed in seeds for condition in CONDITIONS}
    events = {e['kind']: e for e in manifest['schedule']['events']}
    probe_path = output / 'boundary_probe.json'
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    changes = probe.get('first_intersection_action_changes', [])
    rapid_switching = len(changes) > 3 and all(b[0] - a[0] == 1 for a, b in zip(changes[1:], changes[2:]))
    controller_note = ('控制边界：现有 FixedTimeAgent 在前 20 秒后每秒换相；此处仅作相同动作下的事件机制对照。'
                       if rapid_switching else '当前结果用于所记录控制器下的事件机制对照，不代表 NLP 训练收益。')
    fig_dir = output / 'figures'
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'figure.facecolor': 'white', 'axes.titlesize': 12})
    exported = []

    def save(fig, name):
        for extension in ('png', 'pdf'):
            path = fig_dir / f'{name}.{extension}'
            fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
            exported.append(str(path.relative_to(output)))
        plt.close(fig)

    def event_span(ax):
        ax.axvspan(begin, end, color='#E8D7B5', alpha=.35, label='事件作用区间')
        ax.axvline(begin, color='#AB9366', lw=.7, ls=':')
        ax.axvline(end, color='#AB9366', lw=.7, ls=':')
        ax.grid(alpha=.15)

    # 1: spatial scope + independently queried physical/report states.
    fig = plt.figure(figsize=(13.5, 8.4), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=(1, 1.35))
    ax = fig.add_subplot(grid[:, 0])
    config = json.loads(Path(manifest['settings']['sim_config']).read_text())
    net = sumolib.net.readNet(str(Path(config['dir']) / config['roadnetFile']))
    for edge in net.getEdges():
        points = np.array(edge.getShape())
        ax.plot(points[:, 0], points[:, 1], color='#CADAE7', lw=2, zorder=1)
    closure = net.getEdge(events['road_closure']['edge_id'])
    points = np.array(closure.getShape())
    ax.plot(points[:, 0], points[:, 1], color=COLORS['road_closure'], lw=5, zorder=3)
    lane = net.getLane(events['lane_blockage']['lane_id'])
    point = sumolib.geomhelper.positionAtShapeOffset(lane.getShape(), events['lane_blockage']['position'])
    ax.scatter(*point, marker='X', s=160, color=COLORS['lane_blockage'], edgecolor='white', zorder=5)
    for tls in net.getTrafficLights():
        node = net.getNode(tls.getID())
        x, y = node.getCoord()
        ax.scatter(x, y, s=15, color='#44566C', zorder=4)
        ax.annotate(tls.getID().replace('intersection_', ''), (x, y), xytext=(4, 6), textcoords='offset points', fontsize=8)
    ax.annotate('局部阻塞：指定车道上的一点', point, xytext=(0, -65), textcoords='offset points',
                ha='center', color=COLORS['lane_blockage'], arrowprops={'arrowstyle': '->', 'color': COLORS['lane_blockage']})
    center = points.mean(axis=0)
    ax.annotate('整段封闭：一个方向、全部 3 条车道', center, xytext=(5, 65), textcoords='offset points',
                ha='center', color=COLORS['road_closure'], arrowprops={'arrowstyle': '->', 'color': COLORS['road_closure']})
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('hz4×4：局部目标与全网范围\n浅蓝路网均受降雨降速影响')
    data = timelines[(seeds[0], 'combined')]
    settings = [
        ('obstacle_count', data['block_report_active'], '障碍车辆数', [0, 1], '局部车道阻塞'),
        ('closed_lane_count', data['closure_report_active'] * 3, '禁行车道数', [0, 3], '整段道路封闭'),
        ('rain_speed_ratio', np.where(data['rain_report_active'] > 0, events['global_rain']['speed_factor'], 1),
         '原限速的比例', [.7, 1], '全局降雨'),
    ]
    for index, (key, reported, ylabel, ticks, title) in enumerate(settings):
        ax = fig.add_subplot(grid[index, 1])
        event_span(ax)
        ax.step(data['time'], data[key], where='post', color='#2F7D68', lw=3, label='从 SUMO 查询的物理状态')
        ax.step(data['time'], reported, where='post', color='#523B8E', lw=1.5, ls='--', label='报告中的事件状态')
        ax.set(xlim=(begin - 90, end + 150), yticks=ticks, ylabel=ylabel, title=title)
        ax.margins(y=.25)
        ax.text(.98, .85, f"最大报告滞后：{summary['max_report_lag_s']:g} 秒", transform=ax.transAxes,
                ha='right', fontsize=9, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85})
        if index == 0:
            ax.legend(loc='lower right', fontsize=8)
        if index == 2:
            ax.set_xlabel('仿真时间（秒）')
    fig.suptitle('事件执行与文本报告是否一致？\n位置不同、物理作用不同；报告随实际状态同步更新', fontsize=16)
    fig.supxlabel('封路曲线表示禁行权限；已进入路口的车辆可能继续完成穿越。降雨覆盖车道也包含路口内部连接。', fontsize=9)
    save(fig, '01_scope_and_alignment')

    # 2: targeted traffic effects and seed-paired differences, 30 s bins.
    def binned(data, metric):
        count = len(data['time']) // 30
        if metric == 'mean_speed_mps':
            speeds = data['sum_speed_mps'][:count * 30].reshape(count, 30).sum(1)
            vehicles = data['vehicles'][:count * 30].reshape(count, 30).sum(1)
            return np.divide(speeds, vehicles, out=np.zeros_like(speeds), where=vehicles > 0)
        return data[metric][:count * 30].reshape(count, 30).mean(1)

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 10), layout='constrained')
    panels = [('lane_blockage', 'target_lane_queue', '目标车道排队车辆数（辆）'),
              ('road_closure', 'closure_area_queue', '封闭道路及上游区域排队数（辆）'),
              ('global_rain', 'mean_speed_mps', '全网车辆平均速度（米/秒）')]
    for index, (condition, metric, ylabel) in enumerate(panels):
        normal = np.array([binned(timelines[(seed, 'normal')], metric) for seed in seeds])
        event = np.array([binned(timelines[(seed, condition)], metric) for seed in seeds])
        times = np.arange(1, normal.shape[1] + 1) * 30
        left, right = axes[index]
        for series, name in ((normal, 'normal'), (event, condition)):
            left.plot(times, series.mean(0), label=LABELS[name], color=COLORS[name], lw=1.8)
            left.fill_between(times, series.min(0), series.max(0), color=COLORS[name], alpha=.12)
        difference = event - normal
        right.plot(times, difference.mean(0), color=COLORS[condition], lw=1.8)
        right.fill_between(times, difference.min(0), difference.max(0), color=COLORS[condition], alpha=.18)
        right.axhline(0, color='#55616D', lw=.8)
        left.set_title(LABELS[condition] + '：与无事件对照')
        right.set_title('同种子差值：事件 − 无事件')
        for ax in (left, right):
            event_span(ax)
            ax.set(xlim=(max(0, begin - 300), min(end + 600, manifest['settings']['seconds'])), ylabel=ylabel)
        left.legend(loc='upper right', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('仿真时间（秒）')
    fig.suptitle(f'交通是否受到实际影响？\n{len(seeds)} 个种子均值；阴影为种子范围；每点为 30 秒统计', fontsize=16)
    fig.supxlabel(controller_note, fontsize=9)
    save(fig, '02_traffic_response')

    # 3: full quantitative comparison with individual seeds visible.
    metrics = {(r['condition'], r['metric']): r for r in summary['metrics']}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), layout='constrained')
    panels = [('event_queue', '事件期间平均排队数变化（辆）'),
              ('completed_trip_time', '已完成车辆平均行程时间变化（秒）'),
              ('arrived', f'截至 {manifest["settings"]["seconds"]} 秒的完成车辆数变化（辆）')]
    labels = [LABELS[c] for c in CONDITIONS[1:]]
    for ax, (metric, label) in zip(axes, panels):
        for index, condition in enumerate(CONDITIONS[1:]):
            row = metrics[(condition, metric)]
            mean = row['paired_delta_mean']
            ax.barh(index, mean, height=.5, color=COLORS[condition], alpha=.45)
            ax.errorbar(mean, index, xerr=[[mean - row['paired_delta_min']], [row['paired_delta_max'] - mean]],
                        color=COLORS[condition], capsize=4, fmt='none')
            offsets = np.linspace(-.14, .14, len(row['paired_deltas']))
            ax.scatter(row['paired_deltas'], index + offsets, color=COLORS[condition], s=22, zorder=4)
            ax.annotate(f'{mean:+.2f}', (mean, index), xytext=(4 if mean >= 0 else -4, 11),
                        textcoords='offset points', ha='left' if mean >= 0 else 'right', fontsize=9)
        ax.axvline(0, color='#55616D', lw=.8)
        ax.set(yticks=range(4), yticklabels=labels, xlabel=label)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.15)
        ax.margins(x=.3)
    fig.suptitle('单事件与组合事件：相对于同种子无事件条件的变化\n每个圆点为一个种子，横线为种子范围；不是置信区间', fontsize=15)
    fig.supxlabel('控制器不读取报告；行程时间仅统计已完成车辆，须结合完成数量解释。\n' + controller_note, fontsize=9)
    save(fig, '03_paired_metrics')

    # 4: compact absolute table for readers who need values, not only curves.
    columns = ['条件', '事件期排队\n（辆）', '事件期均速\n（米/秒）', '完成车辆\n（辆）',
               '路网内未到达\n（辆）', '已完成行程\n（秒）']
    table_rows = [[LABELS[c]] + [f"{metrics[(c, k)]['mean']:.2f}" for k in
                  ('event_queue', 'event_speed', 'arrived', 'running', 'completed_trip_time')]
                  for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(12.5, 4.3), layout='constrained')
    ax.axis('off')
    table = ax.table(cellText=table_rows, colLabels=columns, cellLoc='center', loc='center',
                     colWidths=[.20, .15, .15, .15, .15, .20])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#D5DEE7')
        if row == 0:
            cell.set_facecolor('#34495F'); cell.get_text().set_color('white')
        elif row % 2:
            cell.set_facecolor('#F1F5F8')
    fig.suptitle(f'hz4×4 量化对照：{len(seeds)} 个种子的算术均值\n事件 {begin:g}–{end:g} 秒；每轮 {manifest["settings"]["seconds"]} 秒；原项目 FixedTimeAgent', fontsize=15)
    fig.supxlabel(f'同一车流、同一种子配对、相同信号动作；全局降雨为 {100 * events["global_rain"]["speed_factor"]:g}% 限速代理。\n'
                  + controller_note, fontsize=9)
    save(fig, '04_comparison_table')
    data_files = [output / 'summary.json', output / 'manifest.json'] + sorted(output.glob('seed_*/*/timeline.csv'))
    if probe_path.exists():
        data_files.append(probe_path)
    (output / 'figures.json').write_text(json.dumps({'files': exported,
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files},
        'interpretation': controller_note}, ensure_ascii=False, indent=2) + '\n')
