"""Render Plan 2 final time-series and existing checkpoint summaries."""
import argparse, csv, json, math, os, tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "simofflinetsc-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NETWORKS = ("sumohz1x1_config2", "sumohz1x1", "sumohz1x1_config4", "sumohz1x1_config3")
SCENES = dict(zip(NETWORKS, ("S1", "S2", "S3", "S4")))
POLICIES = ("fixedtime", "maxpressure", "only_q1", "only_q4", "full", "non_current_full")
LABELS = {"fixedtime":"FixedTime","maxpressure":"MaxPressure","only_q1":"Only Q1","only_q4":"Only Q4","full":"Full (current scene)","non_current_full":"Full (other scenes)"}
COLORS = dict(zip(POLICIES, ("#7f7f7f","#e69f00","#56b4e9","#cc79a7","#009e73","#d55e00")))
UPDATES=(0,14400,36000,72000,108000,144000)
BOOTSTRAP_SEED=20260724

def _save(fig, base, dpi):
    base=Path(base); base.parent.mkdir(parents=True,exist_ok=True); out=[]
    for ext in ("png","pdf"):
        p=base.with_suffix('.'+ext); fig.savefig(p,dpi=dpi if ext=='png' else None,bbox_inches='tight'); out.append(p)
    plt.close(fig); return out

def _ci(values):
    a=np.asarray(values,float); mean=np.nanmean(a,axis=0)
    if len(a)==1:return mean,mean,mean
    rng=np.random.default_rng(BOOTSTRAP_SEED); idx=rng.integers(0,len(a),size=(1000,len(a)))
    means=np.nanmean(a[idx],axis=1); lo,hi=np.nanpercentile(means,(2.5,97.5),axis=0); return mean,lo,hi

def _trend(frame):
    frame=frame.sort_values('simulation_time_seconds').copy(); window=6
    frame['reward']=frame.reward_network_mean.expanding().mean()
    frame['reward_sum']=frame.reward_network_sum.expanding().mean()
    frame['queue_mean']=frame.queue_network_mean.rolling(window,min_periods=1).mean()
    frame['queue_sum']=frame.queue_network_sum.rolling(window,min_periods=1).mean()
    frame['delay']=frame.delay_network_weighted_mean.expanding().mean()
    frame['throughput_window']=frame.throughput_interval.rolling(window,min_periods=1).sum()
    frame['throughput_cumulative_plot']=frame.throughput_cumulative
    return frame

def _load_timeseries(eval_root, algorithm, baseline):
    rows=[]; root=Path(eval_root)/algorithm; manifest=json.loads((root/'collection_manifest.json').read_text())
    for item in manifest['controllers']:
        path=root/item['controller_id']/'records.jsonl'
        with path.open() as h:
            for line in h:
                r=json.loads(line); r.update(policy=item['policy'],seed=item['offline_training_seed'],network=item['network']); rows.append(r)
    with (Path(baseline)/'records.jsonl').open() as h:
        for line in h:
            r=json.loads(line)
            if r['agent'] in ('fixedtime','maxpressure') and r['network'] in NETWORKS:
                r.update(policy=r['agent'],seed=r['evaluation_seed']); rows.append(r)
    frame=pd.DataFrame(rows); return pd.concat([_trend(g) for _,g in frame.groupby(['network','policy','seed'])],ignore_index=True)

def _timeseries_figures(frame,out,dpi):
    specs=(("reward","Cumulative mean reward (network mean)"),("reward_sum","Cumulative mean reward (network sum)"),("queue_mean","Queue (lane mean, 60 s moving mean)"),("queue_sum","Queue (network sum, 60 s moving mean)"),("delay","Cumulative weighted delay"),("throughput_window","Throughput per 60 s"),("throughput_cumulative_plot","Cumulative throughput"))
    table=[]; figures=[]
    for net in NETWORKS:
        fig,axes=plt.subplots(4,2,figsize=(15,15),constrained_layout=True); axes.flat[-1].set_visible(False); sub=frame[frame.network==net]
        for ax,(field,title) in zip(axes.flat,specs):
            for policy in POLICIES:
                p=sub[sub.policy==policy]; pivot=p.pivot_table(index='seed',columns='simulation_time_seconds',values=field).sort_index(axis=1)
                if pivot.empty: continue
                mean,lo,hi=_ci(pivot.to_numpy()); x=pivot.columns.to_numpy(); ax.plot(x,mean,label=LABELS[policy],color=COLORS[policy],lw=1.6); ax.fill_between(x,lo,hi,color=COLORS[policy],alpha=.12)
                table.extend(dict(network=net,scene=SCENES[net],policy=policy,metric=field,simulation_time_seconds=float(xx),mean=float(m),ci95_lower=float(l),ci95_upper=float(u),seed_count=len(pivot)) for xx,m,l,u in zip(x,mean,lo,hi))
            ax.set(title=title,xlabel='Simulation time (s)',xlim=(0,3600)); ax.grid(alpha=.2)
        axes.flat[0].legend(ncols=2,fontsize=8); fig.suptitle(f"{SCENES[net]} ({net}) final-checkpoint performance")
        figures+=_save(fig,Path(out)/'figures'/'timeseries'/f'{SCENES[net].lower()}_final_timeseries',dpi)
    return table,figures

def _checkpoint_rows(run_list,algorithm,baseline):
    rows=[]
    with open(run_list,newline='') as h:
        for spec in csv.DictReader(h):
            if spec['include'].lower()!='true' or spec['algorithm']!=algorithm:continue
            policy='non_current_full' if spec['dataset_kind']=='leave_one_out' else {'Q1':'only_q1','Q4':'only_q4','full':'full'}[spec['dataset_stage']]
            with open(Path(spec['run_dir'])/'metrics'/'offline_records.jsonl') as m:
                for line in m:
                    r=json.loads(line)
                    if r['record_type'] in ('EVALUATION','FINAL_EVALUATION'):
                        rows.append(dict(network=spec['evaluation_network'],policy=policy,seed=int(spec['offline_training_seed']),update=int(r['training_update']),travel_time=r['travel_time'],reward=r['reward_mean'],queue=r['queue'],delay=r['delay'],throughput=r['throughput']))
    with open(Path(baseline)/'summary.csv',newline='') as h:
        for r in csv.DictReader(h):
            if r['agent'] in ('fixedtime','maxpressure') and r['network'] in NETWORKS:
                rows.append(dict(network=r['network'],policy=r['agent'],seed=int(r['evaluation_seed']),update=-1,travel_time=float(r['travel_time']),reward=float(r['reward_mean']),queue=float(r['queue']),delay=float(r['delay']),throughput=float(r['throughput'])))
    return pd.DataFrame(rows)

def _checkpoint_figures(frame,out,dpi):
    specs=(("travel_time","Average travel time (s)"),("reward","Reward mean"),("queue","Queue"),("delay","Delay"),("throughput","Throughput")); table=[]; figures=[]
    for net in NETWORKS:
        fig,axes=plt.subplots(3,2,figsize=(15,12),constrained_layout=True); axes.flat[-1].set_visible(False); sub=frame[frame.network==net]
        for ax,(field,title) in zip(axes.flat,specs):
            for policy in POLICIES:
                p=sub[sub.policy==policy];
                if p.empty:continue
                if policy in ('fixedtime','maxpressure'):
                    vals=p[field].to_numpy(float); mean,lo,hi=_ci(vals[:,None]); ax.axhline(mean[0],color=COLORS[policy],ls='--',lw=1.4,label=LABELS[policy]); ax.fill_between(UPDATES,lo[0],hi[0],color=COLORS[policy],alpha=.1); continue
                pivot=p.pivot_table(index='seed',columns='update',values=field).reindex(columns=UPDATES); mean,lo,hi=_ci(pivot.to_numpy()); ax.plot(UPDATES,mean,color=COLORS[policy],marker='o',label=LABELS[policy]); ax.fill_between(UPDATES,lo,hi,color=COLORS[policy],alpha=.12)
                table.extend(dict(network=net,scene=SCENES[net],policy=policy,metric=field,update=int(x),mean=float(m),ci95_lower=float(l),ci95_upper=float(u),seed_count=len(pivot)) for x,m,l,u in zip(UPDATES,mean,lo,hi))
            ax.set(title=title,xlabel='Offline gradient update'); ax.grid(alpha=.2)
        axes.flat[0].legend(ncols=2,fontsize=8); fig.suptitle(f"{SCENES[net]} ({net}) checkpoint performance"); figures+=_save(fig,Path(out)/'figures'/'checkpoints'/f'{SCENES[net].lower()}_checkpoint_performance',dpi)
    return table,figures

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-list',required=True); p.add_argument('--algorithm',choices=('batch_dqn','cql_dqn'),required=True); p.add_argument('--evaluation-root',required=True); p.add_argument('--baseline-package',required=True); p.add_argument('--output',required=True); p.add_argument('--dpi',type=int,default=160); a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True); (out/'tables').mkdir()
    ts=_load_timeseries(a.evaluation_root,a.algorithm,a.baseline_package); ts_table,figs1=_timeseries_figures(ts,out,a.dpi); cp=_checkpoint_rows(a.run_list,a.algorithm,a.baseline_package); cp_table,figs2=_checkpoint_figures(cp,out,a.dpi)
    pd.DataFrame(ts_table).to_csv(out/'tables'/'timeseries_summary.csv',index=False); pd.DataFrame(cp_table).to_csv(out/'tables'/'checkpoint_summary.csv',index=False)
    manifest={'schema_version':1,'status':'completed','algorithm':a.algorithm,'run_list':str(Path(a.run_list).resolve()),'decision_record_reproduction':str((Path(a.evaluation_root)/a.algorithm).resolve()),'baseline_package':str(Path(a.baseline_package).resolve()),'semantics':{'checkpoint':'final_update_144000','timeseries':'original_seed_decision_record_reproduction','queue_smoothing_seconds':60,'reward_delay':'cumulative_mean','confidence_interval':'95pct_percentile_bootstrap_over_5_seeds'},'figures':[str(x.relative_to(out)) for x in figs1+figs2],'tables':['tables/timeseries_summary.csv','tables/checkpoint_summary.csv']}
    (out/'plotting_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(out)
if __name__=='__main__':main()
