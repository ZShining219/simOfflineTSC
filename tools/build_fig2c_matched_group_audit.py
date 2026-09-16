#!/usr/bin/env python3
"""Build transition-level Fig.2c Bellman consistency audit from formal NPZ shards."""
from pathlib import Path
import hashlib, json, csv
import numpy as np, pandas as pd, torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/FIG2C_MATCHED_GROUP_BELLMAN_AUDIT'; OUT.mkdir(parents=True,exist_ok=True)
SCENES=['sumohz1x1_config2','sumohz1x1','sumohz1x1_config4','sumohz1x1_config3']
ALIAS=dict(zip(SCENES,['S1','S2','S3','S4']))
PAIRS=[(SCENES[i],SCENES[j]) for i in range(4) for j in range(i+1,4)]
RUNROOT=ROOT/'output_data/tsc/sumo_dqn'

def load():
 rows=[]
 for scene in SCENES:
  for seed in range(5):
   dirs=sorted(RUNROOT.joinpath(scene).glob(f'p1_formal_dqn_{scene}_seed{seed}_400ep_*'))
   if not dirs: raise FileNotFoundError((scene,seed))
   run=dirs[-1]
   for p in sorted((run/'trajectory/episodes').glob('episode_*.npz')):
    with np.load(p,allow_pickle=False) as z:
     ep=int(z['episode_id'][0])
     if ep>100: continue
     s=z['state'].reshape(-1,8).astype(np.int16); ns=z['next_state'].reshape(-1,8).astype(np.int16); n=len(s)
     for k in range(n):
      obs=tuple(int(v) for v in s[k]); phase=int(np.asarray(z['current_phase']).reshape(-1)[k]); act=int(np.asarray(z['action']).reshape(-1)[k])
      key='|'.join(map(str,obs))+f'|p{phase}|a{act}'
      rows.append(dict(scene=scene,scene_alias=ALIAS[scene],seed=seed,episode=ep,decision_step=int(z['decision_step'].reshape(-1)[k]),simulation_time_s=(int(z['decision_step'].reshape(-1)[k])-1)*10,run_identity=str(run),npz_path=str(p),npz_transition_index=k,observation_key=key,**{f'observation_{i}':obs[i] for i in range(8)},current_phase=phase,action=act,reward=float(z['reward'].reshape(-1)[k]),terminated=bool(z['terminated'].reshape(-1)[k]),truncated=bool(z['truncated'].reshape(-1)[k]),**{f'next_observation_{i}':int(ns[k,i]) for i in range(8)},next_phase=int(z['next_phase'].reshape(-1)[k])))
 return pd.DataFrame(rows)

def model(path):
 s=torch.load(path,map_location='cpu',weights_only=False)['agents'][0]['target_model_state_dict']
 m=torch.nn.Sequential(torch.nn.Linear(16,20),torch.nn.ReLU(),torch.nn.Linear(20,20),torch.nn.ReLU(),torch.nn.Linear(20,8))
 m.load_state_dict({'0.weight':s['dense_1.weight'],'0.bias':s['dense_1.bias'],'2.weight':s['dense_2.weight'],'2.bias':s['dense_2.bias'],'4.weight':s['dense_3.weight'],'4.bias':s['dense_3.bias']}); return m.eval()

def main():
 x=load(); base=[]; gid=0
 for a,b in PAIRS:
  la=x[x.scene==a]; rb=x[x.scene==b]
  keys=set(la.observation_key)&set(rb.observation_key)
  for key in sorted(keys):
   L=la[la.observation_key==key].sort_values(['seed','episode','decision_step']).reset_index(drop=True)
   R=rb[rb.observation_key==key].sort_values(['seed','episode','decision_step']).reset_index(drop=True)
   for q in range(min(len(L),len(R))):
    u,v=L.iloc[q],R.iloc[q]; gid+=1
    d={ 'matched_group_id':f'MG{gid:07d}','scene_i':a,'scene_j':b,'scene_alias_i':ALIAS[a],'scene_alias_j':ALIAS[b],'scene_pair':f'{ALIAS[a]}-{ALIAS[b]}','source_run_i':u.run_identity,'source_run_j':v.run_identity,'seed_i':u.seed,'seed_j':v.seed,'episode_i':u.episode,'episode_j':v.episode,'decision_step_i':u.decision_step,'decision_step_j':v.decision_step,'simulation_time_i_s':u.simulation_time_s,'simulation_time_j_s':v.simulation_time_s,'observation_key':key,'exact_observation_match':True,'exact_action_match':bool(u.action==v.action),'action_i':u.action,'action_j':v.action,'reward_i':u.reward,'reward_j':v.reward,'delta_reward_signed':v.reward-u.reward,'delta_reward_abs':abs(v.reward-u.reward),'next_observation_equal':bool(all(u[f'next_observation_{k}']==v[f'next_observation_{k}'] for k in range(8)) and u.next_phase==v.next_phase),'terminated_i':u.terminated,'terminated_j':v.terminated,'truncated_i':u.truncated,'truncated_j':v.truncated}
    for k in range(8): d[f'observation_{k}']=u[f'observation_{k}']; d[f'next_observation_i_{k}']=u[f'next_observation_{k}']; d[f'next_observation_j_{k}']=v[f'next_observation_{k}']
    d['phase_i']=u.current_phase; d['phase_j']=v.current_phase; d['next_phase_i']=u.next_phase; d['next_phase_j']=v.next_phase; base.append(d)
 B=pd.DataFrame(base); B.to_csv(OUT/'02_matched_group_base.csv',index=False)
 ev=[]
 for es in SCENES:
  for seed in range(5):
   cp=sorted(RUNROOT.joinpath(es).glob(f'p1_formal_dqn_{es}_seed{seed}_400ep_*/checkpoints/resumable/episode_0400.pt'))[-1]; m=model(cp)
   ni=np.column_stack([B[f'next_observation_i_{k}'] for k in range(8)]+[np.eye(8)[B.next_phase_i.astype(int)]])
   nj=np.column_stack([B[f'next_observation_j_{k}'] for k in range(8)]+[np.eye(8)[B.next_phase_j.astype(int)]])
   with torch.no_grad(): qi=m(torch.tensor(ni,dtype=torch.float32)).numpy(); qj=m(torch.tensor(nj,dtype=torch.float32)).numpy()
   z=B[['matched_group_id','scene_i','scene_j','scene_pair','delta_reward_signed','reward_i','reward_j']].copy(); z['evaluator_id']=f'{ALIAS[es]}_seed{seed}'; z['evaluator_scene']=es; z['evaluator_training_seed']=seed; z['evaluator_checkpoint']=str(cp); z['q_next_i_max']=qi.max(1); z['q_next_j_max']=qj.max(1); z['bootstrap_i']=.95*z.q_next_i_max; z['bootstrap_j']=.95*z.q_next_j_max; z['delta_bootstrap_signed']=z.bootstrap_j-z.bootstrap_i; z['delta_bootstrap_abs']=z.delta_bootstrap_signed.abs(); z['target_i']=z.reward_i+z.bootstrap_i; z['target_j']=z.reward_j+z.bootstrap_j; z['delta_target_signed']=z.target_j-z.target_i; z['delta_target_abs']=z.delta_target_signed.abs(); ev.append(z)
 E=pd.concat(ev,ignore_index=True); E.to_csv(OUT/'03_matched_group_evaluator_bellman.csv',index=False)
 S=E.groupby('matched_group_id').agg(evaluator_count=('delta_bootstrap_signed','size'),delta_b_median=('delta_bootstrap_signed','median'),delta_b_q1=('delta_bootstrap_signed',lambda x:x.quantile(.25)),delta_b_q3=('delta_bootstrap_signed',lambda x:x.quantile(.75)),delta_b_mean=('delta_bootstrap_signed','mean'),delta_b_std=('delta_bootstrap_signed','std'),delta_b_min=('delta_bootstrap_signed','min'),delta_b_max=('delta_bootstrap_signed','max'),delta_y_median=('delta_target_signed','median'),delta_y_q1=('delta_target_signed',lambda x:x.quantile(.25)),delta_y_q3=('delta_target_signed',lambda x:x.quantile(.75)),delta_y_mean=('delta_target_signed','mean'),delta_y_std=('delta_target_signed','std'),delta_y_min=('delta_target_signed','min'),delta_y_max=('delta_target_signed','max')).reset_index(); S=S.merge(B[['matched_group_id','scene_pair','scene_i','scene_j','delta_reward_signed','delta_reward_abs']],on='matched_group_id'); S.to_csv(OUT/'04_matched_group_evaluator_summary.csv',index=False)
 def stats(vals):
  a=np.asarray(vals,float); return dict(n=len(a),median=np.median(a),mean=np.mean(a),q1=np.quantile(a,.25),q3=np.quantile(a,.75),p05=np.quantile(a,.05),p95=np.quantile(a,.95),std=np.std(a,ddof=1) if len(a)>1 else 0,mad=np.median(np.abs(a-np.median(a))))
 rows=[]
 for name,v in [('delta_reward',S.delta_reward_signed),('median_delta_bootstrap',S.delta_b_median),('median_delta_target',S.delta_y_median),('abs_delta_reward',S.delta_reward_abs),('abs_median_delta_bootstrap',S.delta_b_median.abs()),('abs_median_delta_target',S.delta_y_median.abs())]: rows.append({'quantity':name,**stats(v)})
 pd.DataFrame(rows).to_csv(OUT/'06_global_distribution_summary.csv',index=False)
 pair=[]
 for p,g in S.groupby('scene_pair'):
  for n,v in [('delta_reward_abs',g.delta_reward_abs),('delta_bootstrap_abs',g.delta_b_median.abs()),('delta_target_abs',g.delta_y_median.abs())]: pair.append({'scene_pair':p,'quantity':n,**stats(v)})
 pd.DataFrame(pair).to_csv(OUT/'05_scene_pair_summary.csv',index=False)
 episodes_n = pd.concat([B[['scene_i','seed_i','episode_i']].astype(str).agg('/'.join,axis=1), B[['scene_j','seed_j','episode_j']].astype(str).agg('/'.join,axis=1)]).nunique()
 runs_n = pd.concat([B.source_run_i, B.source_run_j]).nunique()
 (OUT/'07_pseudoreplication_audit.md').write_text(f'# Pseudoreplication audit\n\nUnique exact keys: {B.observation_key.nunique()}; matched transition pairs/groups: {len(B)}; evaluator rows: {len(E)}; source runs: {runs_n}; source episodes: {episodes_n}; evaluators: {E.evaluator_id.nunique()}. Evaluator rows are learned-Q evaluations, not independent environment samples. Pairing is deterministic one-to-one zip within each exact key after sorting; no Cartesian product.\n',encoding='utf-8')
 (OUT/'00_README.md').write_text('# FIG2C matched-group Bellman audit\n\nScope: HOA_USED_100 = formal DQN episodes 1-100, time_control=none. Observation is 8 lane-count fields plus current phase one-hot (16-D DQN input); exact key includes all 8 counts, phase, action. Reward is saved training reward (mean over ten 1-s simulator steps), gamma=0.95. Existing formal evaluator uses target network max Q(next observation, action), no Double-DQN selection/evaluation split, and no terminal mask. Scenes: S1=sumohz1x1_config2, S2=base (raw sumohz1x1), S3=sumohz1x1_config4, S4=sumohz1x1_config3. One-to-one deterministic pairing within key avoids Cartesian expansion.\n',encoding='utf-8')
 (OUT/'01_provenance_audit.csv').write_text('artifact,source\nrunlist,output_data/analysis/plan1/p1_formal_20_runlist_20260722.csv\nmatching_logic,tools/plan1_observation_conditioned_outcomes.py\ntransition_source,output_data/tsc/sumo_dqn/*/trajectory/episodes/episode_*.npz\nevaluator_source,output_data/tsc/sumo_dqn/*/checkpoints/resumable/episode_0400.pt\n',encoding='utf-8')
 finite_ok = np.isfinite(E.select_dtypes('number').to_numpy()).all()
 (OUT/'08_validation_report.md').write_text(f'# Validation\n\nExact observation/action match enforced by key; next observation equality retained. Evaluators: {E.evaluator_id.nunique()} (expected 20). Bellman identity max error: {np.max(np.abs(E.delta_target_signed-(E.delta_reward_signed+E.delta_bootstrap_signed))):.3g}. NaN/inf present: {not finite_ok}.\n',encoding='utf-8')
if __name__=='__main__': main()
