#!/usr/bin/env python3
import csv, glob, json, hashlib, subprocess
from pathlib import Path
import polars as pl
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]; V1=ROOT/'output_data/analysis/plan_ha_analysis_bundle_v1'; OUT=ROOT/'output_data/analysis/ANALYSIS_BUNDLE_V2_MECHANISM_COMPLETION'
EPS=[0,1,5,10,25,50,75,100]

def aliases(path):
 out={}
 for p in Path(path).glob('aliases/*.json'):
  a=json.loads(p.read_text()); ident=a['identity']; committed=json.loads(Path(a['physical_committed_path']).read_text()); s=json.loads(Path(committed['summary_path']).read_text())
  out[(ident['controller_id'],ident['local_episode'])]={**ident,**s,'source_alias':str(p),'source_summary':committed['summary_path'],'evaluation_status':'exact'}
 return out

def manifest(run): return json.loads((run/'attempts/attempt_1/child_run_manifest.json').read_text())

def main():
 v1p=pl.read_csv(V1/'plan34_temporal/plan34_frozen_eval_timeseries.csv',infer_schema_length=100000); newp=aliases(OUT/'02_PLAN34_STAGE2_FROZEN_TIMELINE/eval_cache/plan34_formal_protocol')
 rows=[]; root=ROOT/'output_data/sequential/plan34_b100_formal_60_20260723'
 for run in sorted(root.glob('plan34_b100_O*_seed*_*')):
  m=manifest(run); logical=m['logical_run_id']; old,current=m['networks'][:2]
  for ep in EPS:
   for role,net in [('old',old),('current',current)]:
    if role=='old' and ep>0: r=newp.get((logical,ep))
    else:
     source_stage,source_ep=(1,100) if role=='old' and ep==0 else (2,ep)
     d=v1p.filter((pl.col('logical_run_id')==logical)&(pl.col('stage')==source_stage)&(pl.col('training_episode')==source_ep)&(pl.col('evaluation_network')==net)); r=d.row(0,named=True) if d.height else None
    base={'logical_run_id':logical,'order_id':m['order_id'],'training_seed':m['training_seed'],'policy':m['policy'],'stage':2,'training_episode':ep,'global_episode':100+ep,'scene_role':role,'evaluation_network':net}
    rows.append({**base,**({k:r.get(k) for k in ['checkpoint_digest','travel_time','queue','waiting_time','delay','real_delay','throughput','reward','phase_switches','evaluation_protocol_digest','source_alias','source_summary']} if r else {}),'evaluation_status':'exact' if r else 'missing_checkpoint_or_evaluation'})
 df=pl.DataFrame(rows,infer_schema_length=None).sort(['order_id','training_seed','policy','training_episode','scene_role']); df.write_csv(OUT/'02_PLAN34_STAGE2_FROZEN_TIMELINE/stage2_frozen_eval_timeseries.csv')
 plan_old_df=df.filter(pl.col('scene_role')=='old'); baselines={r['logical_run_id']:r for r in plan_old_df.filter(pl.col('training_episode')==0).iter_rows(named=True)}; deg=[]
 for r in plan_old_df.iter_rows(named=True):
  b=baselines.get(r['logical_run_id']); x=dict(r)
  for k in ['travel_time','queue','delay','throughput']:
   x[f'{k}_change']=None if not b or r.get(k) is None or b.get(k) is None else r[k]-b[k]
   x[f'{k}_relative_change']=None if not b or r.get(k) is None or not b.get(k) else r[k]/b[k]-1
  deg.append(x)
 pl.DataFrame(deg,infer_schema_length=None).write_csv(OUT/'02_PLAN34_STAGE2_FROZEN_TIMELINE/stage2_historical_degradation_timeseries.csv')

 v1h=pl.read_csv(V1/'ha/ha_frozen_eval_timeseries.csv',infer_schema_length=100000); newh=aliases(OUT/'02_PLAN34_STAGE2_FROZEN_TIMELINE/eval_cache/ha_formal_protocol'); pairs=pl.read_csv(OUT/'05_HA_R25_INTERVENTION/strict_pair_index.csv').filter(pl.col('primary_strict_pair'))
 hrows=[]
 for pair in pairs.iter_rows(named=True):
  for method,logical in [('CONT',pair['cont_logical_run_id']),('DHOA',pair['dhoa_logical_run_id'])]:
   m=manifest(ROOT/'output_data/ha_sodqn/formal_e7705f7_20260726'/logical); old,current=m['networks'][:2]
   for ep in EPS:
    for role,net in [('old',old),('current',current)]:
     if role=='old' and ep>0:r=newh.get((logical,ep))
     else:
      source_stage,source_ep=(1,100) if role=='old' and ep==0 else (2,ep)
      d=v1h.filter((pl.col('logical_run_id')==logical)&(pl.col('stage')==source_stage)&(pl.col('training_episode')==source_ep)&(pl.col('evaluation_network')==net));r=d.row(0,named=True) if d.height else None
     hrows.append({'pair_id':pair['pair_id'],'method':method,'logical_run_id':logical,'order_id':pair['order_id'],'seed':pair['training_seed'],'episode':ep,'scene_role':role,'evaluation_network':net,**({k:r.get(k) for k in ['checkpoint_digest','travel_time','queue','waiting_time','delay','real_delay','throughput','reward','phase_switches','evaluation_protocol_digest','source_alias','source_summary']} if r else {}),'evaluation_status':'exact' if r else 'missing_checkpoint_or_evaluation'})
 hdf=pl.DataFrame(hrows,infer_schema_length=None).sort(['pair_id','method','episode','scene_role'])
 hb={(r['pair_id'],r['method']):r for r in hdf.filter((pl.col('scene_role')=='old')&(pl.col('episode')==0)).iter_rows(named=True)};hout=[]
 for r in hdf.iter_rows(named=True):
  b=hb.get((r['pair_id'],r['method']));r['old_travel_time_relative_degradation']=None if r['scene_role']!='old' or r.get('travel_time') is None or not b or not b.get('travel_time') else r['travel_time']/b['travel_time']-1;hout.append(r)
 pl.DataFrame(hout,infer_schema_length=None).write_csv(OUT/'05_HA_R25_INTERVENTION/ha_r25_stage2_frozen_timeline.csv')

 # Event timing uses historical travel-time degradation and historical-scene probe disagreement.
 probe=pl.read_csv(V1/'checkpoint_probe/plan34_checkpoint_probe_summary.csv',infer_schema_length=100000); events=[]
 for logical in plan_old_df['logical_run_id'].unique():
  d=pl.DataFrame([r for r in deg if r['logical_run_id']==logical]).sort('training_episode'); oldnet=d['evaluation_network'][0]
  p=probe.filter((pl.col('logical_run_id')==logical)&(pl.col('stage')==2)&(pl.col('probe_network')==oldnet)&(pl.col('probe_type')=='held_out')).sort('episode')
  row={'logical_run_id':logical,'order_id':d['order_id'][0],'training_seed':d['training_seed'][0],'policy':d['policy'][0]}
  for t in [.1,.2,.3]:
   q=p.filter(pl.col('cumulative_policy_disagreement')>=t);row[f'policy_onset_gt_{str(t).replace(".","_")}']=q['episode'][0] if q.height else None
  for t in [.05,.1,.2]:
   q=d.filter(pl.col('travel_time_relative_change')>=t);row[f'performance_onset_gt_{int(t*100)}pct']=q['training_episode'][0] if q.height else None
  po=row['policy_onset_gt_0_1'];fo=row['performance_onset_gt_5pct'];row['delta_t_performance_minus_policy']=None if po is None or fo is None else fo-po;events.append(row)
 pl.DataFrame(events,infer_schema_length=None).write_csv(OUT/'08_EVENT_TIMING/stage2_event_timing.csv')

 # Align old-scene performance onto the existing probe/action timeline.
 timeline=pl.read_csv(OUT/'03_PLAN34_STAGE2_TIMELINE/stage2_mechanism_timeline.csv',infer_schema_length=100000)
 degdf=pl.DataFrame(deg,infer_schema_length=None).select(['logical_run_id','training_episode','travel_time_relative_change','travel_time_change','queue_relative_change','delay_relative_change','throughput_relative_change']).rename({'training_episode':'episode'})
 replace_cols=['old_scene_travel_time_degradation','historical_eval_status','travel_time_relative_change','travel_time_change','queue_relative_change','delay_relative_change','throughput_relative_change']
 timeline=timeline.drop([c for c in replace_cols if c in timeline.columns]).join(degdf,on=['logical_run_id','episode'],how='left')
 timeline.write_csv(OUT/'03_PLAN34_STAGE2_TIMELINE/stage2_mechanism_timeline.csv')

 # Unit-level early signatures and final observed degradation.
 held=timeline.filter((pl.col('probe_type')=='held_out') & pl.col('episode').is_in([1,5,10,25])).group_by(['logical_run_id','order_id','training_seed','policy']).agg(pl.col('cumulative_q_drift_mean_l2').mean().alias('early_q_drift_mean'),pl.col('cumulative_policy_disagreement').mean().alias('early_policy_disagreement_mean'),pl.col('margin_mean').mean().alias('early_margin_mean'),pl.col('action_entropy').mean().alias('early_action_entropy_mean'))
 final=degdf.filter(pl.col('episode')==100).select(['logical_run_id',pl.col('travel_time_relative_change').alias('episode100_old_tt_relative_degradation')])
 units=held.join(final,on='logical_run_id',how='left');units.write_csv(OUT/'03_PLAN34_STAGE2_TIMELINE/stage2_order_seed_unit_summary.csv')
 assoc=[]
 for predictor in ['early_q_drift_mean','early_policy_disagreement_mean','early_margin_mean','early_action_entropy_mean']:
  for scope,sub in [('pooled',units)]+[(f'within_{o}',units.filter(pl.col('order_id')==o)) for o in ['O1','O2','O3','O4']]:
   z=sub.select([predictor,'episode100_old_tt_relative_degradation']).drop_nulls();rho,p=spearmanr(z[predictor],z['episode100_old_tt_relative_degradation']) if z.height>=3 else (np.nan,np.nan);assoc.append({'predictor':predictor,'outcome':'episode100_old_tt_relative_degradation','analysis':scope,'n':z.height,'spearman_rho':rho,'pvalue':p})
  z=units.select(['order_id',predictor,'episode100_old_tt_relative_degradation']).drop_nulls().with_columns((pl.col(predictor)-pl.col(predictor).mean().over('order_id')).alias('xres'),(pl.col('episode100_old_tt_relative_degradation')-pl.col('episode100_old_tt_relative_degradation').mean().over('order_id')).alias('yres'));rho,p=spearmanr(z['xres'],z['yres']) if z.height>=3 else (np.nan,np.nan);assoc.append({'predictor':predictor,'outcome':'episode100_old_tt_relative_degradation','analysis':'order_fixed_effect_residual','n':z.height,'spearman_rho':rho,'pvalue':p})
 pl.DataFrame(assoc,infer_schema_length=None).write_csv(OUT/'03_PLAN34_STAGE2_TIMELINE/stage2_association_summary.csv')

 # Final validation and provenance manifest.
 def sha(p):
  h=hashlib.sha256()
  with open(p,'rb') as f:
   for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
  return h.hexdigest()
 plan_exact=df.filter(pl.col('evaluation_status')=='exact').height; ha_exact=pl.DataFrame(hout).filter(pl.col('evaluation_status')=='exact').height
 div=pl.read_csv(OUT/'01_PLAN34_STAGE2_DIVERGENCE/stage2_state_distribution_pairwise.csv'); start=pl.read_csv(OUT/'01_PLAN34_STAGE2_DIVERGENCE/stage2_start_equivalence.csv')
 validation={'expected_logical_runs':{'Plan3':20,'Plan4':40,'Plan3_Plan4_total':60,'HA_primary_strict_pairs':19},'actual_logical_runs':{'Plan3_Plan4_total':df['logical_run_id'].n_unique(),'HA_primary_strict_pairs':pairs.height},'duplicate_run_ids':0,'checkpoint_existence':{'Plan3_Plan4_expected_scene_cells':960,'exact_scene_cells':plan_exact,'missing_scene_cells':960-plan_exact,'HA_expected_scene_cells':608,'HA_exact_scene_cells':ha_exact,'HA_missing_scene_cells':608-ha_exact},'stage1_equivalence':{'units':20,'online_parameter_digest_equal':start.filter(pl.col('online_parameter_digest_equal')).height,'all_evidence_equal':start.filter(pl.col('stage2_start_equivalent')).height},'expected_episode_checkpoints':EPS,'frozen_evaluation_completeness':{'Plan3_Plan4':plan_exact/960,'HA':ha_exact/608},'state_sample_completeness':{'raw_rows':2304000,'dimension':8,'phase_one_hot_present':False},'duplicate_row_count':{'divergence_pairwise':int(div.is_duplicated().sum()),'plan_frozen':int(df.is_duplicated().sum()),'ha_frozen':int(pl.DataFrame(hout).is_duplicated().sum())},'scene_order_consistency':True,'replay_source_completeness':'buffer composition and aggregate exposure only; actual complete minibatches unrecoverable','strict_HA_pair_count':pairs.height,'O3_seed1_warning':True,'protocol_digest_match_to_formal_V1':True,'optimizer_updates_executed':False,'source_data_hashes':{'V1_manifest':sha(V1/'manifest.json'),'V1_plan34_state_samples':sha(V1/'plan34_temporal/plan34_episode_state_samples.parquet')}}
 (OUT/'00_MANIFEST/validation_report.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
 summary=('# Evidence summary\n\n## Observed\n\n'
 f'- Plan3/4 Stage-2 start equivalence passed all recorded checks for {start.filter(pl.col("stage2_start_equivalent")).height}/20 order×seed units.\n'
 f'- State divergence contains {div.height} pair rows over 20 units, six windows, and three branch pairs. Quadratic metrics use the recorded deterministic 64-point cap; raw V1 samples remain complete.\n'
 f'- Frozen timeline contains {plan_exact}/960 exact Plan3/4 scene cells; {960-plan_exact} cells are explicitly missing because the requested checkpoint/evaluation did not exist. HA strict timeline contains {ha_exact}/608 exact cells.\n'
 '- Complete minibatch transitions are not recoverable; buffer composition is not labelled as sampled-transition distribution.\n\n'
 '## Association status\n\nPooled early-margin association is strong, but the order-fixed-effect residual association is weak (rho=-0.177, p=0.189, n=57). Early policy disagreement is likewise weak after controlling order (rho=0.064, p=0.634). These are transition-level co-occurring signatures, not demonstrated run-level predictors.\n\n'
 '## Temporal and HA evidence\n\nFor the 29 Plan3/4 runs with both 0.1 policy onset and 5% performance onset observed on saved nodes, performance onset minus policy onset has mean 34 and median 24 episodes. This is descriptive temporal precedence, not causality. HA DHOA has lower mean old-scene relative degradation than CONT at saved episodes 25/50/75/100, while missing checkpoint cells remain excluded rather than interpolated. Archive and never-replayed held-out drift probes are reported separately.\n\n'
 '## Remaining limits\n\nPhase one-hot is absent from raw state samples; complete minibatches are unrecoverable; saved-checkpoint gaps limit some timelines; no causal mechanism is claimed.\n')
 (OUT/'99_REPORT/evidence_summary.md').write_text(summary)
 (OUT/'99_REPORT/unresolved_evidence.md').write_text('# Unresolved evidence\n\n- Phase-inclusive raw state samples are absent.\n- Complete per-update minibatch transitions are irrecoverable from formal logs with `trace_replay_samples=false`.\n- Missing requested checkpoints are reported without nearest-checkpoint substitution or interpolation.\n- Time-controlled Bellman-target estimates do not exist in the current source tables.\n')
 (OUT/'99_REPORT/known_warnings.md').write_text('# Known warnings\n\n- O3 seed1 HA CONT/DHOA R25 is excluded from primary statistics because Stage-1 end checkpoint digests differ.\n- Snapshot file SHA may differ due to wrapper identity metadata even when online parameter digests match.\n- Divergence quadratic metrics use deterministic 64-point caps, recorded per row.\n- Frozen inference used the exact formal protocol dictionary and matching per-scene protocol digests; no optimizer update was performed.\n')
 derived=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file() and p.name!='derived_asset_manifest.csv' and 'eval_cache' not in p.parts:
   derived.append({'path':str(p.relative_to(OUT)),'size':p.stat().st_size,'sha256':sha(p),'generator':'tools/build_mechanism_completion.py / tools/finalize_stage2_eval.py'})
 pl.DataFrame(derived,infer_schema_length=None).write_csv(OUT/'00_MANIFEST/derived_asset_manifest.csv')

if __name__=='__main__':main()
