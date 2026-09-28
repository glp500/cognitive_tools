"""Rebuild the supervisor data round from archived pilot_v1 evidence.

Run from the repository root with python-pptx, matplotlib, pandas, numpy installed.
Input data and the reference deck are read-only. Outputs stay in results/presentation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cognitive_tools.scenarios import build_environment_maps
OUT=ROOT/'results/presentation'
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True,parents=True)
DATA=OUT/'data'; DATA.mkdir(exist_ok=True)
RUNS=ROOT/'results/q_learning_baseline/experiments'
AN=ROOT/'results/q_learning_baseline/social_analysis/pilot_v1_analysis'
REF=Path('/home/gavinl/Downloads/Gavin Lip. Cogntive Tools Project Draft.pptx')
BLUE='#31688E'; TEAL='#218F8D'; GOLD='#B88718'; INK='#111111'; GREY='#666666'
COLORS={'R1':BLUE,'R2':TEAL,'R3':GOLD}
SCENARIOS=['uniform_high','patchy_high','split_high_low']
NAMES={'uniform_high':'Uniform high','patchy_high':'Patchy high','split_high_low':'Split high / low'}
LABELS={'R1':'Local','R2':'Mixed','R3':'Global'}
plt.rcParams.update({'font.family':'Liberation Sans','font.size':19,'axes.titlesize':23,
 'axes.labelsize':19,'xtick.labelsize':17,'ytick.labelsize':18,'axes.spines.top':False,
 'axes.spines.right':False,'axes.edgecolor':'#777777','axes.linewidth':.8,
 'text.color':INK,'axes.labelcolor':INK,'xtick.color':GREY,'ytick.color':GREY,
 'savefig.facecolor':'white','figure.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
manifest=json.loads((AN/'analysis_manifest.json').read_text())
summary=pd.read_csv(AN/'data/paired_adaptive_minus_r0_summary.csv')
trajectory=pd.read_csv(AN/'data/trajectory_summary.csv')
primary={}; configs={}; train={}; audit={'source_commit':'b0340b10dd33499a714af5c22ab124b05f680a3a','checks':[],'inputs':[]}
KEY=['scenario','population','replicate']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert manifest['warnings']==[]
for row in manifest['inputs']:
 rid=row['run_id']; d=RUNS/rid; cfg=json.loads((d/'config.json').read_text()); configs[rid]=cfg
 assert sha(d/'config.json')==row['config_sha256']
 assert cfg['run_metadata']['git_commit_sha']==audit['source_commit']
 assert cfg['run_metadata']['git_worktree_dirty'] is False
 assert cfg['replicates']==10 and cfg['training_steps']==5000 and cfg['evaluation_steps']==1000
 assert cfg['network_eval']=='frozen'
 e=pd.read_csv(d/'data/evaluation_summary.csv')
 e=e[(e.strategy=='q_learning')&(e.evaluation_mode=='fresh_reset')].copy()
 assert len(e)==60 and not e.duplicated(KEY).any()
 assert e.groupby(['scenario','population']).size().eq(10).all()
 assert set(e.replicate)==set(range(10))
 if cfg['social_mode']=='fixed':assert e.network_start.eq('terminal').all()
 primary[rid]=e.set_index(KEY).sort_index()
 train[rid]=pd.read_csv(d/'data/training_timeseries.csv')
 audit['inputs'].append({'run':rid,'config_sha256':sha(d/'config.json'),
  'evaluation_sha256':sha(d/'data/evaluation_summary.csv'),
  'training_sha256':sha(d/'data/training_timeseries.csv'),'primary_rows':len(e)})
for row in manifest['inputs']:
 if row['treatment']!='R0':continue
 rid=row['run_id']; aid=row['paired_adaptive_run_id']
 assert configs[rid]['matched_rewire_schedule_sha256']==sha(RUNS/aid/'data/rewiring_schedule.csv')
 a=pd.read_csv(RUNS/aid/'data/rewiring_schedule.csv'); b=pd.read_csv(RUNS/rid/'data/rewiring_schedule.csv')
 j=a.merge(b,on=KEY+['time'],validate='one_to_one',suffixes=('_a','_b'))
 assert len(j)==len(a)==len(b)==6000
 assert j.successful_rewires_a.eq(j.successful_rewires_b).all()
 assert j.target_rewires_b.eq(j.successful_rewires_a).all()
audit['checks'].append('All 9 config hashes, 540 unique primary rows, and all 18,000 matched schedule checkpoints verified.')
metric_cols={'resource_fraction':'eval_mean_mean_resource_fraction','reserve_welfare':'eval_mean_mean_reserve_welfare',
 'wealth_gini':'final_wealth_gini','low_extraction':'eval_mean_low_extraction_rate'}
# Recompute each plotted difference directly from matched raw evaluation records.
paired=[]
for r in summary.itertuples():
 if r.metric not in metric_cols:continue
 c=metric_cols[r.metric]
 a=primary[r.adaptive_run_id].loc[(r.scenario,r.population),c]
 b=primary[r.r0_run_id].loc[(r.scenario,r.population),c]
 assert a.index.equals(b.index)
 delta=a-b
 assert np.isclose(delta.mean(),r.mean_adaptive_minus_r0,atol=1e-13)
 assert len(delta)==r.n_pairs==10
 # Reconstruct archived bootstrap seeds and intervals independently of analysis.py.
 parts=[1729,'paired',r.adaptive_run_id,r.r0_run_id,r.adaptive_label,r.r0_label,
  float(r.rewire_theta),float(r.rewire_mu),r.scenario,int(r.population),r.metric]
 seed=int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')%(2**32-1)
 rng=np.random.default_rng(seed)
 vals=delta.to_numpy(); means=vals[rng.integers(0,10,size=(2000,10))].mean(axis=1)
 ci=np.quantile(means,[.025,.975])
 assert np.allclose(ci,[r.bootstrap_ci_low,r.bootstrap_ci_high],atol=1e-12)
 for rep,v in delta.items():paired.append({'scenario':r.scenario,'population':r.population,'treatment':r.adaptive_label,'replicate':rep,'metric':r.metric,'difference':v})
audit['checks'].append('54 paired means and 95% percentile-bootstrap intervals independently reproduced from raw evaluation records.')
summary.to_csv(DATA/'paired_summary.csv',index=False)
pd.DataFrame(paired).to_csv(DATA/'paired_replicates.csv',index=False)
all_primary=pd.concat([d.reset_index().assign(run_id=k,treatment=configs[k]['treatment']) for k,d in primary.items()],ignore_index=True)
all_primary.to_csv(DATA/'primary_evaluation_rows.csv',index=False)


def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(FIG/f'{name}.{ext}',dpi=210,bbox_inches='tight',pad_inches=.12)
 plt.close(fig)

def clean(ax,axis='y'):
 ax.grid(axis=axis,color='#E4E4E4',linewidth=.7,zorder=0)
 ax.set_axisbelow(True)

# Explicitly illustrative capacity maps from pilot replicate 0, seed 42.
for scenario in SCENARIOS:
 capacity,*_=build_environment_maps(scenario,width=10,height=10,seed=42)
 fig,ax=plt.subplots(figsize=(3,3)); ax.imshow(capacity,cmap='viridis',vmin=0,vmax=1);ax.set_xticks([]);ax.set_yticks([])
 for spine in ax.spines.values():spine.set_visible(False)
 save(fig,f'capacity_{scenario}')

fig,ax=plt.subplots(figsize=(.55,2.2))
cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap='viridis'),cax=ax,ticks=[0,.5,1])
cb.ax.tick_params(labelsize=15); cb.outline.set_visible(False)
save(fig,'capacity_scale')

# Population-level organization: one panel per ecology, both network and behavior.
# Every line is a mean over ten independent replicates, without smoothing.
org_rows=[]
for pop in [32,64]:
 fig,axs=plt.subplots(2,3,figsize=(18,7.0),sharex=True)
 for col,scenario in enumerate(SCENARIOS):
  for t in ['R1','R2','R3']:
   for matched,style in [(False,'-'),(True,(0,(4,3)))]:
    rid=f'pilot_v1_{"r0_" if matched else ""}{t.lower()}_mu010'
    d=train[rid]; d=d[(d.scenario==scenario)&(d.population==pop)]
    for ri,metric in enumerate(['visibility_gini','low_extraction_rate']):
     g=d.groupby('time')[metric].agg(['mean','count'])
     assert g['count'].eq(10).all()
     # Visibility time-zero records are available; behavior is undefined at time zero.
     axs[ri,col].plot(g.index,g['mean'],color=COLORS[t],ls=style,lw=2.5 if not matched else 1.8,alpha=1 if not matched else .85)
     for time,r in g.iterrows():org_rows.append({'scenario':scenario,'population':pop,'treatment':t,'matched':matched,'time':time,'metric':metric,'mean':r['mean'],'n':r['count']})
  axs[0,col].set_title(NAMES[scenario],pad=12)
  for ri in range(2):
   ax=axs[ri,col];ax.set_xlim(0,5000);ax.set_xticks([0,2500,5000],['0','2,500','5,000']);clean(ax)
   ax.set_ylim((0,.5) if ri==0 else (0,.6));ax.set_yticks([0,.25,.5] if ri==0 else [0,.2,.4,.6]);
  axs[1,col].set_xlabel('Training step')
 axs[0,0].set_ylabel('Visibility Gini\n0 = equally observed')
 axs[1,0].set_ylabel('Share of low\nextraction actions')
 handles=[Line2D([0],[0],color=COLORS[t],lw=3,label=f'{LABELS[t]} ({t})') for t in COLORS]
 handles += [Line2D([0],[0],color=INK,lw=2.5,label='Adaptive'),Line2D([0],[0],color=INK,lw=2,ls='--',label='Matched random')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.03),ncol=5,frameon=False,fontsize=18)
 fig.subplots_adjust(left=.09,right=.99,bottom=.12,top=.85,wspace=.23,hspace=.23)
 save(fig,f'organization_N{pop}')
pd.DataFrame(org_rows).to_csv(DATA/'organization_trajectories.csv',index=False)

# Paired differences: the same horizontal scale across ecological panels.
ORDER=[('R1',32),('R1',64),('R2',32),('R2',64),('R3',32),('R3',64)]
YS=[6,5,3.7,2.7,1.4,.4]
def forest(ax,d,metric,scale=100,xlim=(-5,5),ylabels=True):
 ax.axvline(0,color='#333333',lw=1.1,zorder=1)
 for (t,pop),y in zip(ORDER,YS):
  r=d[(d.adaptive_label==t)&(d.population==pop)&(d.metric==metric)].iloc[0]
  mean,lo,hi=scale*np.array([r.mean_adaptive_minus_r0,r.bootstrap_ci_low,r.bootstrap_ci_high])
  ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o' if pop==32 else 's',ms=8,
   color=COLORS[t],mfc='white' if pop==32 else COLORS[t],mew=1.6,lw=2,capsize=3,zorder=3)
 ax.set_yticks(YS,[f'{LABELS[t]} · {pop}' for t,pop in ORDER] if ylabels else [])
 ax.set_ylim(-.2,6.65);ax.set_xlim(xlim);clean(ax,axis='x');ax.tick_params(axis='y',length=0)
 ax.spines['left'].set_visible(False)

for metric,name,xlim,xlabel in [
 ('resource_fraction','sustainability',(-5,5),'Difference in resource / capacity (percentage points)'),
 ('reserve_welfare','welfare',(-7.5,7.5),'Difference in reserve welfare (percentage points)'),
 ('wealth_gini','inequality',(-.04,.04),'Difference in final wealth Gini')]:
 fig,axs=plt.subplots(1,3,figsize=(18,5.5))
 for col,scenario in enumerate(SCENARIOS):
  forest(axs[col],summary[summary.scenario==scenario],metric,scale=1 if metric=='wealth_gini' else 100,xlim=xlim,ylabels=col==0)
  axs[col].set_title(NAMES[scenario],pad=14)
 fig.supxlabel(xlabel,y=.02,fontsize=20)
 fig.subplots_adjust(left=.13,right=.99,bottom=.17,top=.87,wspace=.15)
 save(fig,name)

# Detailed read of heterogeneous ecology: full set of six paired comparisons,
# not just the favorable R2/N64 condition. Broader figures remain in the package.
fig,axs=plt.subplots(1,3,figsize=(18,5.5))
for i,(metric,title,xlim,scale) in enumerate([
 ('resource_fraction','Resource / capacity',(-5,5),100),
 ('reserve_welfare','Reserve welfare',(-5,5),100),
 ('wealth_gini','Final wealth inequality',(-.04,.04),1)]):
 forest(axs[i],summary[summary.scenario=='split_high_low'],metric,scale=scale,xlim=xlim,ylabels=i==0)
 axs[i].set_title(title,pad=14)
 axs[i].set_xlabel('Difference (pp)' if scale==100 else 'Difference (Gini units)')
fig.subplots_adjust(left=.13,right=.99,bottom=.18,top=.87,wspace=.15)
save(fig,'split_tradeoffs')

# Raw replicate distributions avoid density estimates with only ten replicates.
ids=['pilot_v1_b0','pilot_v1_s1','pilot_v1_s2','pilot_v1_r1_mu010','pilot_v1_r0_r1_mu010','pilot_v1_r2_mu010','pilot_v1_r0_r2_mu010','pilot_v1_r3_mu010','pilot_v1_r0_r3_mu010']
short=['B0','S1','S2','R1','R0-local','R2','R0-mixed','R3','R0-global']
fig,axs=plt.subplots(2,3,figsize=(18,10),sharex=True,sharey=True)
for ri,pop in enumerate([32,64]):
 for col,scenario in enumerate(SCENARIOS):
  ax=axs[ri,col]
  for j,rid in enumerate(ids):
   v=primary[rid].loc[(scenario,pop),'eval_mean_mean_resource_fraction'].values
   clr=COLORS.get(configs[rid]['treatment'],GREY)
   ax.scatter(v,j+np.linspace(-.14,.14,len(v)),s=30,color=clr,alpha=.65)
   ax.plot(v.mean(),j,'|',ms=18,mew=2.5,color=INK)
  ax.set_yticks(range(9),short);ax.invert_yaxis();ax.set_xlim(0,.5);clean(ax,axis='x')
  ax.set_title(f'{NAMES[scenario]} · N={pop}')
  if ri==1:ax.set_xlabel('Mean resource / capacity')
fig.subplots_adjust(left=.10,right=.99,bottom=.08,top=.94,wspace=.22,hspace=.25)
save(fig,'resource_distributions_all_conditions')

resource=summary[summary.metric=='resource_fraction']
audit['resource_intervals_crossing_zero']=int(((resource.bootstrap_ci_low<=0)&(resource.bootstrap_ci_high>=0)).sum())
audit['resource_comparisons']=len(resource)
audit['resource_mean_effect_range_pp']=[float(resource.mean_adaptive_minus_r0.min()*100),float(resource.mean_adaptive_minus_r0.max()*100)]
audit['checks'].append('All plotting panels retain ecology and population strata; no agent-level pseudoreplication or pooling of independent replicate counts.')
(OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))

# Use the supplied longer deck's slide size, layouts, masters, and theme.
prs=Presentation(REF)
for slide_id in list(prs.slides._sldIdLst):
 prs.part.drop_rel(slide_id.rId);prs.slides._sldIdLst.remove(slide_id)
W=20;H=11.25

def rgb(s):return RGBColor.from_string(s.lstrip('#'))
def text(slide,x,y,w,h,content,size=28,bold=False,color=INK,align=PP_ALIGN.LEFT):
 sh=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
 tf=sh.text_frame;tf.clear();tf.word_wrap=True
 tf.margin_left=0;tf.margin_right=0;tf.margin_top=0;tf.margin_bottom=0
 for i,line in enumerate(content.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.alignment=align
  p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color)
  p.space_after=Pt(4)
 return sh

def rect(s,x,y,w,h,fill='#F5F5F5',line=None,radius=False):
 sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
 sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
 sh._element.spPr.append(OxmlElement('a:effectLst'))
 if line:sh.line.color.rgb=rgb(line)
 else:sh.line.fill.background()
 return sh

def arrow(s,x1,y1,x2,y2,color=GREY,width=2.2,dash=False):
 sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2))
 sh.line.color.rgb=rgb(color);sh.line.width=Pt(width)
 end=OxmlElement('a:tailEnd');end.set('type','triangle');sh._element.spPr.get_or_add_ln().append(end)
 if dash:
  from pptx.enum.dml import MSO_LINE_DASH_STYLE
  sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
 return sh

def circle(s,x,y,d,fill=BLUE):
 sh=s.shapes.add_shape(MSO_SHAPE.OVAL,Inches(x),Inches(y),Inches(d),Inches(d));sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill);sh.line.fill.background();return sh

def base(title,num,sub=None):
 s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb('#FFFFFF')
 text(s,.9,.63,18.15,.95,title,40.5,True)
 if sub:text(s,.95,1.73,18.1,.85,sub,27,color=GREY)
 text(s,.95,10.79,15,.28,'COGNITIVE TOOLS  /  SUPERVISOR DATA ROUND  /  PILOT v1',13,color=GREY)
 text(s,18.0,10.72,1.0,.35,f'{num} / 7',15,color=GREY,align=PP_ALIGN.RIGHT)
 return s

def picture(s,name,x,y,w,h=None):
 from PIL import Image
 path=FIG/f'{name}.png'
 if h is None:return s.shapes.add_picture(str(path),Inches(x),Inches(y),width=Inches(w))
 iw,ih=Image.open(path).size;scale=min(w/iw,h/ih);pw,ph=iw*scale,ih*scale
 return s.shapes.add_picture(str(path),Inches(x+(w-pw)/2),Inches(y+(h-ph)/2),width=Inches(pw),height=Inches(ph))

def note(s,t):s.notes_slide.notes_text_frame.text=t

def node(s,x,y,w,h,title,body,color=BLUE):
 rect(s,x,y,w,h,fill='#F5F5F5');rect(s,x,y,.075,h,fill=color)
 text(s,x+.23,y+.2,w-.45,.7,title,30,True)
 text(s,x+.23,y+1.02,w-.45,h-1.13,body,25)

# 1. RQ, individual -> population -> outcome.
s=base('How do local observations become collective outcomes?',1)
text(s,1.0,2.05,17.8,1.6,'Can changing whom agents observe produce population-level\npatterns that sustain a shared renewable resource?',37,True)
node(s,1,4.45,5.15,2.65,'Individual observations','What do my selected\ninformation sources do?',BLUE)
node(s,7.45,4.45,5.15,2.65,'Population-level patterns','Who becomes visible?\nWhich actions become common?',TEAL)
node(s,13.9,4.45,5.15,2.65,'Collective outcomes','Resource sustainability\nWelfare and inequality',GOLD)
arrow(s,6.32,5.65,7.25,5.65);arrow(s,12.76,5.65,13.7,5.65)
text(s,1,7.65,2.5,.6,'HYPOTHESIS',21,True,color=BLUE)
text(s,1,8.35,18,1.35,'Prediction-error-driven rewiring produces patterns that improve\nsustainability beyond the same amount of random network turnover.',30)
note(s,'Purpose: request feedback on a focused question and mechanism, not sell a positive result. The hypothesis is directional and is not established by this pilot. Collective organization means measured population patterns in visibility and extraction behavior; it does not mean conscious coordination. Ecology and search scope are conditions, welfare and inequality are consequences. The design is inspired by Schrama et al. (2025), https://doi.org/10.1016/j.isci.2025.112831, and Oh & Schauf (2025), https://doi.org/10.1038/s41598-025-23634-3. No published figure or model result is reproduced here.')

# 2. Mechanism with an editable local network illustration and feedback loop.
s=base('One feedback loop connects agents, networks, and ecology',2,'Agents stay in place. Information links can change; resources do not travel along them.')
# Simple source->observer network, explicitly schematic.
positions=[(1.9,3.6),(4.0,3.6),(1.9,5.65),(4.0,5.65),(3.0,4.6)]
for i,j in [(0,4),(1,4),(2,4)]:arrow(s,positions[i][0]+.25,positions[i][1]+.25,positions[j][0]+.25,positions[j][1]+.25,color='#A7A7A7')
arrow(s,4.25,5.9,3.25,4.85,color=TEAL,dash=True)
for k,(x,y) in enumerate(positions):circle(s,x,y,.55,TEAL if k==4 else BLUE)
text(s,1.15,7.15,4.85,1.1,'Observe previous actions\nof a few selected sources',27,True,align=PP_ALIGN.CENTER)
text(s,1.25,8.47,4.7,.9,'Schematic: arrows carry\ninformation to the focal agent.',20,color=GREY,align=PP_ALIGN.CENTER)
node(s,7.0,3.1,5.15,2.3,'Learn extraction','Local resource state +\nobserved peer behavior',BLUE)
node(s,13.7,3.1,5.15,2.3,'Resource renewal','Low or high extraction;\nrenewal and spatial coupling',GOLD)
node(s,13.7,6.6,5.15,2.55,'Predict and compare','Expected vs. observed\npeer behavior → local surprise',TEAL)
node(s,7.0,6.6,5.15,2.55,'Replace a source','If surprise is high:\nlocal or global search',TEAL)
arrow(s,5.35,4.4,6.75,4.4);arrow(s,12.38,4.2,13.45,4.2)
arrow(s,16.2,5.65,16.2,6.34,color=GOLD)
arrow(s,13.45,7.9,12.4,7.9,color=TEAL)
arrow(s,6.75,7.9,5.85,7.9,color=TEAL);arrow(s,5.85,7.9,5.85,5.35,color=TEAL)
text(s,7.0,9.72,11.8,.6,'Reward = own realized harvest. Sustainability is measured, not rewarded.',23,True)
note(s,'Conceptual diagram; arrows summarize the feedback rather than the exact computation sequence. Training order: form state from current ecology and previous source actions; choose extraction; ecological transition; observe current actions through pre-rewiring graph; prediction error; optional rewiring; forecast update using pre-rewiring observation; next-state and Q update. Error is social surprise, not a truth/correctness label or an assessment of source competence. Current-source removal and replacement are random conditional on an observer being triggered. Agents do not move and do not share Q tables. Information arrows source->observer. Local search uses sources of sources with global fallback. Related work differs: Oh & Schauf use correctness and DeGroot consensus, whereas this model uses surprise and Q-learning.')

# 3. Controls, maps and protocol.
s=base('The key test holds the amount of rewiring constant',3,'Adaptive rewiring versus matched random turnover, within the same ecological condition.')
text(s,1,2.8,8.9,.6,'Three reference conditions',28,True)
for y,label,body in [(3.7,'B0','Ecological learning only'),(4.55,'S1','Fixed random observation network'),(5.4,'S2','Fixed network with uneven visibility')]:
 text(s,1.05,y,1,.55,label,29,True);text(s,2.3,y,7.5,.7,body,27)
rect(s,10.55,2.85,8.4,3.45,fill='#F3F6F7')
text(s,10.9,3.1,7.6,.65,'R1 / R2 / R3   ↔   matched R0',31,True)
text(s,10.9,4.05,7.6,1.6,'Local / mixed / global search\nSame event counts at every checkpoint\nDifferent rule for which observers rewire',26)
for i,scenario in enumerate(SCENARIOS):
 x=1.05+3.12*i;picture(s,f'capacity_{scenario}',x,7.0,1.9,1.9)
 text(s,x-.18,9.07,2.75,.58,NAMES[scenario],23,True)
picture(s,'capacity_scale',9.55,7.0,.78,1.95)
text(s,1.0,9.82,9.3,.55,'Capacity maps · replicate 0 · common scale 0–1',19,color=GREY)
text(s,11.0,7.02,7.9,2.7,'3 ecologies × 2 populations (32, 64)\n10 replicates per condition\n5,000 training + 1,000 evaluation steps\nFresh ecology; learned policies and graphs frozen',27)
note(s,'Pilot v1: 9 runs x 3 scenarios x 2 populations x 10 replicates = 540 training conditions. Same seed family, ecological and learning settings; adaptive mu=.10, threshold=.25, every 50 steps, social k=4, BA m=2; theta 0,.25,1. R0 event counts match the paired adaptive run at each scenario/population/replicate/time checkpoint. SHA-256 source schedules verified. Primary analysis uses q_learning/fresh_reset/terminal graph, not continuation or reset-initial graph. Capacity maps show seed 42, replicate 0, constructed from the frozen scenario code; maps are illustrative, with shared viridis scale 0-1. Source files: pilot_v1_*/config.json; original commit b0340b10dd33499a714af5c22ab124b05f680a3a, clean. S2 varies attention capacity as well as visibility.')

# 4. Organizational patterns and action prevalence.
s=base('Local search concentrates attention—even with random turnover',4,'Population-level patterns during training · N = 64 · all three ecologies')
picture(s,'organization_N64',.6,2.55,18.8,6.55)
text(s,1.0,9.33,18.0,.65,'Organization appears in both adaptive and matched-random networks.',29,True)
text(s,1.0,10.06,18.0,.4,'Lines: means of 10 replicates, unsmoothed; descriptive trends, not uncertainty intervals. N=32 companion supplied.',19,color=GREY)
note(s,'This slide operationalizes collective organization as visibility concentration and prevalence of low-extraction actions. All six rewiring treatments shown, all three ecologies, N=64. Solid=adaptive; dashed=its event-count-matched R0; blue=local, teal=mixed, ochre=global. Each line is a mean of 10 independent replicate trajectories; no smoothing and no inference from line separation. The full N=32 figure is supplied to avoid presenting one population as the whole pilot. Local search visibly concentrates attention, and R0 often tracks the same pattern. This is not evidence that the error trigger uniquely organizes the population. Low extraction remains below one half in the displayed mean trajectories. Source: pilot_v1_*/data/training_timeseries.csv. Design rationale: Oh & Schauf (2025), Fig. 6 separates structure from performance over time; our metrics and mechanism differ. https://pmc.ncbi.nlm.nih.gov/articles/PMC12618610/')

# 5. Full causal comparison, not a cherry-picked example.
s=base('Adaptive rewiring has no clear general sustainability advantage',5,'Difference from matched random turnover · every ecology, search scope, and population')
picture(s,'sustainability',.65,2.9,18.5,5.85)
text(s,1,8.85,18,.65,'15 of 18 intervals cross zero; the direction also changes across conditions.',29,True)
text(s,1,9.66,18,.8,'Right of zero = more resource under adaptive rewiring. Rows show search scope and population.\n10 paired replicates; 95% bootstrap intervals (2,000 resamples), exploratory and not multiplicity-adjusted.',20,color=GREY)
note(s,'Outcome is the replicate-level mean resource/capacity during 1000-step fresh-reset evaluation with frozen terminal graphs and frozen Q policies. Differences are adaptive minus its own exact R0, paired by scenario, population, and replicate. Three panels share the same x range [-5,+5] percentage points. Circle/open=N32; square/filled=N64; row labels are explicit. All 18 comparisons shown. 15 intervals cross zero; this is uncertainty, not proof of equivalence. Three intervals exclude zero: local/split/N32 negative, mixed/split/N64 positive, global/split/N32 positive. They are exploratory, unadjusted intervals, not three confirmed discoveries. All point estimates span -2.105 to +1.497 pp. Underlying means and bootstrap intervals were independently reproduced from raw records. Archive analysis seed 1729, 2000 percentile-bootstrap resamples. Source: paired_adaptive_minus_r0_summary.csv. Replicate outcome distributions are supplied separately, following the distribution-preserving principle illustrated by Schrama et al. (2025), Fig. 2, https://pmc.ncbi.nlm.nih.gov/articles/PMC12226385/. This pilot is not evidence of bistability.')

# 6. Consequences, deliberately explicit subset and sign semantics.
s=base('A resource gain is not a complete welfare story',6,'Closer look: split high / low ecology · all six adaptive-versus-R0 comparisons')
picture(s,'split_tradeoffs',.65,2.85,18.5,5.8)
text(s,1,8.75,18,.7,'Mixed search, N=64: resource +1.02 pp; reserve welfare +1.62 pp.',29,True)
text(s,1,9.49,18,.9,'Its wealth-inequality interval crosses zero. More resource does not establish a fairer outcome.\nPositive = more resource / welfare, but more inequality. Same 10 pairs and 95% bootstrap method as slide 5.',20,color=GREY)
note(s,'This is a transparent close-up of split_high_low, the ecology where the resource effects change sign; all six scope/population comparisons are shown, not only the favorable one. All-ecology welfare and final-wealth-Gini figures are supplied in figures/. Reserve welfare is energy/capacity, averaged during evaluation; wealth Gini is measured at the final evaluation step. R2/N64 resource effect .010238 (95% CI .004494,.016077), reserve welfare .016240 (.003325,.028887), final wealth Gini .001631 (-.008370,.011563). Percentage points are differences of 0-1 fractions, not relative percent changes. Inequality uses Gini units; negative is less unequal. Outcome dimensions need separate interpretations; the intervals are exploratory and not multiplicity-adjusted. See data/paired_summary.csv and data_audit.json.')

# 7. Synthesis and supervisor discussion.
s=base('The project can stay focused on one micro-to-macro test',7,'Individual observation → population patterns → sustainability, welfare, and inequality')
node(s,1.0,3.0,5.55,3.75,'What the pilot shows','Local search changes\nwho becomes visible.\n\nRandom turnover can produce\na similar concentration.',BLUE)
node(s,7.22,3.0,5.55,3.75,'What remains unresolved','Does the error trigger create\na distinct collective pattern?\n\nDoes that pattern improve\nresource outcomes?',TEAL)
node(s,13.45,3.0,5.55,3.75,'What the next test needs','Specify the expected\npopulation pattern.\n\nKeep matched R0 and\nreplicate-level uncertainty.',GOLD)
text(s,1,7.55,17.9,.7,'Feedback I want from you',32,True)
text(s,1,8.4,17.8,1.65,'Is the question narrow enough?\nIs “collective organization” defined clearly enough to test?\nWhich link in the mechanism needs the strongest evidence?',29)
note(s,'Closing discussion, not a claim of confirmation. Keep project scope to renewable ecology, independent extraction learning, limited peer observation, and rewiring. The pilot supports a structural search-scope pattern but does not establish an adaptive sustainability advantage. Ask the supervisor to sharpen a falsifiable prediction about population-level behavior/visibility and its relationship to resources. Avoid scope expansion into technology, inheritance, trade, misinformation, or ecological shocks in this seven-slide argument. Larger confirmatory runs should follow a fixed hypothesis and declared primary comparison, not selection of the most favorable pilot cell. Full source and figure-design rationale: figure_review.md; reproducible build: build_data_round.py; audit: data_audit.json.')

assert len(prs.slides)==7
for i,s in enumerate(prs.slides,1):
 for sh in s.shapes:
  assert sh.left>=0 and sh.top>=0 and sh.left+sh.width<=prs.slide_width+1000 and sh.top+sh.height<=prs.slide_height+1000,(i,sh.name)
prs.save(OUT/'cognitive_tools_supervisor_data_round_7_slides.pptx')
print(json.dumps({'slides':len(prs.slides),'audit':audit['checks'],'zero_crossing':audit['resource_intervals_crossing_zero']},indent=2))
