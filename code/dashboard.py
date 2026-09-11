"""Review view of the supplied dashboard, connected to regenerated real payloads.
No synthetic rows or mock importances. No hosted-model calls.
"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

st.set_page_config(page_title='Institutional Performance & Diagnostic Hub',layout='wide',initial_sidebar_state='expanded')
st.markdown('<style>.block-container{padding-top:1rem;padding-bottom:.4rem}h1{font-size:1.65rem!important}h2{font-size:1.15rem!important}h3{font-size:.95rem!important}p{margin-bottom:.15rem}hr{margin:.4rem 0}[data-testid="stMetricValue"]{font-size:1.45rem}div[data-testid="stVerticalBlock"]{gap:.15rem}[data-testid="stAlert"]{padding:.4rem}[data-testid="stAlert"] p{font-size:.85rem}[data-testid="stCaptionContainer"] p{font-size:.72rem}header{visibility:hidden}</style>',unsafe_allow_html=True)
payloads=json.loads((Path(__file__).parent.parent/'data/full_fit.json').read_text())['payloads']
bycode={p['course_code']:p for p in payloads}
# Keep real payloads intact; use a display-only pseudonym mapping.
codes=[13103,5001167]+[c for c in sorted(bycode) if c not in [13103,5001167]]
aliases={c:('Program A' if i==0 else 'Program B' if i==1 else f'Program {i+1:03d}') for i,c in enumerate(codes)}
st.sidebar.subheader('Language / Idioma')
st.sidebar.write('English')
st.sidebar.divider();st.sidebar.subheader('ENAMED Audit Panel')
st.sidebar.caption('Course-level indicators from ENAMED 2025 assessment microdata.')
selected=st.sidebar.selectbox('Select program',codes,format_func=aliases.get)
p=bycode[selected];alias=aliases[selected]
st.sidebar.divider();st.sidebar.subheader('National baseline')
st.sidebar.metric('Median of course means',f"{p['national_median']:.2f}")
st.sidebar.info('Program identifiers are masked for double-anonymous review. The indicators are computed from real course-level data. Aggregation alone does not guarantee anonymity.')
st.title('Institutional Performance & Diagnostic Hub')

c1,c2,c3=st.columns([1,1,1.35]);c1.metric('Program',alias);c2.metric('Mean exam score',f"{p['avg_score']:.2f}",f"{p['avg_score']-p['national_median']:+.2f} vs median")
c3.metric('Predicted tier',p['predicted_tier']);c3.caption(f"Model probability: {p['confidence_pct']:.1f}% (in-sample; not calibrated)")
st.divider();left,right=st.columns([1.35,1])
with left:
 st.subheader('Local diagnostic prioritization')

 d=pd.DataFrame(p['contributions'])
 fig,ax=plt.subplots(figsize=(8,3.5));fig.patch.set_facecolor('white')
 ax.barh(range(len(d)),d['impact'],color=cm.viridis([i/(len(d)-1) for i in range(len(d))]),height=.62)
 ax.set_yticks(range(len(d)),[f"{r['label']} ({r['feature']})" for _,r in d.iterrows()],fontsize=10)
 ax.invert_yaxis();ax.set_xlim(0,max(d['impact'])*1.18)
 for i,v in enumerate(d['impact']):ax.text(v+.06,i,f'{v:.2f}',va='center',fontsize=10)
 ax.set_xlabel('Local diagnostic score',fontsize=10);ax.tick_params(axis='y',length=0)
 ax.spines[['top','right','left']].set_visible(False);fig.tight_layout();st.pyplot(fig);plt.close(fig)
with right:
 st.subheader('Questionnaire factors for inspection')
 st.caption('Descriptive labels applied by predefined rules.')
 for c in p['contributions']:
  if c['sentiment']=='strength':st.success(f"**{c['label']} ({c['feature']})** — {c['value_pct']:.1f}%")
 for c in p['contributions']:
  if c['sentiment']=='friction':st.warning(f"**{c['label']} ({c['feature']})** — {c['value_pct']:.1f}%. Inspect this response pattern with the coordinator.")
st.divider()
# Same numeric formatting and ranking as local_explanation in the supplied package.
fr=sorted([c for c in p['contributions'] if c['sentiment']=='friction'],key=lambda c:c['impact'],reverse=True)[:3]
stg=sorted([c for c in p['contributions'] if c['sentiment']=='strength'],key=lambda c:c['impact'],reverse=True)[:2]
delta=round(p['avg_score']-p['national_median'],2)
text=f"{alias} was classified as \"{p['predicted_tier']}\" ({p['confidence_pct']:.1f}%). The course mean ({p['avg_score']:.2f}) is {abs(delta):.2f} point(s) {'above' if delta>=0 else 'below'} the national median ({p['national_median']:.2f})."
if stg:text+=' Key strengths: '+', '.join(f"{c['label']} ({c['impact']:.2f})" for c in stg)+'.'
with st.expander('Deterministic explanation — audited values'):
 st.info(text)
if False and fr:st.caption('Action items (friction points): '+'; '.join(f"{c['label']} ({c['feature']}): {c['value_pct']:.1f}% — impact {c['impact']:.2f}" for c in fr)+'.')


