"""Reproduce EDA, time-respecting model comparison and production evaluation."""
from pathlib import Path
import json,hashlib,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from core import ingest,daily,train_series,save_model
from monitor import evaluate

out=Path('reports');out.mkdir(exist_ok=True)
d=ingest('data/cs-train')
country_revenue=d.groupby('country').price.sum().sort_values(ascending=False)
country_revenue.to_csv(out/'country_revenue.csv')
fig,axs=plt.subplots(2,1,figsize=(10,7))
ts=daily(d)
axs[0].plot(ts.index,ts.revenue);axs[0].set(title='Daily revenue: all countries',ylabel='Revenue (dataset currency)')
country_revenue.head(10).plot.bar(ax=axs[1]);axs[1].set(title='Training revenue by country',ylabel='Revenue (dataset currency)')
fig.tight_layout();fig.savefig(out/'eda.png',dpi=150);plt.close(fig)
summary=[]
for c in ['all','united_kingdom','usa']:
    series=daily(d,c);bundle,comparison=train_series(series)
    save_model(Path('artifacts')/(c+'.joblib'),bundle)
    comparison.to_csv(out/(c+'_validation.csv'),index=False)
    for r in bundle['metrics']:summary.append({'country':c,**r,'selected':bundle['selected']})
metrics=pd.DataFrame(summary);metrics.to_csv(out/'model_comparison.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,(country,g) in zip(axs,metrics.groupby('country',sort=False)):
    ax.bar(g.model,g.mae,color=['#9198a1','#3976b7','#30a080']);ax.tick_params(axis='x',rotation=25);ax.set(title=country,ylabel='Validation MAE')
fig.suptitle('Purged chronological validation: learned models versus baseline');fig.tight_layout();fig.savefig(out/'baseline_comparison.png',dpi=150);plt.close(fig)
monitor=evaluate('artifacts','data/cs-train','data/cs-production',['all','united_kingdom','usa'],out,'logs')
frame=pd.read_csv(out/'all_production.csv')
fig,ax=plt.subplots(figsize=(10,4));date=pd.to_datetime(frame.date)
for col in ['actual','prediction','baseline']:ax.plot(date,frame[col],label=col)
ax.set(title='Untouched production period: next 30-day revenue',ylabel='Revenue');ax.legend();fig.tight_layout();fig.savefig(out/'production_comparison.png',dpi=150);plt.close(fig)
fingerprint=hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('data/cs-train').glob('*.json*'))).encode()).hexdigest()
info={'records':len(d),'start':str(d.date.min().date()),'end':str(d.date.max().date()),'span_days':(d.date.max()-d.date.min()).days,'training_sha256':fingerprint}
(out/'data_summary.json').write_text(json.dumps(info,indent=2))
html='''<!doctype html><html><head><meta charset="utf-8"><title>AAVAIL final report</title><style>body{max-width:1050px;margin:40px auto;font:16px/1.6 Arial;color:#233246}h1,h2{color:#163c5b}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border:1px solid #bbb}.note{padding:16px;background:#eef4fa}</style></head><body><h1>AAVAIL Revenue Forecasting and Model Deployment</h1><p>Prepared for peer review — Basim Mohamed</p><h2>Business objective and data</h2><p>Forecast the total revenue of the next 30 days for a selected country or all countries combined. Revenue planning error is the business metric; MAE, RMSE and WAPE evaluate forecasting performance. Currency is not documented in the provided schema, so values retain dataset units.</p>'''
html+=f'<p>Training records: {len(d):,}; dates {info["start"]} to {info["end"]}, spanning {info["span_days"]} days. United Kingdom has the highest training revenue ({country_revenue.iloc[0]:,.2f}). Daily zero values are filled for absent transaction dates. Negative prices are retained as possible credits; no artificial revenue clipping is applied to targets.</p><img src="eda.png">'
html+='''<h2>Methods and leakage control</h2><p>Features use only data strictly before the forecast origin: previous 7, 14, 28 and 70 days of revenue, matching prior-year 30-day revenue, and previous 30-day mean purchase counts and views. We compare a trailing 28-day revenue baseline scaled to 30 days, random forest, and histogram gradient boosting. The first 75% of supervised origins forms the candidate training period; the last 25% is validation. A 30-origin gap purges overlapping target windows. The lowest validation MAE selects a model, including the baseline if it wins. The selected model is then refitted using fully observed training targets.</p><p>This deliberately improves the supplied reference's shuffled split: neighboring 30-day targets share observations. Results from that reference are therefore not a fair gold-standard accuracy benchmark. We preserve the business target and feature intent, while using a chronological benchmark. No supplied fitted gold-standard model or prediction file exists in the reference repository.</p><h2>Validation comparison</h2>'''
html+=metrics.round(2).to_html(index=False)+'<img src="baseline_comparison.png"><h2>Production monitoring</h2>'
html+=pd.DataFrame(monitor).drop(columns='model_version').round(4).to_html(index=False)+'<img src="production_comparison.png">'
html+='''<p>The production JSON files are held out from fitting and selection. Forecasts at later origins use observations available before that origin, with the trained model fixed. A label is evaluated only after all 30 target days are observed. WAPE measures absolute revenue planning error relative to actual revenue; non-overlapping origins are also reported because rolling targets are correlated. Feature-range novelty flags unusual inputs; they are prompts to investigate, not proof of concept drift. The monitoring script flags MAE above 1.5 times validation MAE. This threshold is illustrative and must be calibrated operationally.</p><h2>Deployment and tests</h2><p>Flask exposes POST /train, POST /predict, GET /logfile and GET /health. Predictions include country, date, horizon, baseline, model version and novelty; JSONL logs record request identifiers, timestamp and runtime. Training writes models atomically; a lock prevents concurrent training in the configured single worker. Tests use TemporaryDirectory fixtures for models, synthetic data and logs, avoiding production writes. Dockerfile bundles application, models, data, tests and reports; tests run during image build. Gunicorn uses one worker with four threads.</p><div class="note"><strong>Docker validation limitation:</strong> Docker is not installed in the authoring environment. The image has not been built or run here. Run verify_docker.sh before claiming the containerization rubric criterion passes. The local Flask application and tests are independently verified.</div><h2>Limitations and next steps</h2><p>Only three scopes ship pretrained: all, United Kingdom and USA. The train endpoint supports other countries with sufficient data. Train runs synchronously; for scale, move training to a queue, shared storage, multi-process log collection and external scheduling. No authentication is included: run this educational service locally, not on a public network. Logs and models should be mounted persistently. Re-running analyze.py regenerates the models and report; automatic cadence is not implemented. Increase temporal validation folds, calibrate alert thresholds, compare specialized forecasting approaches, and monitor country-level business impact before operational adoption.</p></body></html>'''
(out/'Final_Report.html').write_text(html)
print(metrics.to_string(index=False));print(json.dumps(monitor,indent=2))
