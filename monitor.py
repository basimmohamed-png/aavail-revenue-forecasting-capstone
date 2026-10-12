"""Delayed-label performance monitoring. Never retrains using holdout targets."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
from core import ingest,daily,predict,log_event

def evaluate(model_dir,train_dir,production_dir,countries,output,log_dir):
    history=ingest(train_dir);production=ingest(production_dir)
    combined=pd.concat([history,production],ignore_index=True)
    results=[];output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for country in countries:
        bundle=joblib.load(Path(model_dir)/(country+'.joblib'))
        ts=daily(combined,country)
        start=pd.Timestamp(bundle['training_end'])+pd.Timedelta(days=1)
        stop=ts.index.max()-pd.Timedelta(days=29)
        rows=[]
        for date in pd.date_range(start,stop):
            p=predict(bundle,ts,str(date.date()))
            actual=float(ts.loc[date:date+pd.Timedelta(days=29),'revenue'].sum())
            rows.append({'date':str(date.date()),'actual':actual,**p})
        frame=pd.DataFrame(rows)
        if frame.empty:raise ValueError('No fully observed target windows')
        frame.to_csv(output/(country+'_production.csv'),index=False)
        metrics={'country':country,'origins':len(frame),'selected':bundle['selected'],
                 'mae':float(mean_absolute_error(frame.actual,frame.prediction)),
                 'rmse':float(np.sqrt(mean_squared_error(frame.actual,frame.prediction))),
                 'baseline_mae':float(mean_absolute_error(frame.actual,frame.baseline)),
                 'novelty_rate':float(frame.novelty.mean()),'model_version':bundle['version']}
        # KPI: absolute revenue planning error relative to observed revenue.
        metrics['revenue_wape']=float(np.abs(frame.actual-frame.prediction).sum()/np.abs(frame.actual).sum())
        # Overlapping 30-day windows are correlated: also report independent blocks.
        blocks=frame.iloc[::30]
        metrics['nonoverlap_origins']=len(blocks)
        metrics['nonoverlap_wape']=float(np.abs(blocks.actual-blocks.prediction).sum()/np.abs(blocks.actual).sum())
        selected=next(r for r in bundle['metrics'] if r['model']==bundle['selected'])
        metrics['mae_alert']=bool(metrics['mae']>1.5*selected['mae'])
        log_event(log_dir,{'type':'monitor',**metrics});results.append(metrics)
    (output/'monitoring.json').write_text(json.dumps(results,indent=2))
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',default='artifacts');p.add_argument('--train-dir',default='data/cs-train');p.add_argument('--production-dir',default='data/cs-production');p.add_argument('--countries',nargs='+',default=['all','united_kingdom','usa']);p.add_argument('--output',default='reports');p.add_argument('--log-dir',default='logs')
    a=p.parse_args();print(json.dumps(evaluate(a.model_dir,a.train_dir,a.production_dir,a.countries,a.output,a.log_dir),indent=2))
