"""Revenue forecasts: next 30 calendar days, information strictly before origin."""
from pathlib import Path
import json, re, os, uuid, threading
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

_LOCK = threading.Lock()
FEATURES = ['previous_7','previous_14','previous_28','previous_70','previous_year','recent_purchases','recent_views']

def ingest(directory):
    plain=sorted(Path(directory).glob('*.json'))
    files=plain if plain else sorted(Path(directory).glob('*.json.gz'))
    if not files: raise ValueError('No JSON input files')
    frames=[]
    for path in files:
        d=pd.read_json(path).rename(columns={'StreamID':'stream_id','TimesViewed':'times_viewed','total_price':'price'})
        required={'country','customer_id','day','invoice','month','price','stream_id','times_viewed','year'}
        if not required.issubset(d.columns): raise ValueError(f'Missing columns: {path.name}')
        d['date']=pd.to_datetime(d[['year','month','day']],errors='raise')
        for c in ['price','times_viewed']: d[c]=pd.to_numeric(d[c],errors='raise')
        if not np.isfinite(d[['price','times_viewed']].to_numpy()).all(): raise ValueError('Nonfinite input')
        frames.append(d)
    return pd.concat(frames,ignore_index=True).sort_values('date')

def daily(d,country='all'):
    dates=pd.date_range(d.date.min(),d.date.max(),freq='D')
    if country!='all':
        names={str(c).lower().replace(' ','_'):c for c in d.country.unique()}
        if country not in names: raise ValueError('Unknown country')
        d=d[d.country==names[country]]
    return d.groupby('date').agg(revenue=('price','sum'),purchases=('invoice','size'),views=('times_viewed','sum')).reindex(dates,fill_value=0)

def features(ts,origin):
    origin=pd.Timestamp(origin)
    if origin < ts.index.min()+pd.Timedelta(days=70) or origin > ts.index.max()+pd.Timedelta(days=1):
        raise ValueError('Date outside supported range (70 days history required)')
    r={}
    for n in [7,14,28,70]: r[f'previous_{n}']=float(ts.loc[(ts.index>=origin-pd.Timedelta(days=n))&(ts.index<origin),'revenue'].sum())
    start=origin-pd.Timedelta(days=365)
    r['previous_year']=float(ts.loc[(ts.index>=start)&(ts.index<start+pd.Timedelta(days=30)),'revenue'].sum())
    past=ts.loc[(ts.index>=origin-pd.Timedelta(days=30))&(ts.index<origin)]
    r['recent_purchases']=float(past.purchases.mean());r['recent_views']=float(past.views.mean())
    return pd.DataFrame([r],columns=FEATURES)

def supervised(ts):
    dates=ts.index[70:-29]
    X=pd.concat([features(ts,x) for x in dates],ignore_index=True)
    y=np.array([ts.loc[x:x+pd.Timedelta(days=29),'revenue'].sum() for x in dates])
    return X,y,dates

def log_event(root,event):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    line=json.dumps(event,allow_nan=False)+'\n'
    with _LOCK:
        with (root/'events.jsonl').open('a') as f: f.write(line)

def read_logs(root,limit=100):
    p=Path(root)/'events.jsonl'
    if not p.exists(): return []
    with p.open() as f: return [json.loads(l) for l in f if l.strip()][-limit:]

def save_model(path,bundle):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        joblib.dump(bundle,tmp);os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)

def train_series(ts):
    X,y,dates=supervised(ts)
    if len(y)<160: raise ValueError('Insufficient training history')
    cut=int(len(y)*.75)
    # 30-day targets overlap: purge origin dates whose targets meet validation.
    train=np.arange(cut-30);valid=np.arange(cut,len(y))
    candidates={'random_forest':RandomForestRegressor(n_estimators=60,max_depth=8,min_samples_leaf=5,random_state=42,n_jobs=1),
                'gradient_boosting':HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=15,l2_regularization=2,random_state=42)}
    rows=[];pred={}
    baseline=X.iloc[valid].previous_28.to_numpy()*30/28
    pred['baseline']=baseline
    for name,m in candidates.items():
        m.fit(X.iloc[train],y[train]);pred[name]=np.maximum(0,m.predict(X.iloc[valid]))
    for name,p in pred.items():
        rows.append({'model':name,'mae':float(mean_absolute_error(y[valid],p)), 'rmse':float(np.sqrt(mean_squared_error(y[valid],p)))})
    best=min(rows,key=lambda a:a['mae'])['model']
    model=None if best=='baseline' else candidates[best].fit(X,y)
    return {'model':model,'selected':best,'version':uuid.uuid4().hex,'training_end':str(ts.index.max().date()),
            'feature_min':X.min().to_dict(),'feature_max':X.max().to_dict(),'metrics':rows}, pd.DataFrame({'date':dates[valid],'actual':y[valid],**pred})

def predict(bundle,ts,date):
    date=pd.Timestamp(date)
    # Prevent retrospective predictions from a model fitted on later observations.
    if date<=pd.Timestamp(bundle['training_end']): raise ValueError('Prediction date must follow model training end')
    X=features(ts,date)
    value=float(X.previous_28.iloc[0]*30/28) if bundle['model'] is None else float(bundle['model'].predict(X)[0])
    novelty=any(X[c].iloc[0]<bundle['feature_min'][c] or X[c].iloc[0]>bundle['feature_max'][c] for c in FEATURES)
    return {'prediction':max(0,value),'model_version':bundle['version'],'novelty':bool(novelty),'horizon_days':30,'baseline':float(X.previous_28.iloc[0]*30/28)}
