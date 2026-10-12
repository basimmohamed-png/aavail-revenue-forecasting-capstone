from pathlib import Path
import os,time,uuid,threading,re
import joblib
from flask import Flask,request,jsonify
from core import ingest,daily,train_series,predict,save_model,log_event,read_logs

def create_app(config=None):
    a=Flask(__name__)
    a.config.update(DATA_DIR=os.getenv('DATA_DIR','data/cs-train'),OBS_DIR=os.getenv('OBS_DIR','data/cs-production'),MODEL_DIR=os.getenv('MODEL_DIR','artifacts'),LOG_DIR=os.getenv('LOG_DIR','logs'),MAX_CONTENT_LENGTH=65536)
    a.config.update(config or {})
    lock=threading.Lock();cache={}
    def get_data():
        if 'train' not in cache: cache['train']=ingest(a.config['DATA_DIR'])
        return cache['train']
    def parse():
        b=request.get_json(silent=True)
        if not isinstance(b,dict): raise ValueError('JSON object required')
        return b
    @a.errorhandler(ValueError)
    def invalid(e): return jsonify(error=str(e)),400
    @a.errorhandler(413)
    def large(e): return jsonify(error='Request too large'),413
    @a.get('/health')
    def health(): return jsonify(status='ok')
    @a.post('/train')
    def train():
        b=parse();country=b.get('country','all')
        if not isinstance(country,str): raise ValueError('country must be a string')
        ts=daily(get_data(),country)
        if not lock.acquire(blocking=False): return jsonify(error='Training already running'),409
        try:
            start=time.perf_counter();bundle,_=train_series(ts)
            save_model(Path(a.config['MODEL_DIR'])/(country+'.joblib'),bundle)
            log_event(a.config['LOG_DIR'],{'type':'train','country':country,'model_version':bundle['version'],'runtime':time.perf_counter()-start,'metrics':bundle['metrics']})
            return jsonify(country=country,selected=bundle['selected'],model_version=bundle['version'],metrics=bundle['metrics'])
        finally: lock.release()
    @a.post('/predict')
    def forecast():
        b=parse();country=b.get('country');date=b.get('date')
        if not isinstance(country,str) or not isinstance(date,str): raise ValueError('country and date are required strings')
        ts=daily(get_data(),country)
        path=Path(a.config['MODEL_DIR'])/(country+'.joblib')
        if not path.exists(): return jsonify(error='Model not trained'),404
        obs=Path(a.config['OBS_DIR'])
        if obs.exists():
            # Production observations only supply lag features; training stays fixed.
            import pandas as pd
            if 'combined' not in cache: cache['combined']=pd.concat([get_data(),ingest(obs)],ignore_index=True)
            if country not in cache: cache[country]=daily(cache['combined'],country)
            ts=cache[country]
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',date): raise ValueError('date must be YYYY-MM-DD')
        start=time.perf_counter();result=predict(joblib.load(path),ts,date)
        event={'type':'predict','request_id':uuid.uuid4().hex,'country':country,'date':date,'timestamp':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'runtime':time.perf_counter()-start,**result}
        log_event(a.config['LOG_DIR'],event)
        return jsonify(event)
    @a.get('/logfile')
    def logs():
        try: limit=int(request.args.get('limit',100))
        except ValueError: raise ValueError('Invalid limit')
        if not 1<=limit<=1000: raise ValueError('limit must be between 1 and 1000')
        return jsonify(events=read_logs(a.config['LOG_DIR'],limit))
    return a
app=create_app()
