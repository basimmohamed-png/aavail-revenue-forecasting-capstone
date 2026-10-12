"""Start a real local HTTP server and exercise country/all predictions concurrently."""
import json,threading,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from werkzeug.serving import make_server
from app import create_app
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as temp:
    server=make_server('127.0.0.1',0,create_app({'LOG_DIR':temp}),threaded=True)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base='http://127.0.0.1:'+str(server.server_port)
    def req(country):
        start=time.perf_counter()
        data=json.dumps({'country':country,'date':'2019-08-01'}).encode()
        request=urllib.request.Request(base+'/predict',data=data,headers={'Content-Type':'application/json'})
        response=json.load(urllib.request.urlopen(request,timeout=120))
        assert response['prediction']>=0
        return {'country':country,'prediction':response['prediction'],'elapsed_seconds':time.perf_counter()-start}
    # Warm up data caches, then test concurrent requests.
    result=[req(c) for c in ['all','united_kingdom','usa']]
    with ThreadPoolExecutor(max_workers=4) as pool:load=list(pool.map(req,['all','united_kingdom','usa']*4))
    logs=json.load(urllib.request.urlopen(base+'/logfile'))['events']
    assert len(logs)==15
    assert len({x['request_id'] for x in logs})==15
    server.shutdown()
    Path('reports/api_smoke.json').write_text(json.dumps({'initial_predictions':result,'concurrent_requests':len(load),'max_elapsed_seconds':max(x['elapsed_seconds'] for x in load),'unique_log_events':len(logs)},indent=2))
    print('15 real HTTP predictions and unique isolated log events passed')
