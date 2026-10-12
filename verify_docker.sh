#!/bin/sh
set -eu
docker build -t aavail-capstone .
docker run --rm aavail-capstone python run_tests.py
docker run --rm -d --name aavail-capstone-check -p 127.0.0.1:8000:8000 aavail-capstone
trap 'docker stop aavail-capstone-check >/dev/null 2>&1 || true' EXIT
python - <<'PY'
import time,urllib.request,json
for i in range(30):
    try:
        assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/health'))['status']=='ok'
        break
    except Exception: time.sleep(1)
else: raise SystemExit('Container health check failed')
for country in ['all','united_kingdom']:
    body=json.dumps({'country':country,'date':'2019-08-01'}).encode()
    r=urllib.request.Request('http://127.0.0.1:8000/predict',data=body,headers={'Content-Type':'application/json'})
    assert json.load(urllib.request.urlopen(r))['prediction']>=0
print('Docker tests and predictions passed')
PY
