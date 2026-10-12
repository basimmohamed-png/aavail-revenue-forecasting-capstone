"""Fetch public course data from a pinned upstream commit and verify SHA256."""
from pathlib import Path
import json,hashlib,gzip,urllib.request
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
COMMIT='fd6182c4458bccb56024047a1a85f506a90caf98'
BASE='https://raw.githubusercontent.com/aavail/ai-workflow-capstone/'+COMMIT+'/'
def fetch(item):
    name,expected=item
    path=ROOT/'data'/(name+'.gz')
    if path.exists():
        with gzip.open(path,'rb') as f: data=f.read()
        if hashlib.sha256(data).hexdigest()==expected:return name
    request=urllib.request.Request(BASE+name,headers={'User-Agent':'AAVAIL-Capstone-Course-Data'})
    with urllib.request.urlopen(request,timeout=120) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Checksum mismatch: '+name)
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wb',compresslevel=3) as f:f.write(data)
    return name
if __name__=='__main__':
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(fetch,manifest.items()))
    print('Verified public course files:',len(results))
