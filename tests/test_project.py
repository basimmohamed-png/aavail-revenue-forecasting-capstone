import unittest,tempfile,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from core import *
from app import create_app

class ModelTests(unittest.TestCase):
    def setUp(self):
        dates=pd.date_range('2018-01-01',periods=330)
        self.ts=pd.DataFrame({'revenue':100+10*np.sin(np.arange(330)/14),'purchases':10,'views':20},index=dates)
    def test_features_do_not_use_future(self):
        date=self.ts.index[200];a=features(self.ts,date)
        changed=self.ts.copy();changed.loc[date:,'revenue']=1e9
        pd.testing.assert_frame_equal(a,features(changed,date))
    def test_training_serialization_and_prediction(self):
        bundle,comparison=train_series(self.ts)
        self.assertEqual(len(bundle['metrics']),3)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'model.joblib';save_model(p,bundle)
            result=predict(joblib.load(p),self.ts,self.ts.index[-1]+pd.Timedelta(days=1))
            self.assertTrue(np.isfinite(result['prediction']))
            with self.assertRaises(ValueError):predict(bundle,self.ts,self.ts.index[200])
    def test_invalid_date_and_history(self):
        with self.assertRaises(ValueError): features(self.ts,'not-a-date')
        with self.assertRaises(ValueError): features(self.ts,self.ts.index[3])
    def test_future_target(self):
        X,y,d=supervised(self.ts)
        self.assertAlmostEqual(y[0],self.ts.iloc[70:100].revenue.sum())

class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        data=cls.root/'input';data.mkdir()
        rows=[]
        for dt in pd.date_range('2018-01-01',periods=330):
            for c,p in [('United Kingdom',100),('USA',60)]:
                rows.append(dict(country=c,customer_id=1,day=dt.day,month=dt.month,year=dt.year,invoice='A1',price=p,stream_id=2,times_viewed=3))
        (data/'test.json').write_text(json.dumps(rows))
        cls.client=create_app({'TESTING':True,'DATA_DIR':str(data),'OBS_DIR':str(cls.root/'absent'),'MODEL_DIR':str(cls.root/'models'),'LOG_DIR':str(cls.root/'logs')}).test_client()
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_ingestion(self):
        x=ingest(self.root/'input');self.assertEqual(len(x),660)
        self.assertEqual(daily(x).revenue.iloc[0],160)
        with self.assertRaises(ValueError):ingest(self.root/'missing')
    def test_api_train_predict_country_and_all(self):
        for country in ['all','united_kingdom']:
            response=self.client.post('/train',json={'country':country});self.assertEqual(response.status_code,200)
            response=self.client.post('/predict',json={'country':country,'date':'2018-11-27'})
            self.assertEqual(response.status_code,200);self.assertGreater(response.json['prediction'],0)
    def test_api_errors(self):
        for payload in [{},{'country':'../../etc','date':'2018-11-27'},{'country':'all','date':'bad'}]:
            self.assertIn(self.client.post('/predict',json=payload).status_code,[400,404])
        self.assertEqual(self.client.post('/train',data='x',content_type='application/json').status_code,400)
        self.assertEqual(self.client.get('/logfile?limit=0').status_code,400)
        self.assertEqual(self.client.get('/health').status_code,200)
    def test_logging_isolation_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            log_event(t,{'type':'test','prediction':1.0})
            self.assertEqual(read_logs(t)[0]['prediction'],1.0)
        self.assertTrue(str(self.root).startswith(tempfile.gettempdir()))
        self.assertEqual(self.client.get('/logfile').status_code,200)
    def test_malformed_ingestion(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'bad.json').write_text('[{"country":"US"}]')
            with self.assertRaises(ValueError):ingest(t)
