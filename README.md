# AAVAIL Revenue Forecasting and Model Deployment

**Author:** Basim Mohamed  
**Suggested repository name:** `aavail-revenue-forecasting-capstone`

A peer-review capstone solution for forecasting the next 30 days of revenue for a country or all countries combined. Flask API, isolated unit tests, chronological model comparisons, delayed-label monitoring, and Docker packaging.

## Quick start

Python 3.11 or 3.12:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
python run_tests.py
python -m flask --app app run --port 8000
```

Open http://127.0.0.1:8000/health. Pretrained models are included for `all`, `united_kingdom` and `usa`. API input country identifiers use lowercase and underscores. Do not include a currency symbol: units are those of the source dataset.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"country":"all","date":"2019-08-01"}'
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"country":"united_kingdom","date":"2019-08-01"}'
curl -X POST http://127.0.0.1:8000/train -H "Content-Type: application/json" -d '{"country":"usa"}'
curl http://127.0.0.1:8000/logfile?limit=10
```

On Windows PowerShell, use `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -Body '{"country":"all","date":"2019-08-01"}'
```

Forecast dates must follow the model training end, have at least 70 days history, and be no later than the day after the latest observations. The bundled historical demonstration uses 2019 data; today's date will correctly be rejected. This service does not synthesize unavailable future lag data.

## Reproduce the analysis

```bash
python analyze.py
python monitor.py
python run_tests.py
```

`analyze.py` compares baseline, random forest and histogram gradient boosting for three scopes, fits the best candidate on training data, evaluates untouched production data, and regenerates the HTML report and charts. Models are selected using validation MAE. A 30-day purge prevents overlapping-target leakage. Production observations are used only as past features; production labels are never fitted. `monitor.py` waits for a complete 30-day target window, computes MAE/RMSE/WAPE, flags feature novelty and illustrative performance alerts, and reports non-overlapping-origin WAPE as well.

## Docker

**Docker build and execution verified successfully in GitHub Actions.** Nine tests passed during image build and again inside the running image. Real HTTP predictions passed for `all` and `united_kingdom`. [View successful verification](https://github.com/basimmohamed-png/aavail-revenue-forecasting-capstone/actions/runs/38244620994).

To reproduce locally:

```bash
docker build -t aavail-capstone .
docker run --rm aavail-capstone python run_tests.py
docker run --rm -p 127.0.0.1:8000:8000 aavail-capstone
```

On Linux/macOS/Git Bash, `./verify_docker.sh` builds the image, runs tests and checks real predictions through HTTP. Run it with port 8000 free. Every image build also runs the unit tests. For persistent runtime logs/models, create local folders and bind mount them; note that mounting an empty folder over `/app/artifacts` hides the pretrained models, so train first. The educational API has no authentication and is intended for local demonstration.

## Review checklist

| Requirement | Evidence |
|---|---|
| API unit tests | `tests/test_project.py`, valid predictions plus malformed JSON, invalid country/date and limits |
| Model unit tests | Future feature leakage, target horizon, serialization, prediction range and finite output |
| Logging unit tests | Isolated temporary logs and read/write roundtrip |
| One script, all pass | `python run_tests.py`; captured output `reports/test_results.txt` |
| Performance monitoring | `monitor.py`, `reports/monitoring.json`, WAPE and novelty/MAE alerts |
| Isolated test writes | All fixtures/models/logs live in `TemporaryDirectory`; no production paths |
| Country and combined predictions | Pretrained `all`, `united_kingdom`, `usa`; integration tests and smoke results |
| Automated data ingestion | `core.ingest()` handles source column aliases and invalid schema |
| Multiple models compared | `reports/model_comparison.csv` |
| EDA visualizations | `reports/eda.png` |
| Working Docker image | Image build, nine tests and HTTP predictions passed in GitHub Actions |
| Baseline visualization | `reports/baseline_comparison.png`, `reports/production_comparison.png` |

Read [Final report](FINAL_REPORT.md) directly on GitHub, or open [HTML report](reports/Final_Report.html) locally. To view it, download/open it locally with the images in the same directory; GitHub does not render arbitrary HTML reports. CSV/JSON outputs and PNG charts are directly reviewable on GitHub.

## Files and provenance

Data and business task: https://github.com/aavail/ai-workflow-capstone at commit `fd6182c4458bccb56024047a1a85f506a90caf98`.
Source train/production JSONs are not committed. Run `python download_data.py` to fetch the public course data from the pinned upstream commit and verify every SHA256 hash before saving compressed JSONs. `core.ingest()` supports both plain and compressed JSON. This solution is independently implemented; the supplied reference's shuffled train/test procedure is not used because adjacent 30-day targets overlap. No fitted gold-standard model/prediction artifact is provided by that source, so a direct gold-standard numerical comparison cannot be verified; a transparent baseline is supplied.

The upstream dataset is course material from the upstream repository. No additional license for that data is asserted here. Dependencies retain their own licenses.

## Operational limits

Training is synchronous; the image uses one Gunicorn worker and four threads, with a training lock and atomic model writes. Cached source data remain fixed for a process lifetime; restart the service after replacing input JSONs. For distributed deployment use training queues, shared model storage, centralized logs, authentication and scheduling. Automatic periodic retraining is not implemented (optional assignment extension). An HTTP load smoke test is a functional concurrency check, not a production capacity guarantee.
