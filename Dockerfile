FROM python:3.11-slim
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python download_data.py
RUN python run_tests.py
RUN useradd --create-home appuser && mkdir -p logs artifacts reports && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "300", "--bind", "0.0.0.0:8000", "app:app"]
