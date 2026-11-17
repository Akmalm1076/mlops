
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a portable MLflow database and model artifacts
# inside the container itself.
RUN rm -f mlflow.db && \
    rm -rf mlartifacts && \
    python src/retrain_pipeline.py

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
