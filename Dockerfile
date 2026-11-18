FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate a fresh MLflow registry and model
RUN rm -f mlflow.db && \
    rm -rf mlartifacts mlruns && \
    python -c "from src.retrain_pipeline import train_and_register_model; train_and_register_model()"

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
