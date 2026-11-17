
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Iris MLOps Level 2 API")

MODEL_NAME = "IrisRandomForestModel"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = None


def load_latest_model():

    try:

        client = MlflowClient()

        # Get all registered versions
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        if not versions:
            print("No registered model versions found.")
            return None

        # Select the highest version number
        latest_version = max(
            versions,
            key=lambda version: int(version.version)
        )

        model_uri = (
            f"models:/{MODEL_NAME}/{latest_version.version}"
        )

        print(f"Loading model: {model_uri}")

        return mlflow.sklearn.load_model(model_uri)

    except Exception as e:

        print(f"Model loading failed: {e}")
        return None


# Load model when application starts
@app.on_event("startup")
def startup_event():

    global model

    model = load_latest_model()


class IrisInput(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: IrisInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }
