
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.monitor_drift import check_data_drift


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-continuous-retraining")

MODEL_NAME = "IrisRandomForestModel"
ACCURACY_THRESHOLD = 0.90


def train_and_register_model():

    print("Initiating retraining process...")

    # 1. Load dataset
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    # 2. Train model
    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    with mlflow.start_run() as run:

        model.fit(X_train, y_train)

        # 3. Evaluate model
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        # 4. Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", accuracy)

        print(f"Candidate Model Trained. Accuracy: {accuracy:.4f}")

        # 5. Quality gate
        if accuracy >= ACCURACY_THRESHOLD:

            model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name=MODEL_NAME,
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

            print(
                "Model successfully logged and registered "
                f"under URI: {model_info.model_uri}"
            )

            return True

        else:

            print(
                "Candidate model failed accuracy threshold "
                f"({ACCURACY_THRESHOLD}). Promotion skipped."
            )

            return False


def run_pipeline():

    print("Checking production data for drift...")

    drift_detected = check_data_drift()

    print(f"Dataset Drift Detected: {drift_detected}")

    if drift_detected:

        print(
            "Data drift confirmed! "
            "Triggering retraining pipeline..."
        )

        train_and_register_model()

    else:

        print(
            "No significant drift detected. "
            "Model operating within normal bounds."
        )


if __name__ == "__main__":
    run_pipeline()
