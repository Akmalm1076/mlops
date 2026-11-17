from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_data_drift():

    print("Loading reference and current data...")

    # 1. Load Iris dataset
    iris = load_iris(as_frame=True)

    # Use only input features, not the target label
    data = iris.frame.drop(columns=["target"])

    # 2. Reference data: represents training data
    reference = data.sample(
        n=100,
        random_state=42
    )

    # 3. Current data: simulates production data
    current = data.sample(
        n=100,
        random_state=99
    ).copy()

    # Artificially introduce drift
    current = current + 10

    print("Generating Evidently drift report...")

    # 4. Create drift report
    drift_report = Report(
        metrics=[DataDriftPreset()]
    )

    drift_report.run(
        reference_data=reference,
        current_data=current
    )

    # 5. Save HTML report
    report_path = PROJECT_ROOT / "drift_report.html"

    drift_report.save_html(str(report_path))

    # 6. Extract drift result
    report_dict = drift_report.as_dict()

    dataset_drift = (
        report_dict["metrics"][0]["result"]["dataset_drift"]
    )

    print("--------------------------------")
    print(f"Dataset Drift Detected: {dataset_drift}")
    print("--------------------------------")

    print(f"Report saved at: {report_path}")

    return dataset_drift


if __name__ == "__main__":
    check_data_drift()
