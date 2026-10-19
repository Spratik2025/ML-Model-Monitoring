import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

RANDOM_STATE = 42
PSI_THRESHOLD = 0.25
ACCURACY_THRESHOLD = 0.75
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(RANDOM_STATE)


def calculate_psi(reference, current, bins=10):
    """Calculate PSI using bins defined by the reference distribution."""
    reference = np.asarray(reference).ravel()
    current = np.asarray(current).ravel()

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0

    # Extend edges so values outside the reference range are included.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = np.maximum(ref_counts / max(len(reference), 1), eps)
    cur_pct = np.maximum(cur_counts / max(len(current), 1), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def send_alert(message):
    """Demo alert mechanism: prints and appends a warning to an alert log."""
    line = f"ALERT: {message}"
    print(line)
    with open(os.path.join(OUTPUT_DIR, "alerts.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    # Clear old alert log for a fresh run.
    alert_path = os.path.join(OUTPUT_DIR, "alerts.log")
    if os.path.exists(alert_path):
        os.remove(alert_path)

    X, y = make_classification(
        n_samples=4000,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    joblib.dump(model, "models/model.pkl")
    print("Model trained successfully.")

    # Scenario A: incoming data similar to the training distribution.
    stable_data = rng.normal(
        loc=np.mean(X_train, axis=0),
        scale=np.std(X_train, axis=0),
        size=(1000, X_train.shape[1]),
    )

    # Scenario B: deliberately shifted incoming feature distribution.
    drifted_data = rng.normal(
        loc=np.mean(X_train, axis=0) + 3.0,
        scale=np.std(X_train, axis=0),
        size=(1000, X_train.shape[1]),
    )

    scenarios = {"No Drift": stable_data, "Real Drift": drifted_data}
    drift_results = []

    for scenario, incoming in scenarios.items():
        psi_values = [
            calculate_psi(X_train[:, i], incoming[:, i])
            for i in range(X_train.shape[1])
        ]
        mean_psi = float(np.mean(psi_values))
        detected = mean_psi >= PSI_THRESHOLD
        drift_results.append({
            "Scenario": scenario,
            "Mean PSI": mean_psi,
            "Drift Detected": detected,
        })

        status = "DRIFT DETECTED" if detected else "No meaningful drift"
        print(f"{scenario}: mean PSI={mean_psi:.3f} -> {status}")
        if detected:
            send_alert(f"Data drift detected in scenario '{scenario}' (mean PSI={mean_psi:.3f}).")

    drift_df = pd.DataFrame(drift_results)

    # Simulate batches over time. Later batches have feature shift and label noise.
    performance_results = []
    for batch in range(1, 9):
        shift = max(0, batch - 4) * 0.8
        batch_X = X_test.copy()
        batch_X[:, 0] += shift
        batch_y = y_test.copy()

        if batch >= 6:
            flip_probability = min(0.45, 0.15 * (batch - 5))
            flip_mask = rng.random(len(batch_y)) < flip_probability
            batch_y[flip_mask] = 1 - batch_y[flip_mask]

        predictions = model.predict(batch_X)
        accuracy = float(accuracy_score(batch_y, predictions))
        mean_psi = float(np.mean([
            calculate_psi(X_train[:, i], batch_X[:, i])
            for i in range(X_train.shape[1])
        ]))
        degraded = accuracy < ACCURACY_THRESHOLD
        drift = mean_psi >= PSI_THRESHOLD

        performance_results.append({
            "Batch": batch,
            "Accuracy": accuracy,
            "Mean PSI": mean_psi,
            "Performance Degraded": degraded,
            "Drift Detected": drift,
        })

        print(
            f"Batch {batch}: accuracy={accuracy:.3f}, mean PSI={mean_psi:.3f}"
        )
        if degraded:
            send_alert(
                f"Performance degradation in batch {batch}: accuracy={accuracy:.3f} "
                f"is below {ACCURACY_THRESHOLD:.2f}."
            )
        if drift:
            send_alert(
                f"Data drift detected in batch {batch}: mean PSI={mean_psi:.3f}."
            )

    performance_df = pd.DataFrame(performance_results)

    # Save result tables.
    drift_df.to_csv(os.path.join(OUTPUT_DIR, "drift_results.csv"), index=False)
    performance_df.to_csv(
        os.path.join(OUTPUT_DIR, "performance_results.csv"), index=False
    )

    # Dashboard/report image.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(drift_df["Scenario"], drift_df["Mean PSI"])
    axes[0].axhline(
        PSI_THRESHOLD, linestyle="--", label=f"PSI threshold = {PSI_THRESHOLD}"
    )
    axes[0].set_title("Data Drift Detection")
    axes[0].set_ylabel("Mean PSI")
    axes[0].legend()

    axes[1].plot(
        performance_df["Batch"], performance_df["Accuracy"],
        marker="o", label="Accuracy"
    )
    axes[1].axhline(
        ACCURACY_THRESHOLD,
        linestyle="--",
        label=f"Accuracy threshold = {ACCURACY_THRESHOLD}",
    )
    axes[1].set_title("Model Performance Over Time")
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.tight_layout()
    report_path = os.path.join(OUTPUT_DIR, "monitoring_report.png")
    plt.savefig(report_path, dpi=150)
    plt.close(fig)

    print("\nMonitoring complete.")
    print(f"Dashboard: {report_path}")
    print(f"Drift results: {OUTPUT_DIR}/drift_results.csv")
    print(f"Performance results: {OUTPUT_DIR}/performance_results.csv")
    print(f"Alert log: {alert_path}")


if __name__ == "__main__":
    main()
