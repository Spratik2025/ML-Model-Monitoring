# ML Model Monitoring System

## Overview

This project demonstrates how to monitor a machine learning model after deployment. It detects changes in incoming feature distributions, tracks model accuracy over simulated batches, triggers alerts, and creates a report image.

## Features

- Population Stability Index (PSI) based data drift detection
- Two scenarios: stable incoming data and deliberately shifted data
- Model accuracy monitoring across simulated batches
- Alerts written to the terminal and `reports/alerts.log`
- Dashboard image and CSV result files
- Trained model saved as `models/model.pkl`

## Requirements

- Python 3.9 or newer recommended

## Installation

Open a terminal in the project directory and run:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the script without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the project:

```bash
python app.py
```

## Output

After running, check:

- `reports/monitoring_report.png` — drift and accuracy charts
- `reports/drift_results.csv` — drift scenario results
- `reports/performance_results.csv` — batch performance results
- `reports/alerts.log` — alerts triggered by thresholds
- `models/model.pkl` — trained Random Forest model

The generated reports are created when the application runs, so the output folder may initially be empty.

## Thresholds

- PSI threshold: `0.25`
- Accuracy threshold: `0.75`

PSI thresholds are practical heuristics, not universal guarantees. The accuracy threshold should be selected based on the business requirements for a real project.

## How it works

1. Generate a classification dataset and split it into training and test data.
2. Train a Random Forest classifier.
3. Compare stable and shifted incoming feature distributions with the training data using PSI.
4. Simulate eight batches, calculate accuracy, and calculate mean PSI for each batch.
5. Log an alert if PSI or accuracy crosses its configured threshold.
6. Save charts and CSV reports.

## Real-world response to drift

When drift is detected, first validate the data pipeline and investigate which features changed. Drift alone does not prove that accuracy has fallen. Once ground-truth labels become available, evaluate actual model performance. If degradation is confirmed, retrain through the team's ML pipeline, validate the candidate model, deploy only if it passes acceptance criteria, and keep the previous model available for rollback.

## Important limitation

The project uses synthetic data to make the demonstration reproducible. The later performance batches intentionally add label noise to simulate degradation. In production, use real labels when available and monitor data quality, prediction distributions, and business metrics as well as PSI.
