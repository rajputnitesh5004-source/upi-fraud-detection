# UPI Sentinel — Fraud Detection & Analytics

An end-to-end **synthetic-data** UPI fraud detection portfolio project. It combines feature engineering, anomaly detection, an ensemble risk score, a FastAPI scoring service, an interactive fintech dashboard, and a Jupyter EDA notebook.

> **Data disclaimer:** All transaction records and fraud labels in this repository are synthetically generated. Model results are educational, not a claim of production performance or real UPI fraud detection accuracy.

## Highlights

- **10,000 synthetic transactions** with sender/receiver VPAs, banks, apps, transaction types, categories, amounts, statuses, timestamps and injected fraud labels.
- **Fraud patterns:** unusually high payments to new receivers, large night transfers, and rapid bursts of payments.
- **Detection approaches:** IQR rule, Isolation Forest, time-series z-score, velocity rule, and a weighted ensemble.
- **Time-based holdout evaluation:** precision, recall, F1-score and ROC-AUC are saved to `reports/metrics.json`.
- **FastAPI service:** `POST /score` returns a risk score, `ALLOW` or `HOLD_AND_WARN`, reasons and a warning message.
- **UPI Sentinel dashboard:** historical KPI cards, daily activity, fraud-type mix, hourly activity, category breakdown, model scorecard and live transaction simulation.
- **EDA notebook:** `notebooks/UPI_Fraud_EDA.ipynb` explores class balance, amount distributions, time patterns, fraud types, categories, apps, status, segment fraud rates and numeric relationships.

## Project structure

```text
upi-fraud-detection/
├── app.py                       # FastAPI app and dashboard analytics endpoint
├── detector.py                  # Stateful real-time risk scorer
├── features.py                  # Shared batch/live feature engineering
├── generate_data.py              # Generate 10,000 synthetic transactions
├── train.py                      # Train detectors and write evaluation reports
├── simulate.py                   # Replay simulated transactions against /score
├── requirements.txt
├── data/upi_transactions.csv     # Generated synthetic dataset
├── models/detector.joblib        # Trained model + training-period state
├── notebooks/UPI_Fraud_EDA.ipynb # Exploratory data analysis notebook
├── reports/metrics.json          # Holdout evaluation metrics
├── reports/daily_anomalies.png   # Training pipeline time-series chart
└── static/index.html             # Premium interactive dashboard
```

## Run locally in VS Code (Windows)

Open a terminal in the **inner project folder** — the one containing `app.py`, `train.py`, and `requirements.txt`.

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset

This overwrites the synthetic CSV with a newly generated dataset using the fixed random seed.

```powershell
python generate_data.py
```

### 3. (Optional) Retrain the model and regenerate metrics

Run this after regenerating data or changing feature/model logic.

```powershell
python train.py
```

### 4. Start the dashboard and API

```powershell
python -m uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in your browser. Keep the terminal open while you use the dashboard.

Useful endpoints:

- `GET /` — interactive dashboard
- `GET /health` — service health
- `GET /sample-user` — synthetic sample sender
- `GET /api/analytics` — aggregate historical analytics for charts (does not expose raw VPAs)
- `POST /score` — score one simulated transaction
- `GET /docs` — interactive API documentation

### 5. Replay a scenario (optional)

Keep the server running in one terminal. Open a **second terminal** in the same project folder and run:

```powershell
python simulate.py
```

### 6. Run the EDA notebook

From the project root:

```powershell
python -m jupyter notebook
```

Open `notebooks/UPI_Fraud_EDA.ipynb` and run the cells from top to bottom. The notebook saves its plots to `reports/`.

## API example

```powershell
$body = @{
  sender_vpa = "demo-sender@ybl"
  receiver_vpa = "demo-receiver@okaxis"
  amount_inr = 5000
  txn_type = "P2P"
  timestamp = "2026-10-28T03:00:00"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/score" -Method Post -ContentType "application/json" -Body $body
```

Use the sample sender shown in the dashboard when you want the demo to use a sender already represented in the model's synthetic training history.

## Reported baseline results

The included `reports/metrics.json` currently reports the following time-based holdout scores from the supplied training run:

| Method | Precision | Recall | F1 |
|---|---:|---:|---:|
| IQR rule | 0.696 | 0.722 | 0.709 |
| Isolation Forest | 0.480 | 0.444 | 0.462 |
| Time-series z > 3 | 0.640 | 0.593 | 0.615 |
| Velocity rule | 1.000 | 0.167 | 0.286 |
| **Ensemble** | **0.904** | **0.870** | **0.887** |

Ensemble ROC-AUC is reported as **0.984**. These numbers are from synthetic data and should be reproduced with `python train.py` before using them in a resume or presentation. Synthetic patterns can make results optimistic compared with real-world payments.

## Important limitations

- The dataset is synthetic; it does not represent actual customer behavior or real UPI traffic.
- Fraud labels are artificially injected, so the model can learn patterns that are unusually clear compared with production data.
- The demo is a local educational prototype, not a production payment gateway. It does not connect to a bank, send payments, or place real transactions on hold.
- The real-time detector keeps in-memory state. Restarting the server resets session updates; this is not a durable multi-user production store.
- Before production use, add authentication, authorization, rate limits, durable state, monitoring, drift checks, privacy controls, and a human-reviewed decision policy.

## Suggested portfolio story

Describe the work as an end-to-end synthetic fraud analytics prototype. Discuss the data generation, feature engineering, time-based evaluation, precision/recall trade-off, false-positive cost, and the weakness of detecting a burst before enough transactions have occurred. Be transparent that the dataset and evaluation are synthetic.
