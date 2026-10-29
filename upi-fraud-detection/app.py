"""FastAPI service for the synthetic UPI fraud-detection demo."""
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from detector import RealtimeDetector

BASE = Path(__file__).parent
app = FastAPI(
    title="UPI Fraud Detection API",
    description="Synthetic-data fraud scoring and analytics demo.",
    version="2.0.0",
)
detector = RealtimeDetector(str(BASE / "models/detector.joblib"))


class Txn(BaseModel):
    sender_vpa: str = Field(min_length=3, max_length=160)
    receiver_vpa: str = Field(min_length=3, max_length=160)
    amount_inr: float = Field(gt=0, le=10_000_000)
    txn_type: str = "P2P"
    timestamp: Optional[datetime] = None


@lru_cache(maxsize=1)
def get_dataset():
    path = BASE / "data/upi_transactions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def _count_records(series, limit=10):
    return [{"label": str(k) if pd.notna(k) and str(k) else "Unknown", "value": int(v)}
            for k, v in series.value_counts(dropna=False).head(limit).items()]


@app.post("/score")
def score(t: Txn):
    try:
        return detector.score(t.sender_vpa, t.receiver_vpa, t.amount_inr, t.txn_type, t.timestamp)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/sample-user")
def sample_user():
    return detector.sample_user()


@app.get("/api/analytics")
def analytics():
    """Aggregated synthetic-data analytics used by the dashboard; no raw VPAs are exposed."""
    df = get_dataset()
    if df.empty:
        raise HTTPException(status_code=503, detail="Dataset not found. Run python generate_data.py first.")

    total = int(len(df))
    fraud_count = int(df["is_fraud"].fillna(0).astype(int).sum())
    fraud_rows = df[df["is_fraud"].astype(int) == 1]
    legitimate = df[df["is_fraud"].astype(int) == 0]
    hour_counts = df.assign(hour=df["timestamp"].dt.hour).groupby("hour").size()
    hour_fraud = fraud_rows.assign(hour=fraud_rows["timestamp"].dt.hour).groupby("hour").size()
    daily = df.assign(day=df["timestamp"].dt.floor("D")).groupby("day").agg(
        transactions=("transaction_id", "size"), fraud=("is_fraud", "sum")
    ).reset_index()
    # Downsample to at most 45 points so the chart remains readable.
    if len(daily) > 45:
        positions = [round(i * (len(daily) - 1) / 44) for i in range(45)]
        daily = daily.iloc[positions]
    try:
        metrics = json.loads((BASE / "reports/metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metrics = {"threshold": None, "holdout": {}, "holdout_recall_by_fraud_type": {}}

    return {
        "dataset": {"total_transactions": total, "fraud_count": fraud_count,
                    "legitimate_count": int(total - fraud_count),
                    "fraud_rate_pct": round(100 * fraud_count / max(total, 1), 2),
                    "total_value_inr": round(float(df["amount_inr"].sum()), 2),
                    "average_amount_inr": round(float(df["amount_inr"].mean()), 2),
                    "median_amount_inr": round(float(df["amount_inr"].median()), 2),
                    "fraud_value_inr": round(float(fraud_rows["amount_inr"].sum()), 2),
                    "legitimate_value_inr": round(float(legitimate["amount_inr"].sum()), 2),
                    "date_start": df["timestamp"].min().strftime("%d %b %Y"),
                    "date_end": df["timestamp"].max().strftime("%d %b %Y")},
        "fraud_types": _count_records(fraud_rows["fraud_type"], 8),
        "categories": _count_records(df["category"], 8),
        "apps": _count_records(df["upi_app"], 8),
        "statuses": _count_records(df["status"], 8),
        "hours": [{"label": f"{int(h):02d}:00", "value": int(hour_counts.get(h, 0)),
                   "fraud": int(hour_fraud.get(h, 0))} for h in range(24)],
        "daily": [{"date": row.day.strftime("%d %b"), "transactions": int(row.transactions),
                   "fraud": int(row.fraud)} for row in daily.itertuples(index=False)],
        "model_metrics": metrics,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "upi-fraud-detection", "version": app.version}


@app.get("/")
def index():
    return FileResponse(BASE / "static/index.html")
