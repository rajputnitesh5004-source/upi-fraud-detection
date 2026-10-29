"""Step 2: anomaly detection - IQR rule, Isolation Forest, time-series z-score, and an ensemble."""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from features import FEATURES, build_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TS_THR, VEL_THR = 3.0, 2   # VEL_THR=2 -> the 3rd payment within 10 minutes is flagged
WEIGHTS = dict(iso=0.4, iqr=0.2, ts=0.15, vel=0.25)

raw = pd.read_csv("data/upi_transactions.csv", parse_dates=["timestamp"]).sort_values("timestamp")
cut = raw["timestamp"].quantile(0.8)          # time-based split: train on the past, test on the future
medians = raw[raw.timestamp <= cut].groupby("sender_vpa")["amount_inr"].median().to_dict()
gmed = float(raw.loc[raw.timestamp <= cut, "amount_inr"].median())

df, X = build_features(raw, medians, gmed)
tr = (df.timestamp <= cut).values
te = ~tr
y = df["is_fraud"].values
print(f"train {tr.sum()} rows | holdout {te.sum()} rows | holdout fraud {y[te].sum()} ({y[te].mean():.2%})")

# 1) IQR rule on how far the amount is from the sender's usual amount
q1, q3 = np.percentile(X.loc[tr, "log_amount_ratio"], [25, 75])
iqr_hi = q3 + 1.5 * (q3 - q1)
iqr_flag = (X["log_amount_ratio"] > iqr_hi).astype(int).values

# 2) Isolation Forest (unsupervised - never sees the labels)
iso = IsolationForest(n_estimators=300, contamination=0.03, random_state=42).fit(X[tr])
raw_score = -iso.score_samples(X)
lo, hi = raw_score[tr].min(), raw_score[tr].max()
iso_score = np.clip((raw_score - lo) / (hi - lo), 0, 1)
iso_flag = (iso.predict(X) == -1).astype(int)

# 3) Time-series: amount vs the sender's own rolling history of previous payments
ts_flag = (X["ts_z"] > TS_THR).astype(int).values

# 4) Velocity rule: burst of payments in a short window (time-series of payment counts)
vel_flag = (X["velocity_10m"] >= VEL_THR).astype(int).values

# 5) Ensemble risk score; threshold picked on the training period only
risk = (WEIGHTS["iso"] * iso_score + WEIGHTS["iqr"] * iqr_flag
        + WEIGHTS["ts"] * ts_flag + WEIGHTS["vel"] * vel_flag)


def prf(pred, mask):
    p, r, f, _ = precision_recall_fscore_support(y[mask], pred[mask], average="binary", zero_division=0)
    return dict(precision=round(float(p), 3), recall=round(float(r), 3), f1=round(float(f), 3))


grid = np.arange(0.2, 0.9, 0.02)
threshold = float(max(grid, key=lambda t: prf((risk >= t).astype(int), tr)["f1"]))
alert = (risk >= threshold).astype(int)

results = {
    "IQR rule": prf(iqr_flag, te),
    "Isolation Forest": prf(iso_flag, te),
    "Time-series z>3": prf(ts_flag, te),
    "Velocity rule": prf(vel_flag, te),
    "Ensemble": prf(alert, te),
}
results["Ensemble"]["roc_auc"] = round(float(roc_auc_score(y[te], risk[te])), 3)
by_type = (pd.Series(alert[te & (y == 1)]).groupby(df.loc[te & (y == 1), "fraud_type"].values).mean().round(3).to_dict())
metrics = dict(threshold=round(threshold, 2), holdout=results, holdout_recall_by_fraud_type=by_type)

print(f"\nHoldout results (alert threshold {threshold:.2f})")
print(pd.DataFrame(results).T.fillna("").to_string())
print("\nRecall by fraud type:", by_type)

Path("reports").mkdir(exist_ok=True)
Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))

# time-series view: daily volume vs daily alerts
d = df.assign(alert=alert, day=df.timestamp.dt.floor("D"))
daily = d.groupby("day").agg(txns=("alert", "size"), alerts=("alert", "sum"), fraud=("is_fraud", "sum"))
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(daily.index, daily.txns, color="#555")
ax[0].set_title("Daily transaction volume")
ax[1].bar(daily.index, daily.alerts, color="#d9534f", label="alerts raised")
ax[1].plot(daily.index, daily.fraud, color="black", lw=1, label="actual fraud")
ax[1].axvline(cut, ls="--", color="blue")
ax[1].text(cut, ax[1].get_ylim()[1] * 0.9, " holdout ->", color="blue")
ax[1].legend()
ax[1].set_title("Daily alerts vs actual fraud")
plt.tight_layout()
plt.savefig("reports/daily_anomalies.png", dpi=110)

# artifact for the real-time service: model + user state as of the end of the training period
Path("models").mkdir(exist_ok=True)
train_df = df[tr]
joblib.dump(dict(
    iso=iso, iso_min=float(lo), iso_max=float(hi), iqr_hi=float(iqr_hi), ts_thr=TS_THR, vel_thr=VEL_THR, weights=WEIGHTS,
    threshold=threshold, medians=medians, global_median=gmed,
    pairs=set(zip(train_df.sender_vpa, train_df.receiver_vpa)),
    recent=train_df.groupby("sender_vpa")["amount_inr"].apply(lambda s: list(np.log(s.tail(10)))).to_dict(),
), "models/detector.joblib")
print("Saved models/detector.joblib")
