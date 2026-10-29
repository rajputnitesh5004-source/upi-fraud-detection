"""Feature engineering shared by training (batch) and the real-time detector (single txn)."""
import numpy as np
import pandas as pd

FEATURES = ["log_amount_ratio", "is_night", "hour_sin", "hour_cos",
            "velocity_10m", "is_new_receiver", "ts_z", "is_p2p"]
VEL_WINDOW = 600   # seconds: "how many txns did this sender make in the last 10 minutes"
HISTORY = 10       # rolling window (previous txns) for the time-series z-score
SD_FLOOR = 0.3


def epoch_seconds(ts: pd.Series) -> np.ndarray:
    return (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()


def _velocity(df: pd.DataFrame) -> np.ndarray:
    t_all, out = epoch_seconds(df["timestamp"]), np.zeros(len(df))
    for idx in df.groupby("sender_vpa").indices.values():
        t = t_all[idx]
        out[idx] = np.arange(len(t)) - np.searchsorted(t, t - VEL_WINDOW, side="left")
    return out


def _rolling_z(s: pd.Series) -> pd.Series:
    prev = s.shift()
    mean = prev.rolling(HISTORY, min_periods=3).mean()
    sd = prev.rolling(HISTORY, min_periods=3).std().clip(lower=SD_FLOOR)
    return (s - mean) / sd


def build_features(df: pd.DataFrame, medians: dict, global_median: float):
    """Batch features. Every feature only looks at the past, so it can be reproduced live."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    la = np.log(df["amount_inr"])
    hour = df["timestamp"].dt.hour
    out = pd.DataFrame(index=df.index)
    out["log_amount_ratio"] = la - np.log(df["sender_vpa"].map(medians).fillna(global_median))
    out["is_night"] = (hour < 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["velocity_10m"] = _velocity(df)
    out["is_new_receiver"] = (~df.duplicated(["sender_vpa", "receiver_vpa"])).astype(int)
    out["ts_z"] = la.groupby(df["sender_vpa"]).transform(_rolling_z).fillna(0).clip(-10, 10)
    out["is_p2p"] = (df["txn_type"] == "P2P").astype(int)
    return df, out[FEATURES]


def single_features(amount, ts, txn_type, median, prior_ts, is_new, prior_logs) -> dict:
    """Same features for one incoming transaction (ts is a python datetime, prior_ts epoch seconds)."""
    la, now = float(np.log(amount)), ts.timestamp()
    z = 0.0
    if len(prior_logs) >= 3:
        last = np.array(prior_logs[-HISTORY:])
        z = float(np.clip((la - last.mean()) / max(last.std(ddof=1), SD_FLOOR), -10, 10))
    return {
        "log_amount_ratio": la - float(np.log(median)),
        "is_night": int(ts.hour < 5),
        "hour_sin": float(np.sin(2 * np.pi * ts.hour / 24)),
        "hour_cos": float(np.cos(2 * np.pi * ts.hour / 24)),
        "velocity_10m": sum(1 for t in prior_ts if 0 <= now - t <= VEL_WINDOW),
        "is_new_receiver": int(is_new),
        "ts_z": z,
        "is_p2p": int(txn_type == "P2P"),
    }
