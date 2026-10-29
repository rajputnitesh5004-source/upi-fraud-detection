"""Real-time scorer: keeps per-user state (recent amounts, known receivers, 10-min velocity)."""
import random
from collections import defaultdict, deque
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from features import FEATURES, HISTORY, single_features


class RealtimeDetector:
    def __init__(self, path="models/detector.joblib"):
        a = joblib.load(path)
        self.a = a
        self.pairs = set(a["pairs"])
        self.recent = defaultdict(list, {k: list(v) for k, v in a["recent"].items()})
        self.vel = defaultdict(lambda: deque(maxlen=50))
        self.receivers = defaultdict(list)
        for s, r in self.pairs:
            self.receivers[s].append(r)

    def sample_user(self):
        sender = random.choice([s for s in self.receivers if s in self.a["medians"]])
        return dict(sender_vpa=sender, median=round(self.a["medians"][sender], 2),
                    usual_receiver=random.choice(self.receivers[sender]))

    def score(self, sender_vpa, receiver_vpa, amount_inr, txn_type="P2P", timestamp=None):
        a, ts = self.a, (timestamp or datetime.now()).replace(tzinfo=None)
        median = a["medians"].get(sender_vpa, a["global_median"])
        is_new = (sender_vpa, receiver_vpa) not in self.pairs
        f = single_features(amount_inr, ts, txn_type, median, self.vel[sender_vpa], is_new, self.recent[sender_vpa])
        x = pd.DataFrame([f])[FEATURES]
        raw = -self.a["iso"].score_samples(x)[0]
        iso_score = float(np.clip((raw - a["iso_min"]) / (a["iso_max"] - a["iso_min"]), 0, 1))
        iqr_flag, ts_flag = f["log_amount_ratio"] > a["iqr_hi"], f["ts_z"] > a["ts_thr"]
        vel_flag = f["velocity_10m"] >= a["vel_thr"]
        w = a["weights"]
        risk = w["iso"] * iso_score + w["iqr"] * iqr_flag + w["ts"] * ts_flag + w["vel"] * vel_flag
        suspicious = bool(risk >= a["threshold"])

        reasons = []
        if iqr_flag:
            reasons.append(f"Amount is {np.exp(f['log_amount_ratio']):.1f}x this user's usual payment")
        if ts_flag:
            reasons.append("Amount is far above this user's recent payment history")
        if f["is_night"]:
            reasons.append("Payment made late at night (12am-5am)")
        if f["is_new_receiver"]:
            reasons.append("First payment to this receiver")
        if vel_flag:
            reasons.append(f"{int(f['velocity_10m'])} other payments in the last 10 minutes")

        self.vel[sender_vpa].append(ts.timestamp())
        if not suspicious:  # don't let flagged payments poison the user's baseline
            self.pairs.add((sender_vpa, receiver_vpa))
            self.recent[sender_vpa] = (self.recent[sender_vpa] + [float(np.log(amount_inr))])[-HISTORY:]
        return dict(
            risk_score=round(float(risk), 3), is_suspicious=suspicious,
            action="HOLD_AND_WARN" if suspicious else "ALLOW", reasons=reasons,
            model_scores=dict(isolation_forest=round(iso_score, 3), iqr_flag=bool(iqr_flag), timeseries_flag=bool(ts_flag), velocity_flag=bool(vel_flag)),
            message=(f"Warning: a payment of Rs {amount_inr:,.0f} to {receiver_vpa} was put on hold. "
                     "Was this you? Reply YES to release or NO to block.") if suspicious else "Payment allowed.",
        )
