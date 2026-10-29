"""Replay the busiest fraud day of the unseen holdout period through the live API, in time order."""
import json
import sys
import urllib.request

import pandas as pd

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/score"
df = pd.read_csv("data/upi_transactions.csv", parse_dates=["timestamp"]).sort_values("timestamp")
hold = df[df.timestamp > df.timestamp.quantile(0.8)]
day = hold.groupby(hold.timestamp.dt.date)["is_fraud"].sum().idxmax()
today = hold[hold.timestamp.dt.date == day]
print(f"Replaying {len(today)} transactions from {day} ({int(today.is_fraud.sum())} are fraud)\n")

tp = fp = fn = 0
for _, r in today.iterrows():
    body = dict(sender_vpa=r.sender_vpa, receiver_vpa=r.receiver_vpa, amount_inr=r.amount_inr,
                txn_type=r.txn_type, timestamp=r.timestamp.isoformat())
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    res = json.load(urllib.request.urlopen(req))
    flagged, truth = res["is_suspicious"], bool(r.is_fraud)
    tp += flagged and truth
    fp += flagged and not truth
    fn += truth and not flagged
    if flagged or truth:
        print(f"{r.timestamp:%H:%M:%S}  Rs{r.amount_inr:>9,.0f}  risk={res['risk_score']:.2f}  "
              f"{'HELD ' if flagged else 'allow'}  actual={'FRAUD' if truth else 'ok'}  {'; '.join(res['reasons'])}")
print(f"\ncaught {tp} | missed {fn} | false alarms {fp}")
