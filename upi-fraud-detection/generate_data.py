"""Step 1: generate 10,000 synthetic UPI transactions with injected fraud patterns."""
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED, N_TOTAL = 42, 10_000
random.seed(SEED)
rng = np.random.default_rng(SEED)

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Rahul", "Amit", "Vikram", "Karan", "Manish",
         "Priya", "Ananya", "Sneha", "Pooja", "Neha", "Kavya", "Riya", "Meera", "Anjali", "Shreya",
         "Mohammed", "Imran", "Gurpreet", "Karthik", "Abhishek", "Lakshmi", "Isha", "Deepak"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Reddy", "Nair", "Iyer", "Kumar", "Das",
        "Joshi", "Mehta", "Shah", "Khan", "Yadav", "Rao", "Pillai", "Agarwal", "Mishra", "Kapoor"]
APPS = {"PhonePe": "ybl", "Google Pay": "okaxis", "Paytm": "paytm", "BHIM": "upi"}
BANKS = ["SBI", "HDFC", "ICICI", "Axis", "Kotak", "PNB", "Bank of Baroda", "Canara"]
MERCHANTS = {
    "Groceries": ["BigBasket", "Blinkit", "Zepto", "DMart", "Local Kirana"],
    "Food": ["Swiggy", "Zomato", "Dominos", "Chai Point"],
    "Transport": ["Uber", "Ola", "Rapido", "IRCTC", "FASTag Recharge"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Meesho"],
    "Bills": ["BSES Electricity", "Airtel Recharge", "Jio Recharge", "Tata Power"],
    "Entertainment": ["BookMyShow", "Netflix", "Spotify"],
    "Health": ["Apollo Pharmacy", "1mg", "PharmEasy"],
}
HOUR_W = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 7, 7, 8, 7, 6, 6, 7, 8, 9, 9, 8, 6, 4, 2], float)
HOUR_W /= HOUR_W.sum()
START, END = datetime(2026, 1, 1), datetime(2026, 9, 27)

# ---- entities -------------------------------------------------------------
users = []
for i in range(400):
    name = f"{random.choice(FIRST)} {random.choice(LAST)}"
    app = random.choice(list(APPS))
    users.append(dict(name=name, vpa=f"{name.lower().replace(' ', '')}{i}@{APPS[app]}", app=app,
                      bank=random.choice(BANKS), median=float(np.exp(rng.normal(5.4, 0.7)))))
merchants = [(n, f"{n.lower().replace(' ', '')}@{random.choice(['ybl', 'paytm', 'okicici', 'axl'])}", "P2M", c)
             for c, names in MERCHANTS.items() for n in names]
peers = [(u["name"], u["vpa"], "P2P", "Transfer") for u in users]
for u in users:  # each user has a few usual receivers -> "known" receivers are normal
    u["fav"] = random.sample(merchants, 4) + random.sample(peers, 2)
MULES = [(f"{random.choice(FIRST)} {random.choice(LAST)}",
          f"{random.randint(6_000_000_000, 9_999_999_999)}@{random.choice(['ybl', 'paytm', 'okaxis'])}",
          "P2P", "Transfer") for _ in range(30)]


def random_ts():
    d = START + timedelta(days=int(rng.integers(0, (END - START).days + 1)))
    return d.replace(hour=int(rng.choice(24, p=HOUR_W)), minute=random.randint(0, 59), second=random.randint(0, 59))


def make_row(u, rcv, amount, ts, status="SUCCESS", fraud=0, ftype=""):
    return dict(transaction_id=f"TXN{random.getrandbits(56):014X}", upi_ref_no=str(random.randint(10**11, 10**12 - 1)),
                timestamp=ts, sender_name=u["name"], sender_vpa=u["vpa"], sender_bank=u["bank"], upi_app=u["app"],
                receiver_name=rcv[0], receiver_vpa=rcv[1], txn_type=rcv[2], category=rcv[3],
                amount_inr=float(amount), status=status, is_fraud=fraud, fraud_type=ftype)


def fraud_rows():
    rows = []
    for _ in range(100):  # 1) unusually large amount, new receiver
        u = random.choice(users)
        ts = random_ts().replace(hour=random.randint(9, 21))
        rows.append(make_row(u, random.choice(MULES), round(min(u["median"] * rng.uniform(10, 25), 100_000)), ts,
                             fraud=1, ftype="high_amount"))
    for _ in range(60):  # 2) large transfer in the middle of the night
        u = random.choice(users)
        ts = random_ts().replace(hour=random.randint(1, 4))
        rows.append(make_row(u, random.choice(MULES), round(min(u["median"] * rng.uniform(5, 15), 100_000)), ts,
                             fraud=1, ftype="night_transfer"))
    for _ in range(28):  # 3) rapid burst of transfers to different new receivers
        u, ts = random.choice(users), random_ts()
        for _ in range(random.randint(4, 7)):
            ts += timedelta(seconds=random.randint(15, 60))
            rows.append(make_row(u, random.choice(MULES), round(u["median"] * rng.uniform(1.5, 4)), ts,
                                 fraud=1, ftype="burst"))
    return rows


def normal_rows(n):
    rows, pool = [], merchants + peers
    while len(rows) < n:
        u = random.choice(users)
        rcv = random.choice(u["fav"]) if random.random() < 0.88 else random.choice(pool)
        if rcv[1] == u["vpa"]:
            continue
        amt = u["median"] * float(np.exp(rng.normal(0, 0.6)))
        amt = round(amt) if random.random() < 0.7 else round(amt, 2)
        status = random.choices(["SUCCESS", "FAILED", "PENDING"], [96, 3.3, 0.7])[0]
        rows.append(make_row(u, rcv, min(max(amt, 5), 50_000), random_ts(), status))
    return rows


if __name__ == "__main__":
    fraud = fraud_rows()
    df = pd.DataFrame(fraud + normal_rows(N_TOTAL - len(fraud))).sort_values("timestamp").reset_index(drop=True)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/upi_transactions.csv", index=False)
    print(f"Saved {len(df)} transactions | fraud rate {df.is_fraud.mean():.2%}")
    print(df.fraud_type.replace("", "normal").value_counts())
