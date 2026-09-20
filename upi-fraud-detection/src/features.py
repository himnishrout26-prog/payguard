"""
Feature engineering for UPI fraud scoring.

Two families of signal, both grounded in how the fraud actually shows up
in mobile-money data:

1. Balance-discrepancy signals — PaySim's accounting is double-entry, so
   any row where the ledger doesn't balance is either a data-tracking gap
   (common, benign) or an account being drained (fraud). errorBalanceOrig
   and errorBalanceDest, plus the "drained to zero" and "mule account"
   flags, are the single strongest features in every published PaySim
   analysis.
2. Transaction-velocity signals — how fast an origin account is
   transacting right now vs. its own recent history. Real-time fraud
   rarely looks anomalous on amount alone; it looks anomalous on cadence.

`build_features` is called identically at training time (on a DataFrame)
and at serving time (on a single-row DataFrame built from the API
request), so there is no train/serve skew.
"""
import numpy as np
import pandas as pd

CATEGORICAL_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "orig_drained_to_zero", "dest_is_mule_pattern",
    "amount_to_orig_balance_ratio",
    "hour_of_day", "is_night",
    "orig_txn_count_so_far", "orig_avg_amount_so_far", "amount_vs_orig_avg_ratio",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
]


def build_features(df: pd.DataFrame, history: dict | None = None) -> pd.DataFrame:
    """
    df: raw transaction rows with PaySim-style columns.
    history: optional dict of {nameOrig: {"count": int, "amount_sum": float}}
             used at serving time to carry a caller's running stats across
             calls (a real deployment would back this with a feature store /
             Redis keyed by account; here it's an in-memory dict the API
             layer owns). If None, velocity features are computed from the
             batch itself (used at training time, in step order).
    """
    df = df.copy()

    # --- balance-discrepancy signals ---
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["orig_drained_to_zero"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    ).astype(int)
    df["dest_is_mule_pattern"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    ).astype(int)
    df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)

    # --- time signals ---
    df["hour_of_day"] = df["step"] % 24
    df["is_night"] = df["hour_of_day"].isin([0, 1, 2, 3, 4, 22, 23]).astype(int)

    # --- velocity signals (per origin account) ---
    if history is None:
        # training path: compute expanding stats in step order, leak-free
        # (each row only sees the account's own PRIOR transactions)
        df = df.sort_values("step").reset_index(drop=False)
        grp = df.groupby("nameOrig")["amount"]
        prior_count = grp.cumcount()
        prior_sum = grp.cumsum() - df["amount"]
        df["orig_txn_count_so_far"] = prior_count
        df["orig_avg_amount_so_far"] = np.where(
            prior_count > 0, prior_sum / prior_count.replace(0, 1), df["amount"]
        )
        df = df.sort_values("index").drop(columns=["index"]).reset_index(drop=True)
    else:
        # serving path: pull the caller's running stats from the passed-in store
        h = history.get(df["nameOrig"].iloc[0], {"count": 0, "amount_sum": 0.0})
        df["orig_txn_count_so_far"] = h["count"]
        df["orig_avg_amount_so_far"] = (h["amount_sum"] / h["count"]) if h["count"] > 0 else df["amount"].iloc[0]

    df["amount_vs_orig_avg_ratio"] = df["amount"] / (df["orig_avg_amount_so_far"] + 1.0)

    # --- transaction type one-hot ---
    for t in CATEGORICAL_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    return df[FEATURE_COLUMNS + (["isFraud"] if "isFraud" in df.columns else [])]
