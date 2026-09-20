"""
FastAPI fraud-scoring service.

Run:
    pip install fastapi uvicorn pydantic
    uvicorn app:app --reload --port 8000

Model-agnostic: reads metrics.json to report the actual model type, and
dispatches explain_prediction to whichever family won training.
"""
import json
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from features import build_features, FEATURE_COLUMNS, CATEGORICAL_TYPES
from explain import explain_prediction

from config import ARTIFACT_DIR

FRIENDLY_NAMES = {
    "errorBalanceOrig": "Sender's balance doesn't reconcile with the transaction",
    "errorBalanceDest": "Receiver's balance doesn't reconcile with the transaction",
    "orig_drained_to_zero": "Sender's account was emptied to ₹0",
    "dest_is_mule_pattern": "Receiver account shows a mule-like zero-balance pattern",
    "amount_to_orig_balance_ratio": "Transaction is a large share of the sender's balance",
    "amount_vs_orig_avg_ratio": "Amount is unusually large vs. this sender's typical transaction",
    "orig_txn_count_so_far": "Sender's transaction history length",
    "orig_avg_amount_so_far": "Sender's historical average transaction size",
    "is_night": "Transaction occurred late at night",
    "hour_of_day": "Hour of day",
    "oldbalanceOrg": "Sender's balance before the transaction",
    "newbalanceOrig": "Sender's balance after the transaction",
    "oldbalanceDest": "Receiver's balance before the transaction",
    "newbalanceDest": "Receiver's balance after the transaction",
    "amount": "Transaction amount",
}
for t in CATEGORICAL_TYPES:
    FRIENDLY_NAMES[f"type_{t}"] = f"Transaction type is {t}"


class Transaction(BaseModel):
    step: int = Field(..., description="Time step (hour index), as in PaySim")
    type: str = Field(..., description="One of CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


class ScoreResponse(BaseModel):
    risk_score: float
    is_flagged: bool
    threshold_used: float
    top_contributing_factors: list
    model_version: str


app = FastAPI(title="UPI Fraud Risk Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's origin before real traffic
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- load artifacts once at startup ---
model = joblib.load(f"{ARTIFACT_DIR}/model.joblib")
calibrator = joblib.load(f"{ARTIFACT_DIR}/calibrator.joblib")
with open(f"{ARTIFACT_DIR}/feature_columns.json") as f:
    _feature_cols = json.load(f)
with open(f"{ARTIFACT_DIR}/metrics.json") as f:
    _metrics = json.load(f)
THRESHOLD = _metrics["cost_optimal_threshold"]
MODEL_TYPE = _metrics.get("model_type", type(model).__name__)
MODEL_VERSION = f"{MODEL_TYPE}-v1"

if _feature_cols != FEATURE_COLUMNS:
    raise RuntimeError(
        "Saved feature_columns.json does not match FEATURE_COLUMNS in "
        "features.py. The model was trained against a different feature "
        "order; refusing to serve."
    )

# in-memory per-account velocity store — see module docstring in features.py
# for the production replacement (Redis / DynamoDB keyed by account ID).
_account_history = defaultdict(lambda: {"count": 0, "amount_sum": 0.0})


@app.post("/score", response_model=ScoreResponse)
def score_transaction(txn: Transaction):
    row = pd.DataFrame([txn.model_dump()])
    feat = build_features(row, history=_account_history)
    x_df = feat[FEATURE_COLUMNS]

    # Pass a DataFrame (not a numpy slice) so LightGBM/XGBoost keep
    # feature-name alignment and don't emit warnings or rely on positional
    # matching that could silently break if FEATURE_COLUMNS is reordered.
    raw_score = float(model.predict_proba(x_df)[0, 1])
    calibrated = float(calibrator.predict_proba([[raw_score]])[0, 1])

    _, _, factors = explain_prediction(model, x_df, FEATURE_COLUMNS, top_k=5)
    for f in factors:
        f["description"] = FRIENDLY_NAMES.get(f["feature"], f["feature"])

    # update stats AFTER scoring, so this txn doesn't see its own amount
    h = _account_history[txn.nameOrig]
    h["count"] += 1
    h["amount_sum"] += txn.amount

    return ScoreResponse(
        risk_score=round(calibrated, 6),
        is_flagged=bool(calibrated >= THRESHOLD),
        threshold_used=THRESHOLD,
        top_contributing_factors=factors,
        model_version=MODEL_VERSION,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "model_pr_auc_on_test": _metrics["pr_auc"],
    }