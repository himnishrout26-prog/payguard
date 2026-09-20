# PayGuard — UPI-Style Fraud Detection & Risk Scoring

A fraud-risk scoring service for mobile-money (UPI-style) transactions. Built end to end: feature engineering on transaction and balance data, a gradient-boosted-tree classifier tuned for extreme class imbalance, probability calibration, cost-based decision thresholding, per-prediction explainability, and a FastAPI endpoint that returns a risk score plus the top factors driving it.

**Live demo:** open `payguard.html` in the repo. The real trained model — 145 LightGBM trees, exported to JSON, trained on the full 6.36M-row PaySim dataset — runs entirely client-side. No server, no API call. Try the four preset transactions to see the model separate a large-but-legit transfer from a smaller fraud-shaped drain.

---

## Results

Trained on the real PaySim mobile-money dataset (6,362,620 transactions), time-split 55% train / 15% validation / 15% calibration / 15% test by transaction step. The winning model was chosen by a head-to-head comparison against sklearn's `GradientBoostingClassifier` and XGBoost on the same split, same imbalance handling, and same cost-based threshold procedure.

### Three-way comparison (identical 954K-row test slice)

| Model | PR-AUC | ROC-AUC | Train time | Latency p95 | False positives |
|---|---|---|---|---|---|
| sklearn GBM | 0.9995 | 0.9998 | 1439.4 s | 0.737 ms | 2,656 |
| **LightGBM (winner)** | **0.9999** | **1.0000** | **20.4 s** | **0.705 ms** | **1,264** |
| XGBoost | 0.9995 | 0.9998 | 41.8 s | 1.340 ms | 1,605 |

LightGBM won on every axis that mattered: highest PR-AUC, lowest training time (70× faster than sklearn), and the fewest false positives — 52% fewer than the sklearn baseline and 21% fewer than XGBoost. That last part surprised me going in: I'd assumed LightGBM's speed would come at a precision cost. On this dataset, it didn't.

### Winning model (LightGBM) in detail

| Metric | Value |
|---|---|
| PR-AUC | 0.99987 (vs. 0.0042 for a random classifier — 238× lift) |
| ROC-AUC | 0.9999991 |
| Brier score (calibration quality) | 0.0000334 |
| Fraud transactions caught | 4,009 / 4,010 (99.975%) |
| Fraud value caught | 99.9944% of ₹632.38 crore in test-set fraud |
| False positives | 1,264 out of 950,383 legit transactions (0.133%) |
| Estimated cost with model vs. doing nothing | ₹3.98 lakh vs. ₹632.38 crore |
| Trees in the final model | 145 (of a 400-tree budget — early stopping on a validation split) |
| Training time (3.5M-row training set) | 20.4 seconds |
| Inference latency (single-row p95) | 0.71 ms |

On the one missed fraud transaction: fraud value caught rounds to 100% (99.9944%), but 1 of 4,010 fraud transactions was missed. If asked precisely: "99.98% of fraud transactions, 99.99% of fraud value" — not just "100%."

**Fraud rate shifts across the time-based split** — worth knowing, not a bug: 0.083% in train, 0.077% in validation, 0.059% in calibration, but 0.42% in the test window (~5× higher than training). PaySim's fraud isn't spread evenly over time, so the test period is a genuinely harder evaluation than a uniform random split would give. I kept the time-based split anyway, because that's how the model would actually be deployed.

**Business framing:** a missed fraud costs the full transaction amount. A false positive costs a flat friction fee — modeled at ₹35 for support ticket + step-up verification + a moment of customer annoyance. Because that asymmetry is so large (fraud value in the billions vs. ₹35 friction), the cost-optimal threshold lands very close to zero: catch essentially everything, accept a controlled false-positive rate. The threshold isn't picked by eyeballing a PR curve — it's picked by directly minimizing ₹(missed fraud) + ₹(friction cost × false positives) on a held-out calibration slice, evaluated on a separate, never-touched test slice. The ₹35 figure is a reasonable estimate, not measured data; a real deployment would tune it from actual support-cost and customer-attrition numbers.

---

## How good is this, honestly?

The engineering is real and defensible: leak-free time-based splitting, a 4-way train/validation/calibration/test split so early stopping and calibration never share data, sample-weighted imbalance handling, a calibration step with a documented reason for the method chosen (not just "used the default"), a threshold picked by minimizing an actual cost function instead of eyeballing a PR curve, and a from-scratch explainability implementation that's verified to reconstruct the model's exact output. That combination — not the headline PR-AUC number — is what I think matters here.

The PR-AUC of 0.9999 is not, by itself, evidence of a good model. It's largely a property of this dataset. PaySim's fraud transactions have an almost-deterministic accounting signature (`errorBalanceOrig` / `errorBalanceDest`), so any reasonable tree model finds it. Every published analysis of PaySim shows this near-perfect separability. Presenting 0.9999 as "look how good my model is" without that context would be the wrong takeaway. Presenting it as "I understood why it's this high, and I built the surrounding decision infrastructure (cost thresholding, calibration, explainability) that actually matters in production" is the honest story — and it's the one I want this README to lead with.

---

## The XGBoost debugging story

This is the most interesting engineering detail in the project, and the one I learned the most from.

On my first three-way comparison run, XGBoost reported a catastrophic result: PR-AUC 0.0031, ROC-AUC 0.0001, 1.27M false positives. Worse than a coin flip — the model was anti-correlated with fraud. Something was badly wrong with either my code or my understanding.

I wrote a small diagnostic script (`src/diagnose_comparison.py`) to isolate the failure, and found two independent bugs stacked on top of each other:

**Bug A — early stopping on a noisy metric.** With only ~0.08% positives in the training data, the validation slice's `aucpr` metric bounces around from round to round on pure sampling noise. Across separate runs, XGBoost's early stopping picked `best_iteration` = 3, then 34, then 1 — the argmax was noise-driven, not convergence. When it picked 1, the sklearn wrapper then restricted `predict_proba` to a single tree's worth of splits, which produced near-constant raw scores (14 unique values, std = 0.006).

**Bug B — Platt calibration inverting the ranking.** Platt scaling fits a logistic regression on the raw score. Fed a near-constant input, it can fit a negative coefficient, which flips the sign of every prediction. Raw ROC-AUC 0.97 became 0.03 after "calibration" — the calibrator had turned a working model into an inverted one.

A third, structural issue that compounded both: the original 3-way split used the same held-out slice for two different jobs — model early stopping and calibrator fitting. That meant the calibrator was fit on data whose aggregate signal had already influenced the model's stopping point, optimistically biasing the calibrated probabilities.

**The fix, three parts:**

1. **Drop XGBoost's early stopping.** Fixed 250 rounds with a low learning rate. The signal isn't there to stop on, so stop trying.
2. **Split early stopping off from calibration.** The new pipeline uses a 4-way split: train / validation / calibration / test, each used for exactly one purpose, never overlapping.
3. **Add a calibration guard.** Before fitting the calibrator, check that its coefficient is positive and that calibrated scores don't rank worse than raw scores on the calibration slice. If either fails, refuse to save the model. This catches Bug B at the source, for any model, not just XGBoost.

After the fix, XGBoost reported PR-AUC 0.9995, ROC-AUC 0.9998, 1,605 false positives — a healthy, well-behaved model. It still lost to LightGBM on speed and FP count, but it was a real contest.

The same guard later caught LightGBM briefly inverting on an unrealistically small calibration slice in the diagnostic script. That confirmed the guard is a general safety net, not a one-off XGBoost patch.

---

## Why balance-discrepancy features do most of the work

PaySim's transactions are supposed to be double-entry: the sender's balance should drop by exactly the amount, and the receiver's should rise by exactly the amount. Two engineered features check that:

```
errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
errorBalanceDest = oldbalanceDest + amount - newbalanceDest
```

When these are near zero, the ledger is consistent — normal behavior. Large discrepancies correlate with fraud (the balance update didn't fully "catch up" to the drain), and separately with certain untracked merchant flows — a different, non-fraud reason for the ledger to look odd. That's exactly why the model needs more than one feature and a nonlinear model, not a hand-written rule.

Full feature list is in `src/features.py`. Besides the two error terms it includes drain-to-zero and mule-account flags, amount-to-balance ratio, hour-of-day/night flags, and per-account transaction velocity — count and average amount for that sender's prior transactions, computed leak-free (each row only sees its own account's earlier history, never its own or any future row's).

---

## Architecture

```
src/config.py                 path config, relative to project root (works locally, in Docker, on Render/Railway)
src/generate_data.py          synthetic PaySim-schema dataset generator (used during initial development)
src/features.py               feature engineering, shared by training + serving
src/train.py                  time-based split, imbalance-weighted training, Platt-scaling calibration, cost-based thresholding (sklearn GBM only — kept as a minimal reference)
src/train_compare.py          benchmarks sklearn GBM vs LightGBM vs XGBoost, promotes the winner
src/diagnose_comparison.py    reproduces the XGBoost failure, verifies the fix
src/graph_features.py         optional mule-receiver graph feature (not wired in by default)
src/explain.py                per-prediction feature attribution (native TreeSHAP for LightGBM/XGBoost; Saabas for sklearn)
src/app.py                    FastAPI /score endpoint
src/export_for_ui.py          exports the trained model to JSON for payguard.html, with built-in verification
tests/test_features.py        known-input/known-output pytest cases
payguard.html                 live browser console, runs the exported model client-side
Dockerfile, DEPLOY.md         containerize + deploy to Render/Railway
artifacts/                    trained model, calibrator, metrics.json, comparison_metrics.json, model_export.js
```

**Split strategy:** time-based by step, not random — 55/15/15/15 in chronological order (train / validation / calibration / test). A random split would leak future information into training, and sharing validation with calibration would optimistically bias the calibrator. Both failure modes are avoided.

**Imbalance handling:** sample weights (pos_weight ≈ 1205 on the real dataset), not `class_weight="balanced"`. Sample weights work identically across sklearn, LightGBM, and XGBoost, so the training code doesn't need to change when the model library changes.

**Calibration:** Platt scaling (logistic regression on the raw score), not isotonic regression. I tried isotonic first and found it collapsed to a handful of step-function output levels when the calibration set has too few positive examples relative to its size — too coarse for threshold search to mean anything. Platt stays smooth under this regime. The calibration step has a safety check (see the XGBoost debugging story) that refuses to save a model whose calibrated scores rank worse than its raw scores.

---

## Running it

```bash
pip install -r requirements.txt
pip install lightgbm xgboost   # optional but recommended — enables the three-way comparison

# artifacts/ already contains the trained LightGBM winner.
# To reproduce from scratch, put the real PaySim CSV at data/upi_transactions.csv, then:
cd src
python train_compare.py

# serve
uvicorn app:app --reload --port 8000
```

**Example request:**

```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "step": 14, "type": "TRANSFER", "amount": 45230.0,
  "nameOrig": "C1234567890", "oldbalanceOrg": 45230.0, "newbalanceOrig": 0.0,
  "nameDest": "C9876543210", "oldbalanceDest": 0.0, "newbalanceDest": 45230.0
}'
```

**Example response:**

```json
{
  "risk_score": 0.87,
  "is_flagged": true,
  "threshold_used": 0.0000135,
  "model_version": "lightgbm-v1",
  "top_contributing_factors": [
    {"feature": "errorBalanceDest", "contribution": 1.99, "value": 9046.0,
     "direction": "increases_risk",
     "description": "Receiver's balance doesn't reconcile with the transaction"}
  ]
}
```

---

## Browser demo (`payguard.html`)

`payguard.html` runs the exported LightGBM model entirely client-side. The tree structure and calibrator are exported to a JS blob (`artifacts/model_export.js`, produced by `src/export_for_ui.py`) and re-implemented in vanilla JavaScript. It's not a mockup of the API — it's the API's decision logic running in the browser instead of on a server. Open the file directly; works offline, no build step, no server.

`src/export_for_ui.py` flattens the trained model to a common JSON tree format and, before writing, verifies that the flattened trees reproduce the Python model's raw score on 5,000 real rows. The current LightGBM export passes at a max error of **1.78e-15** — effectively machine epsilon — so the browser produces byte-identical decisions to the API. The exporter handles sklearn, LightGBM, and XGBoost tree formats, so if a future retrain produces a different winner, no code needs to change.

That verification step caught two real bugs during development: LightGBM applies its learning rate inside the dumped leaf values (so re-applying it shrinks every tree ~20×), and it adds a `boost_from_average` bias that isn't part of any tree. Both would have silently produced wrong scores in the browser without the check.

Refresh after retraining with:

```bash
python3 src/export_for_ui.py
```

Deliberately out of scope for the demo, called out on the page itself: the per-account transaction-velocity history is kept in browser memory for the session only (mirrors the in-memory stand-in `app.py` uses server-side) and resets on reload. A real deployment would back it with a feature store.

---

## Honest caveats

This was originally built in a sandboxed environment with no internet access, which meant training on a synthetic PaySim-schema dataset at first. **The pipeline has since been re-run end to end on the real PaySim dataset** — the numbers above are from that real run.

Two substitutions from the original build remain in place:

1. **Explainability**: `src/explain.py` dispatches by model type. For the current production LightGBM model it uses LightGBM's native `pred_contrib=True`, which returns exact TreeSHAP values in log-odds space — no external `shap` package required. The sklearn fallback (Saabas) remains for the sklearn GBM, and is exact for a single tree and additive across the ensemble, verified to reconstruct the model's raw score to 15 decimal places. Saabas doesn't average over feature orderings the way TreeSHAP does, so it can be less intuitive for correlated features — for the current LightGBM model, this limitation doesn't apply.

2. **Browser demo tree-walker**: `payguard.html` walks a normalized flat tree format produced by `src/export_for_ui.py`. The exporter handles sklearn, LightGBM, and XGBoost, and verifies reconstruction before writing. If a future retrain produces a different winner, the exporter and demo keep working without code changes.

---

## What I'd do differently with real infrastructure

- **Monotonic constraints** on the ratio and error features. Risk should never *decrease* as the balance discrepancy grows. LightGBM supports this directly and it's the first thing I'd add on a real deployment — it's a free correctness guarantee that a purely data-driven fit can't promise.
- **A real feature store** (Redis/DynamoDB) for the per-account velocity window, instead of the in-memory dict `app.py` uses as a stand-in. Needed the moment you run more than one server process.
- **Isotonic calibration** if a much larger calibration set makes it viable. The current calibration slice has too few positives for isotonic to produce smooth output.
- **Feature drift monitoring** on the velocity features specifically, since they depend on a rolling window that production traffic patterns will shift over time. The fraud-rate shift observed between the train/val/calib/test windows in this very dataset is a small-scale preview of exactly this problem.
- **Shadow-mode rollout** before promoting any model to block transactions live. Score traffic in parallel with the current model, compare decisions, and only cut over when you have evidence the new one is not worse on the slices you care about.

---

## Optional: a graph-based mule-detection feature

`src/graph_features.py` adds `dest_new_senders_recent` — how many distinct senders contacted a given receiver for the first time in a short window. A receiver suddenly popular with strangers is the textbook money-mule pattern, and this is a genuinely different *kind* of signal from everything else in the pipeline (a property of the account graph, not of a single transaction).

It's **not wired into the main pipeline by default** — adding it changes `FEATURE_COLUMNS`, which means retraining, re-validating the browser demo's JavaScript port against the new model, and updating the pytest cases that pin down the exact feature order. That's real, deliberate work, not a drop-in. The module docstring has the wiring instructions.
