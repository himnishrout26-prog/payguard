# PayGuard — UPI Fraud Detection & Risk Scoring

A fraud-risk scoring service for mobile-money (UPI-style) transactions: feature
engineering on transaction/balance data, a gradient-boosted-tree classifier
tuned for extreme class imbalance, probability calibration, cost-based
decision thresholding, and a FastAPI endpoint that returns a risk score plus
the top factors driving it.

**Live demo**: [PayGuard risk console](https://claude.ai/artifact/TE5jdyJtQ8VYfWa6o4tudB) —
the real trained model (72 trees, exported to JSON, trained on the full real
PaySim dataset) scoring transactions directly in your browser. No server, no
API call. Try the four preset transactions to see the model distinguish a
large-but-legit transfer from a smaller fraud-shaped one.

## Results (real PaySim dataset, 6,362,620 transactions)

Trained on the real PaySim mobile-money dataset (Kaggle, `ntnu-testimon/paysim1`),
time-split 60% train / 20% calibration / 20% test by transaction step.

| Metric | Value |
|---|---|
| **PR-AUC** | **0.9995** (vs. 0.0033 for a random classifier — the positive rate in the test window) |
| ROC-AUC | 0.9999 |
| Brier score (calibration quality) | 0.0000234 |
| Fraud transactions caught | 4,252 / 4,254 (99.95%) |
| Fraud value caught at the deployed threshold | 99.99% of ₹670.07 crore in test-set fraud |
| False positives | 2,631 out of 1,268,270 legit transactions (0.207%) |
| Estimated cost with model vs. doing nothing | ₹8.45 lakh vs. ₹670.07 crore on this test window |
| Trees in the final model | 72 (of a 250-tree budget — early stopping on a validation split) |
| Training time (full 3.8M-row training set) | ~25 minutes |

**On the two missed fraud transactions**: fraud *value* caught rounds to
100%, but 2 of 4,254 fraud transactions were missed — say "99.95% of fraud
transactions, 100% of fraud value" if asked precisely, not just "100%." The
2 misses were low-value enough not to move the value-caught number, but they
happened.

**Fraud rate shifts across the time-based split** — worth knowing, not a
bug: 0.084% in train, 0.060% in calibration, but 0.334% in the test window
(roughly 4x higher). Real PaySim's fraud isn't spread evenly over time; the
test period is a genuinely harder evaluation than a uniform random split
would give.

**Business framing:** a missed fraud costs the full transaction amount (money
gone). A false positive costs a flat friction fee (₹35 — support ticket +
step-up verification + a moment of customer annoyance). Because that
asymmetry is so large — fraud value in the billions vs. a ₹35 friction cost —
the cost-optimal threshold lands very close to zero: catch essentially
everything, accept a controlled false-positive rate. The threshold isn't
picked by eyeballing a PR curve — it's picked by directly minimizing
₹(missed fraud) + ₹(friction cost × false positives) on a held-out
calibration slice, then evaluated on a separate test slice. The ₹35 figure
is a reasonable estimate, not measured data — a real deployment would tune
it from actual support-cost and customer-attrition numbers.

## How good is this, honestly?

The engineering is real and defensible: leak-free time-based splitting,
sample-weighted imbalance handling, calibration with a documented reason for
the method chosen (not just "used the default"), a threshold picked by
minimizing an actual cost function instead of eyeballing a PR curve, and a
from-scratch explainability implementation that's verified to reconstruct
the model's exact output. That combination — not the headline PR-AUC number
— is what a hiring manager is actually checking for.

The PR-AUC of 0.9995 is *not*, by itself, evidence of a good model here —
it's largely a property of this dataset. PaySim's fraud transactions have an
almost-deterministic accounting signature (`errorBalanceOrig`/
`errorBalanceDest`), so any reasonable tree model finds it. The real PaySim
dataset shows this exact near-perfect separability in every published
analysis of it — this project's number is consistent with that, not an
outlier result. Presenting 0.9995 as "look how good my model is" without
that context would be the wrong takeaway; presenting it as "I understood
*why* it's this high, and I built the surrounding decision infrastructure
(cost thresholding, calibration, explainability) that actually matters in
production" is the honest and more impressive story — and it's the one this
README leads with.

## Honest caveats

This was originally built in a sandboxed environment with no internet
access, which meant training on a synthetic PaySim-schema dataset at first.
**The pipeline has since been re-run end to end on the real PaySim dataset**
(6.36M transactions) — the numbers above are from that real run. Two
substitutions from the original build remain in place:

1. **Model**: `src/train.py`'s production model is
   `sklearn.GradientBoostingClassifier`, not LightGBM/XGBoost. `train_compare.py`
   exists to benchmark all three under identical conditions and has been
   validated on a development subset — running it on the full 6.36M-row
   real dataset and reporting the winner is a documented next step, not yet
   done. See "What to do next" below.
2. **Explainability**: `src/explain.py`'s production path is the hand-rolled
   Saabas method, not SHAP. It's exact for a single tree and additive across
   the ensemble — verified to reconstruct the model's raw score to 4 decimal
   places — but unlike TreeSHAP it doesn't average over feature orderings,
   so it can be less intuitive for one-hot categorical features specifically.
   Real `shap.TreeExplainer` support is wired into `explain.py` and activates
   automatically once a LightGBM/XGBoost model + the `shap` package are
   present — see the commented dispatch block in that file.

Both are stated plainly here so neither claim overreaches what's actually
running in production right now.

## Why balance-discrepancy features do most of the work

PaySim's transactions are supposed to be double-entry: the sender's balance
should drop by exactly the amount, and the receiver's should rise by exactly
the amount. Two engineered features check that:

```
errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
errorBalanceDest = oldbalanceDest + amount - newbalanceDest
```

When these are near zero, the ledger is consistent — normal behavior. Large
discrepancies correlate with fraud (the balance update didn't fully "catch
up" to the drain), and separately with certain untracked merchant flows
(a different, non-fraud reason for the ledger to look odd) — which is
exactly why the model needs more than one feature and a nonlinear model,
not a hand-written rule.

Full feature list is in `src/features.py`; besides the two error terms it
includes drain-to-zero and mule-account flags, amount-to-balance ratio,
hour-of-day/night flags, and per-account transaction velocity (count and
average amount for that sender's prior transactions, computed leak-free —
each row only sees its own account's *earlier* history).

## What to do next

Two things remain from the original plan, in the order they're worth doing:

**1. Run `train_compare.py` on the full real dataset.**
```bash
pip install -r requirements-optional.txt
cd src
python train_compare.py
```
This will take longer than `train.py`'s ~25 minutes, since it trains three
models back to back on 3.8M rows — budget an hour or more, and don't
interrupt it. It prints a head-to-head PR-AUC/latency/training-time table
and automatically promotes whichever model wins to `artifacts/model.joblib`.
**If sklearn's GBM wins again**, nothing else changes — you now have a real,
defensible three-way comparison to cite instead of just having used one
library. **If LightGBM or XGBoost wins**, `explain.py` needs no changes (the
SHAP dispatch activates automatically), but `export_for_ui.py` needs a
different tree-exporter for the browser demo — say so and it'll get written
for that specific case.

**2. Deploy the API.**
Do this *after* step 1, not before — the resume link should point at
whichever model actually won the comparison, not get redeployed twice.
Follow `DEPLOY.md` (Render or Railway, both auto-build from the included
`Dockerfile`):
```bash
docker build -t payguard .          # test locally first
docker run -p 8000:8000 payguard
curl http://localhost:8000/health   # confirm it responds before pushing to a host
```
Free-tier services spin down when idle and take 30-60s to wake on the first
request — mention that in your resume link so it doesn't look broken on a
cold hit.

**Suggested order, concretely**: run `train_compare.py` this week while
you're still actively working on the project (so you're around to debug if
LightGBM/XGBoost hit a version issue — Python 3.14 is new enough that this
is a real possibility, not a formality); once you have a final winning model
and its README numbers, deploy once, and treat that deployment as the one
you put on your resume. Redeploying every time you tweak something is more
churn than value — deploy when the model itself is the thing that changed,
not for cosmetic updates.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```
33 known-input/known-output cases against `features.py` — e.g. "a
transaction whose ledger reconciles exactly has zero balance error", "a
brand-new account's velocity features default sensibly." These pin down
exact arithmetic, which matters more here than usual: a silently wrong
feature is worse than a crashing one in a fraud model.

## Optional: a graph-based mule-detection feature

`src/graph_features.py` adds `dest_new_senders_recent` — how many distinct
senders contacted a given receiver for the first time in a short window.
A receiver suddenly popular with strangers is the textbook money-mule
pattern, and this is a genuinely different *kind* of signal from everything
else in the pipeline (a property of the account graph, not of a single
transaction).

It's **not wired into the main pipeline by default** — adding it changes
`FEATURE_COLUMNS`, which means retraining, re-validating the browser demo's
JS port against the new model, and updating the pytest cases that pin down
the exact feature order. That's real, deliberate work, not a drop-in. The
module docstring has the wiring instructions when you're ready to do it.
On the real PaySim dataset this hasn't been benchmarked for correlation with
fraud yet (it was checked against the earlier synthetic dataset only, where
it showed ~zero correlation, expectedly, since the synthetic generator
didn't model mule-recruitment patterns) — it's a legitimate idea worth
having in an interview answer, not something validated with a number here
yet.

## Architecture

```
src/config.py            path config, relative to project root (works locally, in Docker, on Render/Railway)
src/generate_data.py     synthetic PaySim-schema dataset generator (used for initial development; production now trained on real data)
src/features.py          feature engineering, shared by training + serving
src/train.py             time-based split, imbalance-weighted training,
                          Platt-scaling calibration, cost-based thresholding
src/train_compare.py     benchmarks sklearn GBM vs LightGBM vs XGBoost
src/graph_features.py    optional mule-receiver graph feature (not wired in by default)
src/explain.py           Saabas per-prediction feature attribution (+ SHAP dispatch)
src/app.py               FastAPI /score endpoint
src/export_for_ui.py     exports the trained model to JSON for payguard.html
tests/test_features.py   33 known-input/known-output pytest cases
payguard.html            live browser console, runs the exported model client-side
Dockerfile, DEPLOY.md    containerize + deploy to Render/Railway
artifacts/                trained model, calibrator, metrics.json, comparison_metrics.json
```

**Split strategy**: time-based (by `step`), not random — 60% train / 20%
calibration / 20% test, in chronological order. A random split would leak
future information into training, which is not how this model will be used
in production.

**Imbalance handling**: sample weights (`pos_weight ≈ 1195` on the real
dataset), not `class_weight="balanced"` — sample weights work identically
across sklearn/LightGBM/XGBoost, so the training code doesn't need to change
when you swap the model library.

**Calibration**: Platt scaling (logistic regression on the raw score), not
isotonic regression. Isotonic regression was tried first and found to
collapse to a handful of step-function output levels when the calibration
set has too few positive examples relative to its size — too coarse for
threshold search to mean anything. Platt scaling stays smooth under this
regime. Worth revisiting isotonic once you have a calibration set with
thousands of positive examples, which the real dataset's 20% calibration
slice may now actually support — untested as of this README.

## Running it

```bash
pip install -r requirements.txt

# artifacts/ already has a trained run on the real PaySim dataset.
# to reproduce from scratch, put the real PaySim CSV at data/upi_transactions.csv, then:
cd src
python train.py

# serve
uvicorn app:app --reload --port 8000
```

Example request:
```bash
curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{
  "step": 14, "type": "TRANSFER", "amount": 45230.0,
  "nameOrig": "C1234567890", "oldbalanceOrg": 45230.0, "newbalanceOrig": 0.0,
  "nameDest": "C9876543210", "oldbalanceDest": 120.0, "newbalanceDest": 36304.0
}'
```

Example response:
```json
{
  "risk_score": 0.87,
  "is_flagged": true,
  "threshold_used": 0.0000222,
  "top_contributing_factors": [
    {"feature": "errorBalanceDest", "contribution": 1.99, "value": 9046.0,
     "direction": "increases_risk",
     "description": "Receiver's balance doesn't reconcile with the transaction"},
    ...
  ]
}
```

## Interactive console (`payguard.html`)

A self-contained, published web UI that runs the real trained model
client-side — the tree structure and calibrator are exported to a ~80KB
JSON blob (`artifacts/model_export.json`, produced by `src/export_for_ui.py`)
and re-implemented in vanilla JS, verified against the live Python model to
match output exactly before shipping. It's not a mockup of the API; it's the
API's decision logic running in the browser instead of on a server. See it
live at the link at the top of this README, or open `payguard.html` directly
(works offline, no build step).

Deliberately out of scope for the demo, called out on the page itself: the
per-account transaction-velocity history is kept in browser memory for the
session only (mirrors the in-memory stand-in `app.py` uses server-side) and
resets on reload — a real deployment would back it with a feature store.

## What I'd do differently with real infrastructure

- Real LightGBM with monotonic constraints on the ratio/error features (risk
  should never *decrease* as the balance discrepancy grows — sklearn's GBM
  doesn't support this as cleanly)
- A real feature store (Redis/DynamoDB) for the per-account velocity window
  instead of the in-memory dict `app.py` uses as a stand-in
- Isotonic calibration if a much larger calibration set makes it viable
  (see the note above)
- TreeSHAP instead of Saabas, mainly for the correlated-feature case
  (errorBalanceOrig/errorBalanceDest are correlated; SHAP handles that split
  of credit more carefully)
- Monitoring for feature drift on the velocity features specifically, since
  they depend on a rolling window that production traffic patterns will
  shift over time — the fraud-rate shift observed between the train/calib/test
  windows in this very dataset is a small-scale preview of exactly this
  problem
