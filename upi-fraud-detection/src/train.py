"""
Train, calibrate, threshold, and save the fraud-scoring model.

Model choice note: the target stack for this project is LightGBM/XGBoost.
This sandbox has no internet access, so those packages can't be pip
installed here. sklearn's GradientBoostingClassifier is used as a drop-in
architectural stand-in — it's the same gradient-boosted-tree family, and
critically it exposes per-tree structure (`.tree_`), which the
explainability module needs. Swapping in real LightGBM/XGBoost later is a
one-line change (see `build_model`) — the feature pipeline, calibration,
thresholding, and API layer are all unaffected.
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    confusion_matrix, brier_score_loss,
)

from features import build_features, FEATURE_COLUMNS
from config import DATA_PATH, ARTIFACT_DIR

# --- business cost assumptions (INR), used for cost-based thresholding ---
# Cost of a missed fraud = the money actually lost (the transaction amount).
# Cost of a false alarm = friction cost of wrongly blocking/step-up-verifying
# a legit UPI transaction: support load + customer trust erosion, modeled as
# a flat cost since it doesn't scale with transaction size.
COST_PER_FALSE_NEGATIVE_IS_AMOUNT = True
FLAT_COST_PER_FALSE_POSITIVE = 35.0  # INR, conservative estimate of friction/support cost


def build_model():
    return GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    # --- to use real LightGBM instead, once you have network access: ---
    # from lightgbm import LGBMClassifier
    # return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
    #                        subsample=0.8, class_weight="balanced", random_state=42)


def find_cost_optimal_threshold(y_true, proba, amounts, fp_cost=FLAT_COST_PER_FALSE_POSITIVE):
    thresholds = np.unique(np.quantile(proba, np.linspace(0, 1, 500)))
    best_t, best_cost = 0.5, np.inf
    costs = []
    for t in thresholds:
        pred = proba >= t
        fn_mask = (~pred) & (y_true == 1)
        fp_mask = pred & (y_true == 0)
        cost = amounts[fn_mask].sum() + fp_mask.sum() * fp_cost
        costs.append((t, cost))
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost, costs


def main():
    t0 = time.time()
    raw = pd.read_csv(DATA_PATH)

    # time-based split (by step) so we never evaluate on the past — this
    # matches how the model will actually be used in production
    raw = raw.sort_values("step").reset_index(drop=True)
    n = len(raw)
    train_cut, calib_cut = int(n * 0.6), int(n * 0.8)
    raw_train = raw.iloc[:train_cut]
    raw_calib = raw.iloc[train_cut:calib_cut]
    raw_test = raw.iloc[calib_cut:]

    print(f"train={len(raw_train)} calib={len(raw_calib)} test={len(raw_test)}")
    print(f"fraud rate  train={raw_train.isFraud.mean():.5f}  "
          f"calib={raw_calib.isFraud.mean():.5f}  test={raw_test.isFraud.mean():.5f}")

    # features computed on the FULL sorted set so velocity features see true
    # history, then sliced back into train/calib/test by the same cut points
    feat_all = build_features(raw)
    feat_train = feat_all.iloc[:train_cut]
    feat_calib = feat_all.iloc[train_cut:calib_cut]
    feat_test = feat_all.iloc[calib_cut:]

    X_train, y_train = feat_train[FEATURE_COLUMNS], feat_train["isFraud"]
    X_calib, y_calib = feat_calib[FEATURE_COLUMNS], feat_calib["isFraud"]
    X_test, y_test = feat_test[FEATURE_COLUMNS], feat_test["isFraud"]

    # class imbalance handled via sample weights (works for any sklearn-API
    # boosting library, unlike class_weight which not all of them support)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    print(f"training on {len(X_train)} rows, {y_train.sum()} positive "
          f"(pos_weight={pos_weight:.1f})...")
    model = build_model()
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # --- probability calibration on the held-out calibration slice ---
    # Platt scaling (logistic regression on the raw score) rather than
    # isotonic: with only ~1-2 hundred positives in the calibration slice,
    # isotonic regression collapses to a handful of step levels, which makes
    # cost-based threshold search coarse and unstable. Platt scaling assumes
    # less and stays smooth under this few-positives regime.
    raw_scores_calib = model.predict_proba(X_calib)[:, 1]
    calibrator = LogisticRegression()
    calibrator.fit(raw_scores_calib.reshape(-1, 1), y_calib)

    # SAFETY CHECK: Platt scaling can invert or degrade the ranking if fit
    # on a near-constant low-variance raw score. Diagnosed via
    # src/diagnose_comparison.py: an aggressively early-stopped XGBoost
    # model produced 14 unique raw scores, and the calibrator fit a
    # negative coefficient, flipping ROC-AUC from 0.97 to 0.03. This
    # model (sklearn GradientBoostingClassifier) hasn't shown this failure
    # mode, but the check is cheap and catches a future hyperparameter
    # change that triggers it.
    calib_coef = calibrator.coef_[0][0]
    n_unique_raw = len(np.unique(raw_scores_calib))
    calibrated_calib = calibrator.predict_proba(raw_scores_calib.reshape(-1, 1))[:, 1]
    raw_auc = roc_auc_score(y_calib, raw_scores_calib)
    cal_auc = roc_auc_score(y_calib, calibrated_calib)

    if calib_coef < 0:
        raise RuntimeError(
            f"Platt calibration inverted the ranking (coef={calib_coef:.4f}). "
            f"Raw calib scores had only {n_unique_raw} unique values "
            f"(std={raw_scores_calib.std():.2e}) -- too degenerate to calibrate "
            f"reliably. Refusing to save a model with corrupted calibration."
        )
    if cal_auc + 0.01 < raw_auc:
        raise RuntimeError(
            f"Platt calibration degraded the ranking: raw AUC {raw_auc:.4f} "
            f"-> calibrated {cal_auc:.4f}. Coef was {calib_coef:.4f} "
            f"(positive, so this is not the inversion case). Refusing to "
            f"save a model whose calibrated scores rank worse than its raw "
            f"scores."
        )
    print(f"Calibration check passed (coef={calib_coef:.4f}, "
          f"{n_unique_raw} unique raw calib scores, "
          f"rank_auc {raw_auc:.4f} -> {cal_auc:.4f})")

    raw_scores_test = model.predict_proba(X_test)[:, 1]
    calibrated_test = calibrator.predict_proba(raw_scores_test.reshape(-1, 1))[:, 1]

    # --- evaluation ---
    pr_auc = average_precision_score(y_test, calibrated_test)
    roc_auc = roc_auc_score(y_test, calibrated_test)
    brier = brier_score_loss(y_test, calibrated_test)
    baseline_pr_auc = y_test.mean()  # PR-AUC of a random classifier = positive rate

    print(f"\nPR-AUC (test)  : {pr_auc:.4f}  (random baseline = {baseline_pr_auc:.5f})")
    print(f"ROC-AUC (test) : {roc_auc:.4f}")
    print(f"Brier score    : {brier:.5f}  (lower is better-calibrated)")

    # --- cost-based thresholding, fit on the CALIB slice, reported on TEST ---
    amounts_calib = raw_calib["amount"].to_numpy()
    best_t, best_cost_calib, _ = find_cost_optimal_threshold(
        y_calib.to_numpy(), calibrated_calib, amounts_calib
    )

    amounts_test = raw_test["amount"].to_numpy()
    pred_test = calibrated_test >= best_t
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
    fraud_amount_caught = amounts_test[(pred_test) & (y_test == 1)].sum()
    fraud_amount_missed = amounts_test[(~pred_test) & (y_test == 1)].sum()
    total_fraud_amount = amounts_test[y_test == 1].sum()

    # cost of doing nothing (approve everything) vs. cost with the model
    cost_do_nothing = total_fraud_amount
    cost_with_model = fraud_amount_missed + fp * FLAT_COST_PER_FALSE_POSITIVE

    print(f"\nCost-optimal threshold: {best_t:.4f}")
    print(f"Confusion matrix (test) — TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Fraud amount caught: ₹{fraud_amount_caught:,.0f} / ₹{total_fraud_amount:,.0f} "
          f"({fraud_amount_caught/total_fraud_amount:.1%})")
    print(f"Legit transactions wrongly blocked: {fp} "
          f"({fp/(fp+tn):.3%} of legit test traffic)")
    print(f"Estimated cost — do nothing: ₹{cost_do_nothing:,.0f}  "
          f"with model: ₹{cost_with_model:,.0f}  "
          f"(₹{cost_do_nothing - cost_with_model:,.0f} saved on this test slice)")

    # --- save artifacts ---
    joblib.dump(model, f"{ARTIFACT_DIR}/model.joblib")
    joblib.dump(calibrator, f"{ARTIFACT_DIR}/calibrator.joblib")
    with open(f"{ARTIFACT_DIR}/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    metrics = {
        "model_type": "sklearn_gbm",
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "baseline_pr_auc_random": baseline_pr_auc,
        "cost_optimal_threshold": best_t,
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "fraud_amount_caught_pct": fraud_amount_caught / total_fraud_amount,
        "false_positive_rate_on_legit": fp / (fp + tn),
        "cost_do_nothing_inr": float(cost_do_nothing),
        "cost_with_model_inr": float(cost_with_model),
        "flat_cost_per_false_positive_inr": FLAT_COST_PER_FALSE_POSITIVE,
        "n_train": len(X_train), "n_calib": len(X_calib), "n_test": len(X_test),
        "train_positive_rate": float(y_train.mean()),
    }
    with open(f"{ARTIFACT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved artifacts to {ARTIFACT_DIR}/  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()