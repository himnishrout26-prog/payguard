"""
Train sklearn's GradientBoostingClassifier, LightGBM, and XGBoost on the
same split, evaluate all three with the same cost-based threshold procedure,
and save the winner as the production artifacts (same filenames train.py
produces, so app.py / export_for_ui.py don't need to know which library won).

Split: 55% train / 15% val / 15% calib / 15% test.
  - val   is used ONLY for early stopping (LightGBM, sklearn's internal)
  - calib is used ONLY for Platt calibration and threshold search
  - test  is held out entirely
Previous version used a 3-way split with calib serving double duty as
val -- that optimistically biases the calibrator. Fixed here.

Run:
    pip install lightgbm xgboost
    python train_compare.py

Writes artifacts/comparison_metrics.json with all three models' numbers.
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    confusion_matrix,
)

from config import DATA_PATH, ARTIFACT_DIR
from features import build_features, FEATURE_COLUMNS

FLAT_COST_PER_FALSE_POSITIVE = 35.0
SPLIT_FRACTIONS = (0.55, 0.15, 0.15)  # remainder = test


# ----------------------------------------------------------------------
# Model builders / fitters
# ----------------------------------------------------------------------

def build_sklearn_gbm():
    return GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.08, max_depth=4, subsample=0.8,
        min_samples_leaf=20, random_state=42,
        validation_fraction=0.1, n_iter_no_change=15,
    )


def fit_sklearn_gbm(model, X_train, y_train, sample_weight, X_val, y_val):
    # sklearn's GBM handles its own internal validation_fraction split for
    # early stopping; X_val is ignored here by design.
    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, time.time() - t0


def build_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1,
    )


def fit_lightgbm(model, X_train, y_train, sample_weight, X_val, y_val):
    import lightgbm as lgb
    t0 = time.time()
    model.fit(
        X_train, y_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model, time.time() - t0


def build_xgboost():
    from xgboost import XGBClassifier
    # No early_stopping_rounds. Diagnosed in src/diagnose_comparison.py:
    # the val-slice aucpr argmax is noise-driven under this imbalance and
    # can lock best_iteration to 1 or 3, which restricts the wrapper's
    # predict_proba to only those trees. Fixed n_estimators sidesteps that.
    return XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        min_child_weight=20, max_delta_step=1, random_state=42,
    )


def fit_xgboost(model, X_train, y_train, sample_weight, X_val, y_val):
    t0 = time.time()
    model.fit(
        X_train, y_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], verbose=False,
    )
    return model, time.time() - t0


MODEL_REGISTRY = {
    "sklearn_gbm": (build_sklearn_gbm, fit_sklearn_gbm),
    "lightgbm":    (build_lightgbm,    fit_lightgbm),
    "xgboost":     (build_xgboost,     fit_xgboost),
}


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def fit_calibrator(raw_scores_calib, y_calib):
    """Fit Platt scaling. Returns (calibrator, ok, diagnostics_str).
    ok=False when the fit degenerates (negative coef or AUC-degrading)."""
    raw = np.asarray(raw_scores_calib).reshape(-1)
    cal = LogisticRegression()
    cal.fit(raw.reshape(-1, 1), y_calib)

    coef = float(cal.coef_[0][0])
    n_unique = len(np.unique(raw))
    diag = f"coef={coef:+.4f}  n_unique={n_unique}  std={raw.std():.3e}"

    if coef < 0:
        return cal, False, diag + "  [INVERTED: negative coef]"

    # Stronger check: does the calibrated score preserve the raw ranking
    # on the calib slice? A mis-fit Platt with a tiny positive coef can
    # still degrade ordering.
    calibrated = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    raw_auc = roc_auc_score(y_calib, raw)
    cal_auc = roc_auc_score(y_calib, calibrated)
    if cal_auc + 0.01 < raw_auc:
        return cal, False, diag + f"  [DEGRADES RANKING: {raw_auc:.4f}->{cal_auc:.4f}]"

    return cal, True, diag + f"  rank_auc={cal_auc:.4f}"


# ----------------------------------------------------------------------
# Threshold search
# ----------------------------------------------------------------------

def find_cost_optimal_threshold(y_true, proba, amounts, fp_cost=FLAT_COST_PER_FALSE_POSITIVE):
    """Returns (threshold, cost). threshold is None when proba is
    degenerate (fewer than 2 unique quantiles) — in that case no meaningful
    threshold exists and the caller should skip the model."""
    thresholds = np.unique(np.quantile(proba, np.linspace(0, 1, 500)))
    if len(thresholds) < 2:
        return None, np.inf

    best_t, best_cost = None, np.inf
    for t in thresholds:
        pred = proba >= t
        fn_mask = (~pred) & (y_true == 1)
        fp_mask = pred & (y_true == 0)
        cost = amounts[fn_mask].sum() + fp_mask.sum() * fp_cost
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(model, calibrator, X_test, y_test, amounts_test, n_latency_samples=500):
    raw_scores = model.predict_proba(X_test)[:, 1]
    calibrated = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    pr_auc = average_precision_score(y_test, calibrated)
    roc_auc = roc_auc_score(y_test, calibrated)
    brier = brier_score_loss(y_test, calibrated)

    # Latency: single-row calls, the same path a /score request takes.
    # DataFrames, not numpy, so LightGBM doesn't warn about feature names.
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_test), size=min(n_latency_samples, len(X_test)), replace=False)
    for i in idx[:10]:
        model.predict_proba(X_test.iloc[[int(i)]])
    times = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict_proba(X_test.iloc[[int(i)]])
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)

    return {
        "pr_auc": float(pr_auc), "roc_auc": float(roc_auc), "brier_score": float(brier),
        "latency_ms_mean": float(times.mean()),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "calibrated_scores": calibrated,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    raw = pd.read_csv(DATA_PATH).sort_values("step").reset_index(drop=True)
    n = len(raw)
    n_train = int(n * SPLIT_FRACTIONS[0])
    n_val = int(n * SPLIT_FRACTIONS[1])
    n_calib = int(n * SPLIT_FRACTIONS[2])
    train_end = n_train
    val_end = train_end + n_val
    calib_end = val_end + n_calib
    print(f"Loaded {n} rows. train={n_train} val={n_val} "
          f"calib={n_calib} test={n - calib_end}")

    feat_all = build_features(raw)
    raw_train = raw.iloc[:train_end]
    raw_val = raw.iloc[train_end:val_end]
    raw_calib = raw.iloc[val_end:calib_end]
    raw_test = raw.iloc[calib_end:]

    feat_train = feat_all.iloc[:train_end]
    feat_val = feat_all.iloc[train_end:val_end]
    feat_calib = feat_all.iloc[val_end:calib_end]
    feat_test = feat_all.iloc[calib_end:]

    X_train, y_train = feat_train[FEATURE_COLUMNS], feat_train["isFraud"]
    X_val, y_val = feat_val[FEATURE_COLUMNS], feat_val["isFraud"]
    X_calib, y_calib = feat_calib[FEATURE_COLUMNS], feat_calib["isFraud"]
    X_test, y_test = feat_test[FEATURE_COLUMNS], feat_test["isFraud"]
    amounts_calib = raw_calib["amount"].to_numpy()
    amounts_test = raw_test["amount"].to_numpy()

    print(f"positives: train={int(y_train.sum())} val={int(y_val.sum())} "
          f"calib={int(y_calib.sum())} test={int(y_test.sum())}")

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    results = {}
    fitted_models = {}

    for name, (build_fn, fit_fn) in MODEL_REGISTRY.items():
        try:
            model = build_fn()
        except ImportError:
            print(f"\n[{name}] not installed, skipping.")
            continue

        print(f"\n[{name}] training...")
        model, train_seconds = fit_fn(model, X_train, y_train, sample_weight, X_val, y_val)

        raw_scores_calib = model.predict_proba(X_calib)[:, 1]
        calibrator, ok, diag = fit_calibrator(raw_scores_calib, y_calib)
        print(f"[{name}] calibrator: {diag}")
        if not ok:
            print(f"[{name}] skipping: calibrator would corrupt the ranking.")
            continue

        metrics = evaluate(model, calibrator, X_test, y_test, amounts_test)
        calibrated_calib = calibrator.predict_proba(raw_scores_calib.reshape(-1, 1))[:, 1]
        best_t, _ = find_cost_optimal_threshold(y_calib.to_numpy(), calibrated_calib, amounts_calib)
        if best_t is None:
            print(f"[{name}] skipping: calibrated scores are degenerate, no usable threshold.")
            continue

        pred_test = metrics.pop("calibrated_scores") >= best_t
        tn, fp, fn, tp = confusion_matrix(y_test, pred_test).ravel()
        fraud_caught = amounts_test[pred_test & (y_test == 1)].sum()
        fraud_total = amounts_test[y_test == 1].sum()

        metrics.update({
            "train_seconds": round(train_seconds, 1),
            "threshold": float(best_t),
            "fraud_amount_caught_pct": float(fraud_caught / fraud_total) if fraud_total > 0 else None,
            "false_positive_rate": float(fp / (fp + tn)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        })
        results[name] = metrics
        fitted_models[name] = (model, calibrator)

        print(f"[{name}] PR-AUC={metrics['pr_auc']:.4f}  train={train_seconds:.1f}s  "
              f"latency_p95={metrics['latency_ms_p95']:.3f}ms  "
              f"fraud_caught={metrics['fraud_amount_caught_pct']:.1%}  FPs={fp}")

    if not results:
        raise SystemExit("No model produced usable results.")

    print("\n=== Comparison ===")
    for name, m in results.items():
        print(f"{name:12s}  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
              f"train={m['train_seconds']:>6.1f}s  "
              f"latency_p95={m['latency_ms_p95']:.3f}ms  "
              f"fraud_caught={m['fraud_amount_caught_pct']:.1%}  "
              f"FPs={m['confusion_matrix']['fp']}")

    winner = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\nWinner (highest PR-AUC): {winner}")

    with open(ARTIFACT_DIR / "comparison_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    model, calibrator = fitted_models[winner]
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(calibrator, ARTIFACT_DIR / "calibrator.joblib")
    with open(ARTIFACT_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)

    full_metrics = dict(results[winner])
    full_metrics["model_type"] = winner
    full_metrics["cost_optimal_threshold"] = full_metrics.pop("threshold")
    full_metrics["test_confusion_matrix"] = full_metrics.pop("confusion_matrix")
    full_metrics["n_train"] = len(X_train)
    full_metrics["n_val"] = len(X_val)
    full_metrics["n_calib"] = len(X_calib)
    full_metrics["n_test"] = len(X_test)
    full_metrics["train_positive_rate"] = float(y_train.mean())
    full_metrics["baseline_pr_auc_random"] = float(y_test.mean())
    full_metrics["flat_cost_per_false_positive_inr"] = FLAT_COST_PER_FALSE_POSITIVE
    fraud_total_amt = float(amounts_test[y_test == 1].sum())
    full_metrics["cost_do_nothing_inr"] = fraud_total_amt
    full_metrics["cost_with_model_inr"] = (
        fraud_total_amt * (1 - full_metrics["fraud_amount_caught_pct"])
        + full_metrics["test_confusion_matrix"]["fp"] * FLAT_COST_PER_FALSE_POSITIVE
    )

    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(f"\nSaved {winner} as production model "
          f"(artifacts/model.joblib, calibrator.joblib, metrics.json)")
    print("Full comparison table saved to artifacts/comparison_metrics.json")

    if winner != "sklearn_gbm":
        print(f"\nNOTE: {winner} won. app.py and /score work correctly with "
              f"this model (explain.py dispatches on model type). "
              f"export_for_ui.py will refuse to write the browser export "
              f"until payguard.html learns the {winner} tree format.")


if __name__ == "__main__":
    main()