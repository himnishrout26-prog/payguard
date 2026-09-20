"""
Diagnose the XGBoost failure mode and verify the fix.

Two independent causes of the ROC-AUC ~0.0002 failure on the test slice:

  (A) Early stopping locks `best_iteration` to a very small number (3-4
      in one run, 34 in another -- the argmax is noise-driven). The
      sklearn wrapper then restricts predict_proba to only those first
      few trees, which cannot separate the harder test slice.
  (B) Platt scaling fit on a near-constant score vector can learn a slope
      that INVERTS the ranking on data it wasn't fit on.

Run from src/:
    python3 diagnose_comparison.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from config import DATA_PATH
from features import build_features, FEATURE_COLUMNS

SAMPLE_N = 500_000
SPLIT = (0.55, 0.15, 0.15)  # train / val / calib / test


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def xgb_predict_all_trees(model, X):
    """Force prediction with every boosting round, bypassing the wrapper's
    restriction to best_iteration when early stopping was used."""
    import xgboost as xgb
    booster = model.get_booster()
    feat_names = list(X.columns) if hasattr(X, "columns") else None
    dm = xgb.DMatrix(X, feature_names=feat_names)
    return booster.predict(dm, iteration_range=(0, 0))  # (0, 0) == all trees


def print_score_stats(label, scores):
    scores = np.asarray(scores).reshape(-1)
    n_unique = len(np.unique(scores))
    print(f"  {label:28s} min={scores.min():.6f} max={scores.max():.6f} "
          f"mean={scores.mean():.6f} std={scores.std():.6f} unique={n_unique}")
    return n_unique


def calibration_report(name, raw_calib, y_calib, raw_test, y_test):
    """Compare raw vs Platt vs isotonic on the test slice. Ranking metrics
    are threshold-free; only score ORDERING matters. Any calibrator that
    reduces ROC-AUC has reordered rows."""
    print(f"\n--- {name}: calibration comparison ---")
    print(f"  raw        ROC-AUC={roc_auc_score(y_test, raw_test):.4f}  "
          f"PR-AUC={average_precision_score(y_test, raw_test):.4f}")

    lr = LogisticRegression().fit(raw_calib.reshape(-1, 1), y_calib)
    platt_test = lr.predict_proba(raw_test.reshape(-1, 1))[:, 1]
    platt_auc = roc_auc_score(y_test, platt_test)
    print(f"  platt      ROC-AUC={platt_auc:.4f}  "
          f"PR-AUC={average_precision_score(y_test, platt_test):.4f}  "
          f"coef={lr.coef_[0, 0]:+.3e}  intercept={lr.intercept_[0]:+.3e}")
    if platt_auc < 0.5:
        print(f"  ^^^ PLATT IS INVERTED (coef sign flips ranking on test)")

    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_calib, y_calib)
    iso_test = iso.predict(raw_test)
    print(f"  isotonic   ROC-AUC={roc_auc_score(y_test, iso_test):.4f}  "
          f"PR-AUC={average_precision_score(y_test, iso_test):.4f}")

    print(f"  calib raw std={raw_calib.std():.3e}  "
          f"calib unique={len(np.unique(raw_calib))}")


# ----------------------------------------------------------------------
# Load and split
# ----------------------------------------------------------------------

print(f"Loading {SAMPLE_N:,} rows...")
raw = pd.read_csv(DATA_PATH).sort_values("step").reset_index(drop=True)
raw = raw.iloc[:SAMPLE_N]

n = len(raw)
n_train = int(n * SPLIT[0])
n_val = int(n * SPLIT[1])
n_calib = int(n * SPLIT[2])
train_end = n_train
val_end = train_end + n_val
calib_end = val_end + n_calib

feat_all = build_features(raw)
X_train = feat_all.iloc[:train_end][FEATURE_COLUMNS]
y_train = feat_all.iloc[:train_end]["isFraud"]
X_val = feat_all.iloc[train_end:val_end][FEATURE_COLUMNS]
y_val = feat_all.iloc[train_end:val_end]["isFraud"]
X_calib = feat_all.iloc[val_end:calib_end][FEATURE_COLUMNS]
y_calib = feat_all.iloc[val_end:calib_end]["isFraud"]
X_test = feat_all.iloc[calib_end:][FEATURE_COLUMNS]
y_test = feat_all.iloc[calib_end:]["isFraud"]

pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
sample_weight = np.where(y_train == 1, pos_weight, 1.0)

print(f"train={len(X_train)}  val={len(X_val)}  calib={len(X_calib)}  test={len(X_test)}")
print(f"positives: train={int(y_train.sum())}  val={int(y_val.sum())}  "
      f"calib={int(y_calib.sum())}  test={int(y_test.sum())}")
print(f"pos_weight={pos_weight:.1f}")


# ----------------------------------------------------------------------
# XGBoost ORIGINAL: early stopping on, wrapper predict uses best_iteration
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("XGBoost: ORIGINAL CONFIG (early stopping on, best_iteration used for predict)")
print("=" * 70)
try:
    from xgboost import XGBClassifier

    model_orig = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        min_child_weight=20, early_stopping_rounds=100, random_state=42,
    )
    model_orig.fit(
        X_train, y_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], verbose=False,
    )

    print(f"  best_iteration = {getattr(model_orig, 'best_iteration', None)}")
    print(f"  best_score     = {getattr(model_orig, 'best_score', None)}")
    print(f"  classes_       = {model_orig.classes_}")

    raw_test_orig = model_orig.predict_proba(X_test)[:, 1]
    raw_calib_orig = model_orig.predict_proba(X_calib)[:, 1]

    print("\n  Score stats (wrapper default predict_proba -> uses best_iteration trees):")
    print_score_stats("calib raw (orig)", raw_calib_orig)
    print_score_stats("test  raw (orig)", raw_test_orig)

    print(f"\n  test ROC-AUC (raw, orig config): "
          f"{roc_auc_score(y_test, raw_test_orig):.4f}")
    print(f"  test PR-AUC  (raw, orig config): "
          f"{average_precision_score(y_test, raw_test_orig):.4f}")

    calibration_report("ORIGINAL", raw_calib_orig, y_calib, raw_test_orig, y_test)
except Exception as e:
    print(f"XGBoost original-config diagnosis failed: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# XGBoost FIXED: no early stopping, predict with all trees
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("XGBoost: FIXED CONFIG (no early stopping, predict with all trees)")
print("=" * 70)
try:
    from xgboost import XGBClassifier

    model_fixed = XGBClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        min_child_weight=20, max_delta_step=1, random_state=42,
        # NO early_stopping_rounds
    )
    model_fixed.fit(
        X_train, y_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], verbose=False,
    )

    print(f"  best_iteration = {getattr(model_fixed, 'best_iteration', None)} "
          f"(should be None when early stopping is off)")
    print(f"  classes_       = {model_fixed.classes_}")

    raw_test_fixed = xgb_predict_all_trees(model_fixed, X_test)
    raw_calib_fixed = xgb_predict_all_trees(model_fixed, X_calib)
    wrapper_test = model_fixed.predict_proba(X_test)[:, 1]

    print("\n  Score stats:")
    print_score_stats("calib raw (fixed)", raw_calib_fixed)
    print_score_stats("test  raw (fixed)", raw_test_fixed)

    same = np.allclose(raw_test_fixed, wrapper_test, atol=1e-9)
    print(f"\n  all-trees prediction matches wrapper predict_proba: {same}")

    print(f"\n  test ROC-AUC (raw, fixed): "
          f"{roc_auc_score(y_test, raw_test_fixed):.4f}")
    print(f"  test PR-AUC  (raw, fixed): "
          f"{average_precision_score(y_test, raw_test_fixed):.4f}")

    calibration_report("FIXED", raw_calib_fixed, y_calib, raw_test_fixed, y_test)

    fraud_idx = y_test[y_test == 1].index[:3]
    legit_idx = y_test[y_test == 0].index[:3]
    print(f"\n  scores on 3 known FRAUD rows: "
          f"{xgb_predict_all_trees(model_fixed, feat_all.loc[fraud_idx, FEATURE_COLUMNS])}")
    print(f"  scores on 3 known LEGIT rows: "
          f"{xgb_predict_all_trees(model_fixed, feat_all.loc[legit_idx, FEATURE_COLUMNS])}")
except Exception as e:
    print(f"XGBoost fixed-config diagnosis failed: {type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# LightGBM reference
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("LightGBM (reference: early stopping works, scores non-degenerate)")
print("=" * 70)
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier

    model_lgb = LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=-1,
    )
    model_lgb.fit(
        X_train, y_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  best_iteration_ = {model_lgb.best_iteration_}")
    print(f"  num_trees       = {model_lgb.booster_.num_trees()}")

    raw_test_lgb = model_lgb.predict_proba(X_test)[:, 1]
    raw_calib_lgb = model_lgb.predict_proba(X_calib)[:, 1]

    print("\n  Score stats:")
    print_score_stats("calib raw (lgb)", raw_calib_lgb)
    print_score_stats("test  raw (lgb)", raw_test_lgb)

    print(f"\n  test ROC-AUC (raw): {roc_auc_score(y_test, raw_test_lgb):.4f}")
    print(f"  test PR-AUC  (raw): {average_precision_score(y_test, raw_test_lgb):.4f}")

    calibration_report("LightGBM", raw_calib_lgb, y_calib, raw_test_lgb, y_test)
except Exception as e:
    print(f"LightGBM diagnosis failed: {type(e).__name__}: {e}")