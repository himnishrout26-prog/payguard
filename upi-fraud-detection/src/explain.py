"""
Per-transaction explainability: "top contributing factors" for a risk score.

Three model families are supported, each with the strongest exact method
available for it:

  - sklearn GradientBoostingClassifier: hand-rolled Saabas walk over
    `.estimators_` (the standard treeinterpreter algorithm). Exact for a
    single tree, additive across the ensemble, decomposition equals the
    model's raw log-odds prediction.
  - LightGBM: native `predict(..., pred_contrib=True)`. Returns exact
    TreeSHAP values in log-odds space; last element is the bias term.
  - XGBoost: `booster.predict(DMatrix, pred_contribs=True)`. Same shape
    and interpretation as LightGBM.

All three return (raw_margin_in_logodds, bias_logodds, [factor, ...]).
The factor list is sorted by absolute contribution, so the same
downstream code renders explanations for any winner.
"""
import numpy as np


# ----------------------------------------------------------------------
# sklearn GradientBoostingClassifier: Saabas walk
# ----------------------------------------------------------------------

def _tree_contributions(tree, x_row):
    """Walk one fitted DecisionTreeRegressor's structure for a sample and
    return {feature_index: contribution}. Contributions are in log-odds
    space (GBC fits its internal trees to the negative gradient of
    log-loss)."""
    t = tree.tree_
    contributions = {}
    node = 0
    while t.children_left[node] != t.children_right[node]:  # not a leaf
        feat = t.feature[node]
        thresh = t.threshold[node]
        child = t.children_left[node] if x_row[feat] <= thresh else t.children_right[node]
        delta = t.value[child][0, 0] - t.value[node][0, 0]
        contributions[feat] = contributions.get(feat, 0.0) + delta
        node = child
    return contributions


def _explain_sklearn_gbm(model, x_row, feature_names, top_k):
    total = {}
    for stage in model.estimators_:  # shape (n_estimators, 1) for binary
        tree = stage[0]
        for feat_idx, val in _tree_contributions(tree, x_row).items():
            total[feat_idx] = total.get(feat_idx, 0.0) + model.learning_rate * val

    # init_ is a DummyClassifier returning the training-set prior; convert
    # to log-odds to match the space the trees contribute in.
    prior = model.init_.predict_proba(x_row.reshape(1, -1))[0]
    bias_logodds = float(np.log(prior[1] / max(prior[0], 1e-12)))

    ranked = sorted(total.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    factors = [
        {
            "feature": feature_names[idx],
            "contribution": round(float(val), 4),
            "value": round(float(x_row[idx]), 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        }
        for idx, val in ranked
    ]
    raw_margin = bias_logodds + sum(total.values())
    return raw_margin, bias_logodds, factors


# ----------------------------------------------------------------------
# LightGBM: native pred_contrib
# ----------------------------------------------------------------------

def _explain_lightgbm(model, x_df, feature_names, top_k):
    # Returns shape (1, n_features + 1); last column is the bias term.
    contribs = model.predict(x_df, pred_contrib=True)[0]
    *feature_contribs, bias_logodds = contribs

    # If the wrapper returns "raw" contributions rather than log-odds, the
    # vector still sums correctly to the raw margin; interpretation is the
    # same either way for ranking purposes.
    ranked_idx = np.argsort(-np.abs(feature_contribs))[:top_k]
    factors = [
        {
            "feature": feature_names[i],
            "contribution": round(float(feature_contribs[i]), 4),
            "value": round(float(x_df.iloc[0, i]), 4),
            "direction": "increases_risk" if feature_contribs[i] > 0 else "decreases_risk",
        }
        for i in ranked_idx
    ]
    raw_margin = float(np.sum(contribs))
    return raw_margin, float(bias_logodds), factors


# ----------------------------------------------------------------------
# XGBoost: booster pred_contribs
# ----------------------------------------------------------------------

def _explain_xgboost(model, x_df, feature_names, top_k):
    import xgboost as xgb
    booster = model.get_booster()
    dm = xgb.DMatrix(x_df, feature_names=list(x_df.columns))
    contribs = booster.predict(dm, pred_contribs=True)[0]
    *feature_contribs, bias_logodds = contribs
    feature_contribs = np.asarray(feature_contribs)

    ranked_idx = np.argsort(-np.abs(feature_contribs))[:top_k]
    factors = [
        {
            "feature": feature_names[i],
            "contribution": round(float(feature_contribs[i]), 4),
            "value": round(float(x_df.iloc[0, i]), 4),
            "direction": "increases_risk" if feature_contribs[i] > 0 else "decreases_risk",
        }
        for i in ranked_idx
    ]
    raw_margin = float(np.sum(contribs))
    return raw_margin, float(bias_logodds), factors


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def explain_prediction(model, x, feature_names, top_k=5):
    """
    model: any of {GradientBoostingClassifier, LGBMClassifier, XGBClassifier}
    x: 1D numpy array OR a 1-row DataFrame. DataFrames are preferred for
       LightGBM/XGBoost (feature-name alignment); a numpy array is accepted
       and converted internally.
    returns: (raw_margin, bias, [{"feature", "contribution", "value", "direction"}, ...])
    """
    import pandas as pd

    model_name = type(model).__name__

    if model_name == "GradientBoostingClassifier":
        x_arr = x if isinstance(x, np.ndarray) else x.iloc[0].to_numpy(dtype=float)
        return _explain_sklearn_gbm(model, x_arr, feature_names, top_k)

    if model_name == "LGBMClassifier":
        x_df = x if isinstance(x, pd.DataFrame) else pd.DataFrame([x], columns=feature_names)
        return _explain_lightgbm(model, x_df, feature_names, top_k)

    if model_name == "XGBClassifier":
        x_df = x if isinstance(x, pd.DataFrame) else pd.DataFrame([x], columns=feature_names)
        return _explain_xgboost(model, x_df, feature_names, top_k)

    raise TypeError(
        f"explain_prediction does not know how to walk model type "
        f"'{model_name}'. Supported: GradientBoostingClassifier, "
        f"LGBMClassifier, XGBClassifier."
    )