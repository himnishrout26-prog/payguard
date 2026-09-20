"""
Export the trained model + calibrator to a JSON blob that the browser-side
console (payguard.html) can walk without a Python backend.

Every family is normalized to the same flat-array format:

  {
    "model_type": "...",
    "feature_names": [...],
    "bias": float,                 # global log-odds bias
    "trees": [
      {
        "shrinkage": float,        # per-tree learning rate
        "feature":    [int...],    # -1 for leaf nodes
        "threshold":  [float...],
        "left":       [int...],
        "right":      [int...],
        "leaf_value":     [float...],
        "internal_value": [float...]
      },
      ...
    ],
    "calib_coef": float,
    "calib_intercept": float,
    "threshold": float,
    "metrics": {...}
  }

The exporter VERIFIES the exported trees reproduce booster.predict() on a
sample of training rows before writing. If reconstruction is off by more
than 1e-3, it refuses to write rather than emit a silently-wrong browser
model.
"""
import json
import joblib
import numpy as np

from config import DATA_PATH, ARTIFACT_DIR
from features import build_features, FEATURE_COLUMNS


# ----------------------------------------------------------------------
# Verify the flattened trees reproduce the model's raw score
# ----------------------------------------------------------------------

def _eval_flat_tree(tree, x):
    node = tree["root"]
    while tree["feature"][node] >= 0:
        f = tree["feature"][node]
        if x[f] <= tree["threshold"][node]:
            node = tree["left"][node]
        else:
            node = tree["right"][node]
    return tree["leaf_value"][node]


def _eval_flat_model(export, x):
    raw = export["bias"]
    for tree in export["trees"]:
        raw += tree["shrinkage"] * _eval_flat_tree(tree, x)
    return raw


def _verify_export(export, X_check_np, expected_raw, model_name):
    max_err = 0.0
    for i in range(len(X_check_np)):
        got = _eval_flat_model(export, X_check_np[i])
        err = abs(got - expected_raw[i])
        max_err = max(max_err, err)
    print(f"    [verify] {model_name}: max |flattened - model| = {max_err:.2e} "
          f"over {len(X_check_np)} rows")
    return max_err < 1e-3


# ----------------------------------------------------------------------
# sklearn GradientBoostingClassifier
# ----------------------------------------------------------------------

def _flatten_sklearn_tree(sklearn_tree):
    t = sklearn_tree.tree_
    n_nodes = t.node_count
    feature = [int(f) for f in t.feature]
    threshold = [float(v) for v in t.threshold]
    left = [int(v) for v in t.children_left]
    right = [int(v) for v in t.children_right]
    # sklearn stores value as shape (n_nodes, 1, 1) in log-odds
    internal_value = [float(v[0]) for v in t.value]
    leaf_value = list(internal_value)
    for i in range(n_nodes):
        if left[i] != right[i]:
            leaf_value[i] = 0.0
    return {
        "shrinkage": None,  # set by caller from model.learning_rate
        "feature": feature,
        "threshold": threshold,
        "left": left,
        "right": right,
        "leaf_value": leaf_value,
        "internal_value": internal_value,
        "root": 0,
    }


def export_sklearn_gbm(model, X_check_df):
    dummy = np.zeros((1, len(FEATURE_COLUMNS)))
    bias_proba = model.init_.predict_proba(dummy)[0]
    bias = float(np.log(bias_proba[1] / max(bias_proba[0], 1e-12)))

    trees = []
    for stage in model.estimators_:
        tree = _flatten_sklearn_tree(stage[0])
        tree["shrinkage"] = float(model.learning_rate)
        trees.append(tree)

    export = {"model_type": "sklearn_gbm", "bias": bias, "trees": trees}
    expected_raw = model.decision_function(X_check_df)
    return export, expected_raw


# ----------------------------------------------------------------------
# LightGBM
# ----------------------------------------------------------------------

def _flatten_lightgbm_node(node, arrays):
    idx = len(arrays["feature"])
    arrays["feature"].append(-1)
    arrays["threshold"].append(0.0)
    arrays["left"].append(-1)
    arrays["right"].append(-1)
    arrays["leaf_value"].append(0.0)
    arrays["internal_value"].append(0.0)

    if "leaf_value" in node:
        arrays["leaf_value"][idx] = float(node["leaf_value"])
        arrays["internal_value"][idx] = float(node["leaf_value"])
    else:
        arrays["feature"][idx] = int(node["split_feature"])
        arrays["threshold"][idx] = float(node["threshold"])
        arrays["internal_value"][idx] = float(node.get("internal_value", 0.0))
        arrays["left"][idx] = _flatten_lightgbm_node(node["left_child"], arrays)
        arrays["right"][idx] = _flatten_lightgbm_node(node["right_child"], arrays)
    return idx


def export_lightgbm(model, X_check_df):
    dump = model.booster_.dump_model()
    trees = []
    for tree_info in dump["tree_info"]:
        arrays = {
            # LightGBM's leaf values in dump_model() are ALREADY multiplied
            # by the learning rate at training time. The "shrinkage" field
            # in the dump is informational, not something to reapply.
            # (Confirmed by the verify step: applying it again produced a
            # 7.45-unit discrepancy on the first export attempt.)
            "shrinkage": 1.0,
            "feature": [], "threshold": [], "left": [], "right": [],
            "leaf_value": [], "internal_value": [],
        }
        arrays["root"] = _flatten_lightgbm_node(tree_info["tree_structure"], arrays)
        trees.append(arrays)

    X_np = X_check_df.to_numpy(dtype=float) if hasattr(X_check_df, "to_numpy") else np.asarray(X_check_df)

    # LightGBM's binary classifier adds an initial log-odds score via
    # boost_from_average that is NOT represented by any tree. Solve for it
    # empirically: bias = mean(expected_raw - sum_of_trees).
    tree_only = np.array([
        sum(_eval_flat_tree(t, X_np[i]) for t in trees)
        for i in range(len(X_np))
    ])
    expected_raw = model.predict(X_check_df, raw_score=True)
    bias = float(np.mean(expected_raw - tree_only))

    export = {"model_type": "lightgbm", "bias": bias, "trees": trees}
    return export, expected_raw


# ----------------------------------------------------------------------
# XGBoost
# ----------------------------------------------------------------------

def _flatten_xgboost_node(node, arrays):
    idx = len(arrays["feature"])
    arrays["feature"].append(-1)
    arrays["threshold"].append(0.0)
    arrays["left"].append(-1)
    arrays["right"].append(-1)
    arrays["leaf_value"].append(0.0)
    arrays["internal_value"].append(0.0)

    if "leaf" in node:
        arrays["leaf_value"][idx] = float(node["leaf"])
        arrays["internal_value"][idx] = float(node["leaf"])
        return idx

    split_feat = int(node["split"][1:])
    arrays["feature"][idx] = split_feat
    arrays["threshold"][idx] = float(node["split_condition"])
    children = node["children"]
    yes = children[0]
    no = children[1] if len(children) > 1 else children[0]
    arrays["left"][idx] = _flatten_xgboost_node(yes, arrays)
    arrays["right"][idx] = _flatten_xgboost_node(no, arrays)
    return idx


def export_xgboost(model, X_check_df):
    import xgboost as xgb
    booster = model.get_booster()
    dump = booster.get_dump(dump_format="json")

    config = json.loads(booster.save_config())
    bias = float(config["learner"]["learner_model_param"]["base_score"])
    if "objective" in config["learner"]["objective"]:
        name = config["learner"]["objective"]["name"]
        if "logistic" in name and 0 < bias < 1:
            bias = float(np.log(bias / (1 - bias)))

    trees = []
    for tree_json in dump:
        node = json.loads(tree_json)
        arrays = {
            "shrinkage": float(model.get_params().get("learning_rate", 0.1)),
            "feature": [], "threshold": [], "left": [], "right": [],
            "leaf_value": [], "internal_value": [],
        }
        arrays["root"] = _flatten_xgboost_node(node, arrays)
        trees.append(arrays)

    export = {"model_type": "xgboost", "bias": bias, "trees": trees}
    dm = xgb.DMatrix(X_check_df, feature_names=list(FEATURE_COLUMNS))
    expected_raw = booster.predict(dm, output_margin=True)
    return export, expected_raw


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

EXPORTERS = {
    "GradientBoostingClassifier": export_sklearn_gbm,
    "LGBMClassifier": export_lightgbm,
    "XGBClassifier": export_xgboost,
}


def main():
    model = joblib.load(f"{ARTIFACT_DIR}/model.joblib")
    calibrator = joblib.load(f"{ARTIFACT_DIR}/calibrator.joblib")
    metrics = json.load(open(f"{ARTIFACT_DIR}/metrics.json"))
    feature_cols = json.load(open(f"{ARTIFACT_DIR}/feature_columns.json"))

    model_name = type(model).__name__
    print(f"Exporting model type: {model_name}")

    if model_name not in EXPORTERS:
        raise SystemExit(
            f"Browser export does not support model type '{model_name}'. "
            f"Supported: {list(EXPORTERS)}."
        )

    import pandas as pd
    print("  Loading small sample of training data for verification...")
    raw = pd.read_csv(DATA_PATH).sort_values("step").reset_index(drop=True)
    sample = raw.iloc[:5000]
    feat = build_features(sample)
    X_check_df = feat[FEATURE_COLUMNS]  # keep as DataFrame for LightGBM predict
    X_check_np = X_check_df.to_numpy(dtype=float)

    print(f"  Flattening {model_name} trees...")
    export, expected_raw = EXPORTERS[model_name](model, X_check_df)

    print("  Verifying export (this is the important step)...")
    ok = _verify_export(export, X_check_np, expected_raw, model_name)
    if not ok:
        raise SystemExit(
            "Refusing to write a browser export that does not reproduce the "
            "Python model. Investigate the flattening for this model type."
        )

    export["feature_names"] = feature_cols
    export["calib_coef"] = float(calibrator.coef_[0][0])
    export["calib_intercept"] = float(calibrator.intercept_[0])
    export["threshold"] = metrics["cost_optimal_threshold"]
    export["metrics"] = metrics

    json_path = f"{ARTIFACT_DIR}/model_export.json"
    js_path = f"{ARTIFACT_DIR}/model_export.js"

    with open(json_path, "w") as f:
        json.dump(export, f)
    print(f"  Wrote {json_path}")

    with open(js_path, "w") as f:
        f.write("window.MODEL_EXPORT = ")
        json.dump(export, f)
        f.write(";\n")
    print(f"  Wrote {js_path}")

    n_trees = len(export["trees"])
    total_nodes = sum(len(t["feature"]) for t in export["trees"])
    print(f"Done. {n_trees} trees, {total_nodes} nodes total.")


if __name__ == "__main__":
    main()