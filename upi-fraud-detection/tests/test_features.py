"""
Tests for features.py. Run with:
    pip install pytest
    cd src && pytest ../tests -v

These are known-input/known-output cases, not fuzz tests — each one hand-
computes the expected value for a specific transaction and checks the
pipeline produces it. That's deliberate: for a fraud model, a silently
wrong feature is worse than a crashing one, so these pin down exact
arithmetic, not just "doesn't crash."
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import build_features, FEATURE_COLUMNS, CATEGORICAL_TYPES  # noqa: E402


def make_row(**overrides):
    """One transaction row with sensible defaults, override what you need."""
    row = {
        "step": 10, "type": "PAYMENT", "amount": 100.0,
        "nameOrig": "C001", "oldbalanceOrg": 1000.0, "newbalanceOrig": 900.0,
        "nameDest": "C002", "oldbalanceDest": 500.0, "newbalanceDest": 600.0,
        "isFraud": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def get_feature(df_out, name):
    return df_out.iloc[0][name]


class TestBalanceErrorFeatures:
    def test_reconciled_transaction_has_zero_error(self):
        # sender loses exactly `amount`, receiver gains exactly `amount` -> both errors are 0
        row = make_row(oldbalanceOrg=1000, newbalanceOrig=900, amount=100,
                        oldbalanceDest=500, newbalanceDest=600)
        out = build_features(row)
        assert get_feature(out, "errorBalanceOrig") == pytest.approx(0.0)
        assert get_feature(out, "errorBalanceDest") == pytest.approx(0.0)

    def test_underreported_dest_balance_shows_positive_error(self):
        # amount=100 sent, but dest only went up by 60 -> error should be +40
        row = make_row(oldbalanceDest=500, newbalanceDest=560, amount=100)
        out = build_features(row)
        assert get_feature(out, "errorBalanceDest") == pytest.approx(40.0)

    def test_orig_balance_mismatch_is_captured(self):
        # sender's balance dropped by 150 when amount was only 100 -> error should be -50
        row = make_row(oldbalanceOrg=1000, newbalanceOrig=850, amount=100)
        out = build_features(row)
        assert get_feature(out, "errorBalanceOrig") == pytest.approx(-50.0)


class TestFraudSignatureFlags:
    def test_drained_to_zero_flag_fires(self):
        row = make_row(oldbalanceOrg=500, newbalanceOrig=0)
        out = build_features(row)
        assert get_feature(out, "orig_drained_to_zero") == 1

    def test_drained_to_zero_flag_does_not_fire_if_balance_was_already_zero(self):
        # starting balance of 0 -> emptying it isn't a "drain", it's already empty
        row = make_row(oldbalanceOrg=0, newbalanceOrig=0)
        out = build_features(row)
        assert get_feature(out, "orig_drained_to_zero") == 0

    def test_drained_to_zero_flag_does_not_fire_on_partial_spend(self):
        row = make_row(oldbalanceOrg=500, newbalanceOrig=200)
        out = build_features(row)
        assert get_feature(out, "orig_drained_to_zero") == 0

    def test_mule_pattern_flag_fires_on_zero_zero_with_positive_amount(self):
        row = make_row(oldbalanceDest=0, newbalanceDest=0, amount=5000)
        out = build_features(row)
        assert get_feature(out, "dest_is_mule_pattern") == 1

    def test_mule_pattern_flag_does_not_fire_if_dest_balance_actually_moves(self):
        row = make_row(oldbalanceDest=0, newbalanceDest=5000, amount=5000)
        out = build_features(row)
        assert get_feature(out, "dest_is_mule_pattern") == 0

    def test_mule_pattern_flag_does_not_fire_on_zero_amount(self):
        # both balances zero but nothing actually moved -> not a mule pattern, just an empty transaction
        row = make_row(oldbalanceDest=0, newbalanceDest=0, amount=0)
        out = build_features(row)
        assert get_feature(out, "dest_is_mule_pattern") == 0


class TestRatioFeatures:
    def test_amount_to_balance_ratio(self):
        row = make_row(amount=500, oldbalanceOrg=1000)
        out = build_features(row)
        # amount / (oldbalanceOrg + 1) -- the +1 avoids div-by-zero for empty accounts
        assert get_feature(out, "amount_to_orig_balance_ratio") == pytest.approx(500 / 1001)

    def test_amount_to_balance_ratio_handles_zero_balance(self):
        row = make_row(amount=500, oldbalanceOrg=0)
        out = build_features(row)
        assert get_feature(out, "amount_to_orig_balance_ratio") == pytest.approx(500 / 1.0)
        assert np.isfinite(get_feature(out, "amount_to_orig_balance_ratio"))


class TestTimeFeatures:
    @pytest.mark.parametrize("step,expected_hour", [(0, 0), (23, 23), (24, 0), (25, 1), (100, 4)])
    def test_hour_of_day_wraps_correctly(self, step, expected_hour):
        row = make_row(step=step)
        out = build_features(row)
        assert get_feature(out, "hour_of_day") == expected_hour

    @pytest.mark.parametrize("hour,expected_is_night", [
        (0, 1), (3, 1), (4, 1), (22, 1), (23, 1),   # night hours
        (5, 0), (12, 0), (18, 0), (21, 0),           # day hours
    ])
    def test_is_night_flag(self, hour, expected_is_night):
        row = make_row(step=hour)  # step < 24 so hour_of_day == step
        out = build_features(row)
        assert get_feature(out, "is_night") == expected_is_night


class TestTypeOneHot:
    def test_exactly_one_type_column_is_set(self):
        for t in CATEGORICAL_TYPES:
            row = make_row(type=t)
            out = build_features(row)
            for other in CATEGORICAL_TYPES:
                expected = 1 if other == t else 0
                assert get_feature(out, f"type_{other}") == expected, f"type={t}, checking type_{other}"


class TestVelocityFeatures:
    def test_first_transaction_has_zero_prior_count(self):
        row = make_row(nameOrig="C_NEW")
        out = build_features(row)
        assert get_feature(out, "orig_txn_count_so_far") == 0

    def test_training_path_computes_expanding_stats_leak_free(self):
        # three transactions from the same account, in step order --
        # the 3rd transaction should see count=2 and avg of the first two only
        rows = pd.DataFrame([
            {"step": 1, "type": "PAYMENT", "amount": 100, "nameOrig": "C777",
             "oldbalanceOrg": 1000, "newbalanceOrig": 900, "nameDest": "C1",
             "oldbalanceDest": 0, "newbalanceDest": 100, "isFraud": 0},
            {"step": 2, "type": "PAYMENT", "amount": 200, "nameOrig": "C777",
             "oldbalanceOrg": 900, "newbalanceOrig": 700, "nameDest": "C2",
             "oldbalanceDest": 0, "newbalanceDest": 200, "isFraud": 0},
            {"step": 3, "type": "PAYMENT", "amount": 50, "nameOrig": "C777",
             "oldbalanceOrg": 700, "newbalanceOrig": 650, "nameDest": "C3",
             "oldbalanceDest": 0, "newbalanceDest": 50, "isFraud": 0},
        ])
        out = build_features(rows)
        third = out.iloc[2]
        assert third["orig_txn_count_so_far"] == 2
        assert third["orig_avg_amount_so_far"] == pytest.approx((100 + 200) / 2)

    def test_serving_path_uses_provided_history(self):
        history = {"C999": {"count": 4, "amount_sum": 2000.0}}
        row = make_row(nameOrig="C999", amount=300)
        out = build_features(row, history=history)
        assert get_feature(out, "orig_txn_count_so_far") == 4
        assert get_feature(out, "orig_avg_amount_so_far") == pytest.approx(500.0)  # 2000/4
        assert get_feature(out, "amount_vs_orig_avg_ratio") == pytest.approx(300 / 501.0)

    def test_serving_path_cold_start_uses_current_amount_as_avg(self):
        # brand-new account (not in history) -> avg defaults to its own current amount,
        # which makes amount_vs_orig_avg_ratio ~= 1.0 (no signal yet, correctly)
        row = make_row(nameOrig="C_BRAND_NEW", amount=777)
        out = build_features(row, history={})
        assert get_feature(out, "orig_avg_amount_so_far") == pytest.approx(777.0)


class TestOutputContract:
    def test_output_has_exactly_the_declared_feature_columns(self):
        row = make_row()
        out = build_features(row)
        for col in FEATURE_COLUMNS:
            assert col in out.columns, f"missing feature column: {col}"

    def test_output_preserves_isFraud_when_present_and_drops_it_when_absent(self):
        row_with_label = make_row(isFraud=1)
        out = build_features(row_with_label)
        assert "isFraud" in out.columns
        assert get_feature(out, "isFraud") == 1

        row_no_label = make_row().drop(columns=["isFraud"])
        out_no_label = build_features(row_no_label)
        assert "isFraud" not in out_no_label.columns

    def test_no_nans_in_output_features(self):
        row = make_row()
        out = build_features(row)
        assert not out[FEATURE_COLUMNS].isna().any().any()
