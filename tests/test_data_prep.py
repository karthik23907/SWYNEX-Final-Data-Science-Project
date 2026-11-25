import pandas as pd
import pytest

from src import data_prep as dp


def make_raw(**overrides) -> pd.DataFrame:
    """Two-row frame mimicking the raw schema (one normal customer, one brand-new)."""
    row = {
        "customerID": ["A-1", "B-2"],
        "gender": ["Female ", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [12, 0],
        "PhoneService": ["No", "Yes"],
        "MultipleLines": ["No phone service", "No"],
        "InternetService": ["DSL", "No"],
        "OnlineSecurity": ["Yes", "No internet service"],
        "OnlineBackup": ["No", "No internet service"],
        "DeviceProtection": ["No", "No internet service"],
        "TechSupport": ["No", "No internet service"],
        "StreamingTV": ["No", "No internet service"],
        "StreamingMovies": ["No", "No internet service"],
        "Contract": ["One year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Mailed check", "Mailed check"],
        "MonthlyCharges": [50.0, 20.0],
        "TotalCharges": ["600.0", " "],
        "Churn": ["No", "Yes"],
    }
    row.update(overrides)
    return pd.DataFrame(row)


def test_column_names_are_snake_case():
    df, _ = dp.clean(make_raw())
    assert "customer_id" in df.columns and "senior_citizen" in df.columns
    assert all(c == c.lower() and " " not in c for c in df.columns)


def test_blank_total_charges_become_zero_for_new_customers():
    df, _ = dp.clean(make_raw())
    assert df["total_charges"].dtype == "float64"
    assert df.loc[df["tenure"] == 0, "total_charges"].iloc[0] == 0.0


def test_blank_total_charges_with_tenure_raises():
    raw = make_raw(tenure=[12, 5])  # blank bill but customer has history -> needs review
    with pytest.raises(ValueError):
        dp.clean(raw)


def test_redundant_levels_collapsed_and_encoded():
    df, _ = dp.clean(make_raw())
    assert df["multiple_lines"].tolist() == [0, 0]
    assert df["online_security"].tolist() == [1, 0]
    assert str(df["online_security"].dtype) == "int8"


def test_unexpected_yes_no_value_raises():
    with pytest.raises(ValueError):
        dp.clean(make_raw(Partner=["Yes", "Maybe"]))


def test_whitespace_stripped_and_contract_ordered():
    df, _ = dp.clean(make_raw())
    assert df["gender"].tolist() == ["Female", "Male"]
    assert df["contract"].cat.ordered
    assert df["contract"].cat.categories.tolist() == dp.CONTRACT_ORDER


def test_input_not_mutated():
    raw = make_raw()
    snapshot = raw.copy()
    dp.clean(raw)
    pd.testing.assert_frame_equal(raw, snapshot)


def test_validation_catches_bad_data():
    df, _ = dp.clean(make_raw())
    df.loc[0, "monthly_charges"] = -5
    with pytest.raises(AssertionError):
        dp.validate(df)


@pytest.mark.skipif(not dp.RAW_PATH.exists(), reason="raw data not present")
def test_full_dataset_end_to_end():
    raw = dp.load_raw()
    df, audit = dp.clean(raw)
    dp.validate(df)
    assert df.shape == (7043, 21)
    assert df.isna().sum().sum() == 0
    assert len(audit.steps) >= 6
