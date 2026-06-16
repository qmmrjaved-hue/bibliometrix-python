"""
Phase 3 — Validation.

validate(df) performs three checks and raises ValidationError on the
first failure, naming the offending column. It also returns a structured
report dict used by the Streamlit dashboard Validation Report section.
"""

import pandas as pd

from .mapping_dicts import LIST_FIELDS, MANDATORY_COLUMNS


class ValidationError(Exception):
    """Raised when the standardized DataFrame fails a schema check."""


def _check_mandatory_columns(df: pd.DataFrame) -> list[str]:
    """
    Return a list of mandatory columns that are missing from df.

    Args:
        df: Standardized DataFrame.

    Returns:
        List of missing column names (empty if all present).
    """
    return [col for col in MANDATORY_COLUMNS if col not in df.columns]


def _check_no_nulls(df: pd.DataFrame) -> list[str]:
    """
    Return a list of columns that contain NaN or None values.

    List-typed columns are checked for None entries within the lists as
    well as for the cell itself being null.

    Args:
        df: Standardized DataFrame (only mandatory columns are checked).

    Returns:
        List of column names with null values (empty if clean).
    """
    problem_cols: list[str] = []
    for col in MANDATORY_COLUMNS:
        if col not in df.columns:
            continue
        if col in LIST_FIELDS:
            # Each cell must be a list (not None/NaN)
            has_null_cell = df[col].apply(lambda v: not isinstance(v, list)).any()
            if has_null_cell:
                problem_cols.append(col)
        else:
            if df[col].isnull().any():
                problem_cols.append(col)
    return problem_cols


def _check_list_fields(df: pd.DataFrame) -> list[str]:
    """
    Return a list of LIST_FIELDS columns whose cells are not list[str].

    Args:
        df: Standardized DataFrame.

    Returns:
        List of column names with type violations (empty if all correct).
    """
    problem_cols: list[str] = []
    for col in LIST_FIELDS:
        if col not in df.columns:
            continue
        def _is_list_of_str(v: object) -> bool:
            return isinstance(v, list) and all(isinstance(x, str) for x in v)

        if not df[col].apply(_is_list_of_str).all():
            problem_cols.append(col)
    return problem_cols


def validate(df: pd.DataFrame) -> dict:
    """
    Validate a standardized DataFrame against the WoS schema.

    Three checks are performed in order:
        1. All mandatory columns are present.
        2. Zero NaN / None values remain in mandatory columns.
        3. All LIST_FIELDS columns contain list[str] values.

    The function raises ValidationError on the first check that fails,
    naming the offending column(s) in the error message.

    Args:
        df: Standardized DataFrame produced by run_pipeline().

    Returns:
        A structured report dict with the following keys:
            - passed (bool): True if all checks passed.
            - checks (list[dict]): One entry per check with:
                - name (str): Human-readable check name.
                - passed (bool): Whether this check passed.
                - problem_columns (list[str]): Empty if passed.

    Raises:
        ValidationError: If any check fails, with the failing column name(s).
    """
    report = {
        "passed": True,
        "checks": [],
    }

    # Check 1 — mandatory columns
    missing = _check_mandatory_columns(df)
    check1 = {
        "name": "All mandatory columns present",
        "passed": len(missing) == 0,
        "problem_columns": missing,
    }
    report["checks"].append(check1)
    if not check1["passed"]:
        report["passed"] = False
        raise ValidationError(
            f"Missing mandatory columns: {missing}. "
            "Ensure run_pipeline() completed without errors."
        )

    # Check 2 — no nulls
    null_cols = _check_no_nulls(df)
    check2 = {
        "name": "Zero NaN / None values in mandatory columns",
        "passed": len(null_cols) == 0,
        "problem_columns": null_cols,
    }
    report["checks"].append(check2)
    if not check2["passed"]:
        report["passed"] = False
        raise ValidationError(
            f"NaN / None values found in column(s): {null_cols}. "
            "Check handle_nulls() in standardizer.py."
        )

    # Check 3 — list[str] type contract
    bad_list_cols = _check_list_fields(df)
    check3 = {
        "name": "Multi-value columns are list[str]",
        "passed": len(bad_list_cols) == 0,
        "problem_columns": bad_list_cols,
    }
    report["checks"].append(check3)
    if not check3["passed"]:
        report["passed"] = False
        raise ValidationError(
            f"Columns not of type list[str]: {bad_list_cols}. "
            "Check enforce_types() in standardizer.py."
        )

    return report
