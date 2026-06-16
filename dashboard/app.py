"""
Bibliometrix Python — Streamlit Dashboard

Five sections:
    1. API Query      — run the ETL pipeline via PubMed/OpenAlex API
    2. File Upload    — run the ETL pipeline from a local file
    3. Validation     — per-check pass/fail status
    4. Analysis       — summary metrics and Plotly charts
    5. About          — architecture description

Design rules:
    - No emojis anywhere
    - Custom CSS injected at startup (DM Sans, deep purple palette)
    - All charts use plotly.graph_objects with custom template
    - Sidebar for all controls

Submission: Qamar Javed — Data Science exam, Prof. Vincenzo Moscato,
            Federico II / UNINA, AY 2025/2026
"""

import io
import logging
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Package stubbing — MUST run before any www.services import.
#
# www/services/__init__.py wildcard-imports Shiny-specific modules that pull
# in prince, igraph, faicons, etc., which are not installed in this Python
# 3.12 Streamlit environment.  Registering lightweight stubs prevents Python
# from running __init__.py; ETL-specific submodules are loaded directly.
#
# Additionally, parsers.py does `from .utils import *`, and utils.py imports
# the same Shiny deps.  We stub www.services.utils with just `re` (the only
# name parsers.py actually uses from utils) so parsers.py loads cleanly.
# ---------------------------------------------------------------------------
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


def _stub_pkg(name: str, fs_path: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [fs_path]       # type: ignore[attr-defined]
        mod.__package__ = name
        sys.modules[name] = mod


_stub_pkg("www",          str(Path(_root) / "www"))
_stub_pkg("www.services", str(Path(_root) / "www" / "services"))

# Stub www.services.utils so parsers.py can do `from .utils import *`
# without importing the Shiny-only dependencies in utils.py.
if "www.services.utils" not in sys.modules:
    import re as _re
    _utils_mod = types.ModuleType("www.services.utils")
    _utils_mod.re = _re          # type: ignore[attr-defined]
    sys.modules["www.services.utils"] = _utils_mod

from www.services.api_retriever import fetch_openalex, fetch_pubmed    # noqa: E402
from www.services.mapping_dicts import LIST_FIELDS                     # noqa: E402
from www.services.standardizer import (                                 # noqa: E402
    export_to_csv,
    load_file,
    run_pipeline,
)
from www.services.validator import ValidationError, validate            # noqa: E402

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #fafaf8;
}
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

[data-testid="stSidebar"] {
    background: #2e1760 !important;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #c4b5fd !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ede9fe !important;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
}
[data-testid="stSidebar"] hr {
    border-color: #4c1d95 !important;
    margin: 1rem 0;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #a78bfa !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e0d4a !important;
    border: 1px solid #4c1d95 !important;
    border-radius: 6px !important;
    color: #ede9fe !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.2) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: #7c3aed !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.65rem 1.25rem !important;
    width: 100% !important;
    letter-spacing: 0.01em;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #6d28d9 !important;
}

.main-header {
    background: #2e1760;
    padding: 2rem 2.25rem;
    border-radius: 10px;
    margin-bottom: 1.75rem;
    box-shadow: 0 2px 12px rgba(46,23,96,0.2);
}
.main-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f5f3ff;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem;
}
.main-header p {
    font-size: 0.85rem;
    color: #a78bfa;
    margin: 0;
    font-weight: 400;
}
.main-header .badge-row {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.hbadge {
    background: rgba(167,139,250,0.15);
    color: #c4b5fd;
    border: 1px solid rgba(167,139,250,0.28);
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.stTabs [data-baseweb="tab-list"] {
    background: #f5f3ff;
    border: none;
    border-radius: 8px;
    padding: 0.3rem;
    gap: 0.2rem;
    box-shadow: none;
    margin-bottom: 1.25rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: #6b21a8;
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 0.48rem 1.1rem;
    transition: all 0.12s;
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: #ede9fe;
    color: #3b0764;
}
.stTabs [aria-selected="true"] {
    background: #7c3aed !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] {
    background: white;
    border-radius: 10px;
    padding: 1.75rem 2rem;
    border: 1px solid #ede9fe;
    box-shadow: 0 1px 6px rgba(46,23,96,0.06);
}

.section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #7c3aed;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    border-bottom: 2px solid #ede9fe;
    padding-bottom: 0.5rem;
    margin-bottom: 1.25rem;
}

.metric-card {
    background: #faf5ff;
    border: 1px solid #ede9fe;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    transition: box-shadow 0.15s;
    height: 100%;
}
.metric-card:hover { box-shadow: 0 4px 16px rgba(109,40,217,0.12); }
.metric-card .label {
    font-size: 0.67rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: #7c3aed;
    margin-bottom: 0.45rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #2e1760;
    line-height: 1.1;
}
.metric-card .value.small { font-size: 1.3rem; }

.check-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.8rem 1rem;
    background: #faf5ff;
    border: 1px solid #ede9fe;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    transition: box-shadow 0.12s;
}
.check-row:hover { box-shadow: 0 2px 8px rgba(109,40,217,0.08); }
.check-row.fail { background: #fff7f7; border-color: #fecaca; }
.check-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0.18rem 0.5rem;
    min-width: 42px;
    flex-shrink: 0;
}
.check-badge.pass { background: #dcfce7; color: #15803d; }
.check-badge.fail { background: #fee2e2; color: #b91c1c; }
.check-label { font-size: 0.875rem; color: #3b0764; font-weight: 500; }
.check-detail { font-size: 0.78rem; color: #7c3aed; margin-left: auto; }

.val-overall {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.85rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    font-size: 0.875rem;
}
.val-overall.pass { background: #f0fdf4; border-left: 4px solid #22c55e; }
.val-overall.fail { background: #fef2f2; border-left: 4px solid #ef4444; }
.val-overall .status-label { font-weight: 700; }
.val-overall.pass .status-label { color: #15803d; }
.val-overall.fail .status-label { color: #b91c1c; }
.val-overall .status-desc { font-size: 0.8rem; color: #6b7280; margin-left: 0.4rem; }

.stDownloadButton > button {
    background: white !important;
    color: #7c3aed !important;
    border: 1.5px solid #7c3aed !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s !important;
}
.stDownloadButton > button:hover {
    background: #7c3aed !important;
    color: white !important;
}

[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.875rem !important; }
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid #ede9fe !important;
}
[data-testid="stProgressBar"] > div {
    background: #7c3aed !important;
    border-radius: 4px !important;
}

.chart-card {
    background: #faf5ff;
    border: 1px solid #ede9fe;
    border-radius: 10px;
    padding: 1.25rem 1.5rem 0.5rem;
    margin-bottom: 1.25rem;
}
.chart-title { font-size: 0.85rem; font-weight: 600; color: #2e1760; margin-bottom: 0.1rem; }
.chart-sub { font-size: 0.72rem; color: #7c3aed; margin-bottom: 0.65rem; }

.about-card {
    background: white;
    border: 1px solid #ede9fe;
    border-radius: 10px;
    padding: 1.75rem 2.25rem;
    line-height: 1.75;
    color: #374151;
    font-size: 0.875rem;
}
.about-card .about-lead {
    font-size: 0.95rem;
    color: #2e1760;
    margin-bottom: 1.5rem;
    line-height: 1.7;
    padding-bottom: 1.25rem;
    border-bottom: 2px solid #ede9fe;
}
.about-card h3 {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7c3aed;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}
.about-card ul { margin: 0; padding-left: 0; list-style: none; }
.about-card ul li::before {
    content: "-";
    color: #7c3aed;
    font-weight: 700;
    display: inline-block;
    width: 1.1rem;
}
.about-card li { margin-bottom: 0.35rem; }
.about-card code {
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    border-radius: 4px;
    padding: 0.1em 0.42em;
    font-size: 0.82em;
    color: #6d28d9;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}
.about-phase {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
    margin-bottom: 0.65rem;
}
.phase-num {
    background: #7c3aed;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    min-width: 1.5rem;
    height: 1.5rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.phase-text { font-size: 0.875rem; color: #374151; }
.about-footer {
    margin-top: 1.75rem;
    padding-top: 1rem;
    border-top: 1px solid #ede9fe;
    font-size: 0.78rem;
    color: #7c3aed;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.col-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.3rem;
    margin-top: 0.5rem;
}
.col-tag {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.7rem;
    background: #f5f3ff;
    color: #6d28d9;
    padding: 0.2rem 0.35rem;
    border-radius: 4px;
    text-align: center;
    font-weight: 500;
    border: 1px solid #ddd6fe;
}

.upload-hint {
    background: #f5f3ff;
    border: 1px solid #ddd6fe;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    font-size: 0.8rem;
    color: #6d28d9;
    line-height: 1.6;
}
</style>
"""

# ---------------------------------------------------------------------------
# Plotly custom template
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="DM Sans, sans-serif", size=12, color="#374151"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=["#7c3aed", "#a855f7", "#c084fc", "#6d28d9", "#4c1d95"],
        xaxis=dict(
            gridcolor="#f5f3ff",
            linecolor="#ede9fe",
            tickfont=dict(size=11),
            title_font=dict(size=11, color="#7c3aed"),
        ),
        yaxis=dict(
            gridcolor="#f5f3ff",
            linecolor="#ede9fe",
            tickfont=dict(size=11),
            title_font=dict(size=11, color="#7c3aed"),
        ),
        margin=dict(l=48, r=24, t=36, b=44),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#ede9fe",
            font=dict(family="DM Sans, sans-serif", size=12, color="#2e1760"),
        ),
    )
)

# File format label → Source literal for load_file()
_FILE_FORMAT_MAP: dict[str, str | None] = {
    "Auto-detect":        None,
    "Scopus CSV":         "SCOPUS_CSV",
    "Scopus BibTeX":      "SCOPUS_BIB",
    "WoS TXT / CIW":     "WOS_TXT",
    "WoS BibTeX":         "WOS_BIB",
    "Dimensions CSV/XLSX":"DIMENSIONS",
    "Lens.org CSV":       "LENS",
    "Cochrane TXT":       "COCHRANE",
    "PubMed TXT":         "PUBMED_FILE",
}


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict = {
        # API Query
        "df":                 None,
        "validation_report":  None,
        "pipeline_error":     None,
        "csv_bytes":          None,
        "source_used":        None,
        # File Upload
        "file_df":            None,
        "file_report":        None,
        "file_error":         None,
        "file_csv_bytes":     None,
        "file_source":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run_api_pipeline(query: str, platform: str, max_results: int) -> None:
    """Execute Extract → Transform → Validate → Load for an API source."""
    st.session_state.df = None
    st.session_state.validation_report = None
    st.session_state.pipeline_error = None
    st.session_state.csv_bytes = None
    st.session_state.source_used = platform

    progress = st.progress(0, text="Starting pipeline...")
    try:
        progress.progress(10, text="Phase 1: Extracting from " + platform + "...")
        if platform == "PubMed":
            raw_df = fetch_pubmed(query, max_results)
        else:
            raw_df = fetch_openalex(query, max_results)

        progress.progress(40, text="Phase 2: Transforming...")
        std_df = run_pipeline(raw_df)

        progress.progress(70, text="Phase 3: Validating...")
        report = validate(std_df)
        st.session_state.validation_report = report

        progress.progress(90, text="Phase 4: Exporting CSV...")
        export_to_csv(std_df, platform.upper())

        export_df = std_df.copy()
        for col in LIST_FIELDS:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(
                    lambda v: ";".join(v) if isinstance(v, list) else str(v)
                )
        buf = io.BytesIO()
        export_df.to_csv(buf, index=False, encoding="utf-8")
        st.session_state.csv_bytes = buf.getvalue()
        st.session_state.df = std_df
        progress.progress(100, text="Pipeline complete.")

    except ValidationError as exc:
        st.session_state.pipeline_error = f"Validation failed: {exc}"
        progress.empty()
    except Exception as exc:
        st.session_state.pipeline_error = f"Pipeline error: {exc}"
        progress.empty()


def _run_file_pipeline(uploaded, format_hint: "str | None") -> None:
    """Execute Extract (file) → Transform → Validate → Load for a file source."""
    st.session_state.file_df = None
    st.session_state.file_report = None
    st.session_state.file_error = None
    st.session_state.file_csv_bytes = None
    st.session_state.file_source = None

    suffix = Path(uploaded.name).suffix or ".tmp"
    progress = st.progress(0, text="Reading file...")
    tmp_path: "Path | None" = None

    try:
        # Save uploaded bytes to a temp file (parsers require file paths)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        progress.progress(15, text="Phase 1: Loading " + uploaded.name + "...")
        raw_df, source = load_file(tmp_path, source=format_hint)  # type: ignore[arg-type]

        progress.progress(40, text="Phase 2: Transforming (" + source + ")...")
        std_df = run_pipeline(raw_df, source=source)

        progress.progress(70, text="Phase 3: Validating...")
        report = validate(std_df)
        st.session_state.file_report = report

        progress.progress(90, text="Phase 4: Exporting CSV...")
        export_to_csv(std_df, source)

        export_df = std_df.copy()
        for col in LIST_FIELDS:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(
                    lambda v: ";".join(v) if isinstance(v, list) else str(v)
                )
        buf = io.BytesIO()
        export_df.to_csv(buf, index=False, encoding="utf-8")
        st.session_state.file_csv_bytes = buf.getvalue()
        st.session_state.file_df = std_df
        st.session_state.file_source = source
        progress.progress(100, text="Pipeline complete.")

    except ValidationError as exc:
        st.session_state.file_error = f"Validation failed: {exc}"
        progress.empty()
    except Exception as exc:
        st.session_state.file_error = f"Pipeline error: {exc}"
        progress.empty()
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Shared preview helper
# ---------------------------------------------------------------------------

def _preview_df(df: pd.DataFrame, source_label: str, csv_bytes: bytes) -> None:
    """Render a success banner, download button, and 20-row preview table."""
    col_l, col_r = st.columns([6, 2])
    with col_l:
        st.success(f"Retrieved {len(df)} records from {source_label}.")
    with col_r:
        if csv_bytes:
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=f"{source_label.lower()}_standardized.csv",
                mime="text/csv",
                use_container_width=True,
            )

    preview_cols = [c for c in ["TI", "AU", "PY", "SO", "TC", "DI", "SR"] if c in df.columns]
    preview_df = df[preview_cols].head(20).copy()
    for col in preview_cols:
        if col in LIST_FIELDS:
            preview_df[col] = preview_df[col].apply(
                lambda v: "; ".join(v[:3]) + (" ..." if len(v) > 3 else "")
                if isinstance(v, list) else str(v)
            )
    st.dataframe(preview_df, use_container_width=True, height=420)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_api_query() -> None:
    st.markdown('<div class="section-title">API Query</div>', unsafe_allow_html=True)

    if st.session_state.pipeline_error:
        st.error(st.session_state.pipeline_error)

    df = st.session_state.df
    if df is not None:
        _preview_df(df, st.session_state.source_used or "API", st.session_state.csv_bytes or b"")
    else:
        st.info("Configure your query in the sidebar and click Run Pipeline.")


def _render_file_upload() -> None:
    st.markdown('<div class="section-title">File Upload</div>', unsafe_allow_html=True)

    if st.session_state.file_error:
        st.error(st.session_state.file_error)

    df = st.session_state.file_df
    if df is not None:
        src = st.session_state.file_source or "file"
        _preview_df(df, src, st.session_state.file_csv_bytes or b"")
    else:
        st.markdown(
            '<div class="upload-hint">'
            'Upload a bibliographic export file using the sidebar controls, '
            'select its format (or leave on Auto-detect), then click '
            '<strong>Process File</strong>. Supported formats: '
            'Scopus CSV, Scopus BibTeX, WoS TXT/CIW, WoS BibTeX, '
            'Dimensions CSV/XLSX, Lens.org CSV, Cochrane CDSR TXT, PubMed MEDLINE TXT.'
            '</div>',
            unsafe_allow_html=True,
        )


def _render_validation_report() -> None:
    st.markdown('<div class="section-title">Validation Report</div>', unsafe_allow_html=True)

    # Prefer file upload report when present; fall back to API report
    report = st.session_state.file_report if st.session_state.file_report is not None else st.session_state.validation_report
    if report is None:
        st.info("Run a query or process a file first to see validation results.")
        return

    overall = report.get("passed", False)
    oc = "pass" if overall else "fail"
    ol = "All checks passed" if overall else "One or more checks failed"
    st.markdown(
        f'<div class="val-overall {oc}">'
        f'<span class="status-label">{"PASS" if overall else "FAIL"}</span>'
        f'<span class="status-desc">{ol}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for check in report.get("checks", []):
        passed = check.get("passed", False)
        bc = "pass" if passed else "fail"
        badge = "PASS" if passed else "FAIL"
        problems = check.get("problem_columns", [])
        detail = (
            f'<span class="check-detail">Problem columns: {", ".join(problems)}</span>'
            if problems else ""
        )
        st.markdown(
            f'<div class="check-row {bc}">'
            f'<span class="check-badge {bc}">{badge}</span>'
            f'<span class="check-label">{check.get("name", "")}</span>'
            f'{detail}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_analysis() -> None:
    st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)

    # Prefer file upload data; fall back to API data
    df = st.session_state.file_df if st.session_state.file_df is not None else st.session_state.df
    if df is None:
        st.info("Run a query or process a file first to see analysis.")
        return

    # Summary metrics
    total_records = len(df)
    year_col = df["PY"].replace("", pd.NA).dropna()
    year_range = f"{year_col.min()} - {year_col.max()}" if len(year_col) > 0 else "N/A"
    unique_journals = df["SO"].replace("", pd.NA).dropna().nunique()

    au_col = df["AU"].apply(lambda v: v if isinstance(v, list) else [])
    all_authors = [a for sub in au_col for a in sub if a]
    unique_authors = len(set(all_authors))

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Total Records", str(total_records), False),
        ("Year Range", year_range, True),
        ("Unique Journals", str(unique_journals), False),
        ("Unique Authors", str(unique_authors), False),
    ]
    for col_widget, (label, value, small) in zip([col1, col2, col3, col4], cards):
        with col_widget:
            cls = "small" if small else ""
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value {cls}">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Publications per year
    py_counts = df["PY"].replace("", pd.NA).dropna().value_counts().sort_index()
    if len(py_counts) > 0:
        fig_year = go.Figure(
            go.Bar(
                x=py_counts.index.tolist(),
                y=py_counts.values.tolist(),
                marker=dict(
                    color=py_counts.values.tolist(),
                    colorscale=[[0, "#ddd6fe"], [1, "#4c1d95"]],
                    showscale=False,
                ),
                hovertemplate="<b>%{x}</b><br>%{y} records<extra></extra>",
            )
        )
        fig_year.update_layout(
            template=PLOTLY_TEMPLATE,
            xaxis_title="Publication Year",
            yaxis_title="Records",
            bargap=0.3,
        )
        st.markdown(
            '<div class="chart-card">'
            '<div class="chart-title">Publications per Year</div>'
            '<div class="chart-sub">Annual output of retrieved records</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_year, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    from collections import Counter  # noqa: PLC0415
    author_counts = Counter(all_authors)
    top_authors = author_counts.most_common(10)

    de_col = df["DE"].apply(lambda v: v if isinstance(v, list) else [])
    all_kw = [k.upper() for sub in de_col for k in sub if k]
    kw_counts = Counter(all_kw).most_common(15)

    c_au, c_kw = st.columns(2)

    with c_au:
        if top_authors:
            authors_list, count_list = zip(*top_authors)
            fig_au = go.Figure(
                go.Bar(
                    x=list(count_list)[::-1],
                    y=list(authors_list)[::-1],
                    orientation="h",
                    marker=dict(color="#2e1760"),
                    hovertemplate="<b>%{y}</b><br>%{x} records<extra></extra>",
                )
            )
            fig_au.update_layout(
                template=PLOTLY_TEMPLATE,
                xaxis_title="Records",
                margin=dict(l=160, r=16, t=28, b=36),
                height=340,
                bargap=0.25,
            )
            st.markdown(
                '<div class="chart-card">'
                '<div class="chart-title">Top 10 Authors</div>'
                '<div class="chart-sub">By record count</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_au, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

    with c_kw:
        if kw_counts:
            kws, kw_freqs = zip(*kw_counts)
            fig_kw = go.Figure(
                go.Bar(
                    x=list(kw_freqs)[::-1],
                    y=list(kws)[::-1],
                    orientation="h",
                    marker=dict(color="#7c3aed"),
                    hovertemplate="<b>%{y}</b><br>%{x} occurrences<extra></extra>",
                )
            )
            fig_kw.update_layout(
                template=PLOTLY_TEMPLATE,
                xaxis_title="Frequency",
                margin=dict(l=200, r=16, t=28, b=36),
                height=340,
                bargap=0.25,
            )
            st.markdown(
                '<div class="chart-card">'
                '<div class="chart-title">Top 15 Author Keywords</div>'
                '<div class="chart-sub">DE field — author-supplied terms</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_kw, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No author keywords (DE) found in this result set.")


def _render_about() -> None:
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="about-card">
  <div class="about-lead">
    A source-agnostic bibliometric ETL pipeline that replicates
    <code>convert2df()</code> from the R Bibliometrix package.
    All output conforms to the WoS Field Tag schema so every analytical
    function in <code>www/services/</code> runs without modification.
    Submitted by <strong>Qamar Javed</strong>.
  </div>

  <h3>Sources — API</h3>
  <ul>
    <li><strong>PubMed</strong> — NCBI E-utilities (esearch + efetch, MEDLINE format) with exponential-backoff retry</li>
    <li><strong>OpenAlex</strong> — REST API with cursor pagination and rate-limit handling</li>
  </ul>

  <h3>Sources — File Upload</h3>
  <ul>
    <li><strong>Scopus CSV</strong> — standard CSV export from the Scopus interface</li>
    <li><strong>Scopus BibTeX</strong> — BibTeX export; EID extracted from URL, TC from note field</li>
    <li><strong>Web of Science TXT / CIW</strong> — MEDLINE-style plaintext; parsed by <code>parse_wos_data()</code></li>
    <li><strong>Web of Science BibTeX</strong> — BibTeX export; WoS accession number used as UT</li>
    <li><strong>Dimensions CSV / XLSX</strong> — export with header row offset (skiprows=1)</li>
    <li><strong>Lens.org CSV</strong> — standard CSV export from the Lens interface</li>
    <li><strong>Cochrane CDSR TXT</strong> — KEY: value format; parsed by <code>parse_cochrane_data()</code></li>
    <li><strong>PubMed TXT</strong> — MEDLINE plaintext file export; LID used for DOI</li>
  </ul>

  <h3>Pipeline Phases</h3>
  <div class="about-phase">
    <span class="phase-num">1</span>
    <span class="phase-text"><strong>Extract</strong> — API retrieval or file loading via <code>load_file()</code>; auto-detects format from extension and content fingerprint</span>
  </div>
  <div class="about-phase">
    <span class="phase-num">2</span>
    <span class="phase-text"><strong>Transform</strong> — rename via <code>mapping_dicts.py</code>, enforce type contracts (AU/AF as list[str], TC as int, PY as 4-digit string), eliminate nulls, compute SR</span>
  </div>
  <div class="about-phase">
    <span class="phase-num">3</span>
    <span class="phase-text"><strong>Validate</strong> — mandatory columns, zero-null guarantee, list[str] type contract; raises <code>ValidationError</code> naming the failing column</span>
  </div>
  <div class="about-phase">
    <span class="phase-num">4</span>
    <span class="phase-text"><strong>Load</strong> — CSV export to <code>data/outputs/</code> with <code>;</code>-delimited multi-value fields for compatibility with all analytical functions</span>
  </div>

  <h3>Mandatory Output Columns (24)</h3>
  <div class="col-grid">
    <span class="col-tag">DB</span><span class="col-tag">UT</span><span class="col-tag">DI</span>
    <span class="col-tag">PMID</span><span class="col-tag">TI</span><span class="col-tag">SO</span>
    <span class="col-tag">JI</span><span class="col-tag">PY</span><span class="col-tag">DT</span>
    <span class="col-tag">LA</span><span class="col-tag">TC</span><span class="col-tag">AU</span>
    <span class="col-tag">AF</span><span class="col-tag">C1</span><span class="col-tag">RP</span>
    <span class="col-tag">CR</span><span class="col-tag">DE</span><span class="col-tag">ID</span>
    <span class="col-tag">AB</span><span class="col-tag">VL</span><span class="col-tag">IS</span>
    <span class="col-tag">BP</span><span class="col-tag">EP</span><span class="col-tag">SR</span>
  </div>

  <h3>Modules</h3>
  <ul>
    <li><code>mapping_dicts.py</code> — 10 source mapping dicts; schema constants (MANDATORY_COLUMNS, LIST_FIELDS, SCALAR_FIELDS)</li>
    <li><code>api_retriever.py</code> — <code>fetch_pubmed()</code>, <code>fetch_openalex()</code></li>
    <li><code>standardizer.py</code> — <code>load_file()</code>, <code>detect_source()</code>, <code>rename_columns()</code>, <code>enforce_types()</code>, <code>handle_nulls()</code>, <code>add_calculated_fields()</code>, <code>run_pipeline()</code></li>
    <li><code>validator.py</code> — <code>validate()</code>, <code>ValidationError</code></li>
    <li><code>dashboard/app.py</code> — this Streamlit interface (five sections)</li>
    <li><code>tests/test_etl.py</code> — pytest suite covering all pipeline phases and all file sources</li>
  </ul>

  <h3>Patches Applied to Existing Code</h3>
  <ul>
    <li><code>histnetwork.py</code> — added DIMENSIONS, LENS, COCHRANE, SCOPUS, WOS routing to WoS/Scopus citation analysis paths</li>
    <li><code>biblionetwork.py</code> — fixed case-insensitive SCOPUS comparison; added new sources to <code>label_short()</code></li>
    <li><code>metatagextraction.py</code> — added SCOPUS, DIMENSIONS, LENS, COCHRANE to AU_UN C3 override check</li>
  </ul>

  <div class="about-footer">
    <span>Qamar Javed — Data Science exam, Prof. Vincenzo Moscato, Federico II / UNINA, AY 2025/2026</span>
    <span>bibliometrix-python ETL extension</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Bibliometrix Python",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _init_state()

    # Header
    st.markdown(
        '<div class="main-header">'
        '<h1>Bibliometrix Python</h1>'
        '<p>Source-agnostic bibliometric ETL pipeline — WoS Field Tag schema output</p>'
        '<div class="badge-row">'
        '<span class="hbadge">PubMed</span>'
        '<span class="hbadge">OpenAlex</span>'
        '<span class="hbadge">Scopus</span>'
        '<span class="hbadge">WoS</span>'
        '<span class="hbadge">Dimensions</span>'
        '<span class="hbadge">Lens</span>'
        '<span class="hbadge">Cochrane</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### API Query")
        query = st.text_input(
            "Search query",
            value="machine learning bibliometrics",
            help="Supports full PubMed Entrez and OpenAlex search syntax.",
        )
        platform = st.selectbox("Platform", ["PubMed", "OpenAlex"])
        max_results = st.selectbox("Max results", [10, 50, 100, 200], index=2)
        run_clicked = st.button("Run Pipeline", use_container_width=True)

        st.markdown("---")
        st.markdown("### File Upload")
        uploaded_file = st.file_uploader(
            "Upload file",
            type=["csv", "bib", "txt", "ciw", "xlsx", "xls"],
            help=(
                "Supported: Scopus CSV/BibTeX, WoS TXT/CIW/BibTeX, "
                "Dimensions CSV/XLSX, Lens CSV, Cochrane TXT, PubMed TXT"
            ),
        )
        file_format = st.selectbox("Format", list(_FILE_FORMAT_MAP.keys()), index=0)
        process_clicked = st.button("Process File", use_container_width=True)

    # Trigger handlers
    if run_clicked:
        if not query.strip():
            st.sidebar.error("Please enter a search query.")
        else:
            _run_api_pipeline(query.strip(), platform, max_results)

    if process_clicked:
        if uploaded_file is None:
            st.sidebar.error("Please upload a file first.")
        else:
            _run_file_pipeline(uploaded_file, _FILE_FORMAT_MAP[file_format])

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["API Query", "File Upload", "Validation", "Analysis", "About"]
    )
    with tab1:
        _render_api_query()
    with tab2:
        _render_file_upload()
    with tab3:
        _render_validation_report()
    with tab4:
        _render_analysis()
    with tab5:
        _render_about()


if __name__ == "__main__":
    main()
