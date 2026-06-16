"""
ETL pipeline test suite.

Tests are organized by pipeline phase. Integration tests (test_fetch_*)
make real API calls and are marked with pytest.mark.integration so they
can be excluded with: pytest -m "not integration"

File-source tests require the sample files in sources/ to be present.
They are marked with pytest.mark.file_sources and can be excluded with:
    pytest -m "not file_sources"

Run all fast unit tests (no network, no file I/O) with:
    pytest -m "not integration and not file_sources"
"""

import sys
import types
from pathlib import Path

import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Stub www.services.utils BEFORE any www.services import so that parsers.py
# can do `from .utils import *` without pulling in Shiny-only deps.
# ---------------------------------------------------------------------------
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _stub_pkg(name: str, fs_path: str) -> None:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = [fs_path]      # type: ignore[attr-defined]
        mod.__package__ = name
        sys.modules[name] = mod


_stub_pkg("www",          str(Path(_ROOT) / "www"))
_stub_pkg("www.services", str(Path(_ROOT) / "www" / "services"))

if "www.services.utils" not in sys.modules:
    import re as _re
    _utils_mod = types.ModuleType("www.services.utils")
    _utils_mod.re = _re          # type: ignore[attr-defined]
    sys.modules["www.services.utils"] = _utils_mod

from www.services.mapping_dicts import LIST_FIELDS, MANDATORY_COLUMNS, SCALAR_FIELDS  # noqa: E402
from www.services.standardizer import (                                                # noqa: E402
    add_calculated_fields,
    detect_source,
    enforce_types,
    export_to_csv,
    handle_nulls,
    load_file,
    rename_columns,
    run_pipeline,
)
from www.services.validator import ValidationError, validate                           # noqa: E402

# ---------------------------------------------------------------------------
# Paths to sample files in sources/
# ---------------------------------------------------------------------------
_SOURCES = Path(_ROOT) / "sources"

_SCOPUS_CSV   = _SOURCES / "Scopus"      / "Scopus.csv"
_SCOPUS_BIB   = _SOURCES / "Scopus"      / "Scopus.bib"
_WOS_TXT      = _SOURCES / "Web_of_Science" / "WoS.txt"
_WOS_CIW      = _SOURCES / "Web_of_Science" / "WoS.ciw"
_WOS_BIB      = _SOURCES / "Web_of_Science" / "WoS.bib"
_DIMENSIONS   = _SOURCES / "Dimensions"  / "Dimensions.csv"
_DIM_XLSX     = _SOURCES / "Dimensions"  / "Dimensions.xlsx"
_LENS         = _SOURCES / "Lens"        / "Lens.csv"
_COCHRANE     = _SOURCES / "Cochrane"    / "citation-export.txt"
_PUBMED_FILE  = _SOURCES / "PubMed"     / "pubmed-allergicrh-set.txt"


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic DataFrames that mimic raw API output
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_pubmed_df() -> pd.DataFrame:
    """Minimal PubMed-like raw DataFrame (2 records)."""
    return pd.DataFrame([
        {
            "PMID": "12345678",
            "TI": "Artificial intelligence in medicine",
            "AU": "Smith J;Jones A",
            "FAU": "Smith, John;Jones, Alice",
            "AD": "University of Naples;MIT",
            "DP": "2022 Mar 15",
            "TA": "J Med Inform",
            "JT": "Journal of Medical Informatics",
            "PT": "Journal Article",
            "MH": "Artificial Intelligence;Medicine",
            "OT": "AI;healthcare",
            "AB": "This paper reviews AI in medicine.",
            "VI": "10",
            "IP": "3",
            "PG": "100-110",
            "AID": "10.1234/abc [doi]",
            "LA": "eng",
            "DB": "PUBMED",
        },
        {
            "PMID": "87654321",
            "TI": "Deep learning for drug discovery",
            "AU": "Brown K",
            "FAU": "Brown, Kate",
            "AD": "Stanford University",
            "DP": "2021 Nov",
            "TA": "Nat Biotechnol",
            "JT": "Nature Biotechnology",
            "PT": "Review",
            "MH": "Deep Learning;Drug Discovery",
            "OT": "deep learning",
            "AB": "Deep learning accelerates drug discovery.",
            "VI": "39",
            "IP": "11",
            "PG": "1400-1408",
            "AID": "10.5678/xyz [doi]",
            "LA": "eng",
            "DB": "PUBMED",
        },
    ])


@pytest.fixture()
def raw_openalex_df() -> pd.DataFrame:
    """Minimal OpenAlex-like raw DataFrame (2 records)."""
    return pd.DataFrame([
        {
            "display_name": "Bibliometric analysis of machine learning",
            "author_names": ["Rossi, Mario", "Bianchi, Luigi"],
            "author_full_names": ["Mario Rossi", "Luigi Bianchi"],
            "affiliations": ["University Federico II"],
            "doi": "10.9999/ml001",
            "publication_year": "2023",
            "source_title": "Scientometrics",
            "source_abbr": "Scientometrics",
            "volume": "128",
            "issue": "2",
            "first_page": "500",
            "last_page": "520",
            "abstract": "A bibliometric analysis of machine learning literature.",
            "concepts": ["Machine Learning", "Bibliometrics"],
            "keywords": ["bibliometrics", "machine learning"],
            "cited_by_count": 12,
            "type": "article",
            "language": "en",
            "referenced_works": [],
            "openalex_id": "https://openalex.org/W111",
            "pmid": "",
            "reprint_author": "Rossi, Mario",
            "DB": "OPENALEX",
        },
        {
            "display_name": "Scientometric study of AI research",
            "author_names": ["Chen, Wei"],
            "author_full_names": ["Wei Chen"],
            "affiliations": ["Tsinghua University"],
            "doi": "10.9999/ai002",
            "publication_year": "2022",
            "source_title": "Journal of Informetrics",
            "source_abbr": "J Informetr",
            "volume": "16",
            "issue": "4",
            "first_page": "101230",
            "last_page": "",
            "abstract": "Scientometric analysis of AI research trends.",
            "concepts": ["Artificial Intelligence", "Scientometrics"],
            "keywords": ["AI", "scientometrics"],
            "cited_by_count": 5,
            "type": "article",
            "language": "en",
            "referenced_works": [],
            "openalex_id": "https://openalex.org/W222",
            "pmid": "",
            "reprint_author": "Chen, Wei",
            "DB": "OPENALEX",
        },
    ])


@pytest.fixture()
def std_pubmed_df(raw_pubmed_df) -> pd.DataFrame:
    return run_pipeline(raw_pubmed_df, source="PUBMED")


@pytest.fixture()
def std_openalex_df(raw_openalex_df) -> pd.DataFrame:
    return run_pipeline(raw_openalex_df, source="OPENALEX")


# ---------------------------------------------------------------------------
# Phase 1 — Integration tests (real API calls)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_fetch_pubmed_returns_nonempty_dataframe():
    from www.services.api_retriever import fetch_pubmed
    df = fetch_pubmed("machine learning", max_results=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "PubMed fetch returned empty DataFrame"
    assert "TI" in df.columns or "display_name" in df.columns or "PMID" in df.columns


@pytest.mark.integration
def test_fetch_openalex_returns_nonempty_dataframe():
    from www.services.api_retriever import fetch_openalex
    df = fetch_openalex("bibliometrics", max_results=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "OpenAlex fetch returned empty DataFrame"
    assert "display_name" in df.columns


# ---------------------------------------------------------------------------
# Phase 2a — detect_source
# ---------------------------------------------------------------------------

def test_detect_source_pubmed(raw_pubmed_df):
    assert detect_source(raw_pubmed_df) == "PUBMED"


def test_detect_source_openalex(raw_openalex_df):
    assert detect_source(raw_openalex_df) == "OPENALEX"


# ---------------------------------------------------------------------------
# Phase 2a — rename_columns
# ---------------------------------------------------------------------------

def test_rename_columns_pubmed_maps_to_wos(raw_pubmed_df):
    renamed = rename_columns(raw_pubmed_df, "PUBMED")
    assert "TI" in renamed.columns
    assert "AF" in renamed.columns   # FAU → AF
    assert "ID" in renamed.columns   # MH → ID
    assert "DE" in renamed.columns   # OT → DE


def test_rename_columns_openalex_maps_to_wos(raw_openalex_df):
    renamed = rename_columns(raw_openalex_df, "OPENALEX")
    assert "TI" in renamed.columns          # display_name → TI
    assert "TC" in renamed.columns          # cited_by_count → TC
    assert "AU" in renamed.columns          # author_names → AU


# ---------------------------------------------------------------------------
# Phase 2b — enforce_types
# ---------------------------------------------------------------------------

def test_enforce_types_list_fields_are_lists(raw_pubmed_df):
    renamed = rename_columns(raw_pubmed_df, "PUBMED")
    typed = enforce_types(renamed)
    for col in LIST_FIELDS:
        if col in typed.columns:
            assert typed[col].apply(lambda v: isinstance(v, list)).all(), \
                f"Column {col} is not list[str] after enforce_types"


def test_enforce_types_py_is_4digit_year(raw_pubmed_df):
    renamed = rename_columns(raw_pubmed_df, "PUBMED")
    typed = enforce_types(renamed)
    assert "PY" in typed.columns
    for val in typed["PY"]:
        assert val == "" or (len(val) == 4 and val.isdigit()), \
            f"PY value {val!r} is not a 4-digit year"


def test_enforce_types_tc_is_int(raw_openalex_df):
    renamed = rename_columns(raw_openalex_df, "OPENALEX")
    typed = enforce_types(renamed)
    assert typed["TC"].dtype == int or typed["TC"].apply(lambda v: isinstance(v, int)).all()


def test_enforce_types_doi_cleaned(raw_pubmed_df):
    renamed = rename_columns(raw_pubmed_df, "PUBMED")
    typed = enforce_types(renamed)
    for val in typed["DI"]:
        assert "[doi]" not in val, f"DOI not cleaned: {val!r}"
        assert "[pmc]" not in val


def test_enforce_types_ep_derived_from_page_range(raw_pubmed_df):
    renamed = rename_columns(raw_pubmed_df, "PUBMED")
    typed = enforce_types(renamed)
    assert typed["EP"].iloc[0] == "110"   # "100-110" → EP = "110"
    assert typed["BP"].iloc[0] == "100"


# ---------------------------------------------------------------------------
# Phase 2c — handle_nulls
# ---------------------------------------------------------------------------

def test_handle_nulls_no_nan_in_output(std_pubmed_df):
    scalar_cols = [c for c in MANDATORY_COLUMNS if c in std_pubmed_df.columns and c not in LIST_FIELDS]
    for col in scalar_cols:
        assert std_pubmed_df[col].isnull().sum() == 0, f"NaN found in {col}"


def test_handle_nulls_list_fields_are_lists(std_pubmed_df):
    for col in LIST_FIELDS:
        if col in std_pubmed_df.columns:
            assert std_pubmed_df[col].apply(lambda v: isinstance(v, list)).all()


# ---------------------------------------------------------------------------
# Phase 2 — full pipeline output checks (API sources)
# ---------------------------------------------------------------------------

def test_pipeline_mandatory_columns_present_pubmed(std_pubmed_df):
    for col in MANDATORY_COLUMNS:
        assert col in std_pubmed_df.columns, f"Mandatory column missing: {col}"


def test_pipeline_mandatory_columns_present_openalex(std_openalex_df):
    for col in MANDATORY_COLUMNS:
        assert col in std_openalex_df.columns, f"Mandatory column missing: {col}"


def test_pipeline_no_nan(std_pubmed_df):
    scalar_cols = [c for c in std_pubmed_df.columns if c not in LIST_FIELDS]
    assert std_pubmed_df[scalar_cols].isnull().sum().sum() == 0


def test_pipeline_tc_is_int(std_pubmed_df):
    assert std_pubmed_df["TC"].apply(lambda v: isinstance(v, int)).all()


def test_pipeline_py_is_4digit_string(std_pubmed_df):
    non_empty = std_pubmed_df["PY"][std_pubmed_df["PY"] != ""]
    assert non_empty.apply(lambda v: len(v) == 4 and v.isdigit()).all()


def test_pipeline_sr_is_nonempty_string(std_pubmed_df):
    assert std_pubmed_df["SR"].apply(lambda v: isinstance(v, str) and len(v) > 0).all()


def test_pipeline_db_set_correctly_pubmed(std_pubmed_df):
    assert (std_pubmed_df["DB"] == "PUBMED").all()


def test_pipeline_db_set_correctly_openalex(std_openalex_df):
    assert (std_openalex_df["DB"] == "OPENALEX").all()


def test_pipeline_openalex_no_nan(std_openalex_df):
    scalar_cols = [c for c in std_openalex_df.columns if c not in LIST_FIELDS]
    assert std_openalex_df[scalar_cols].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Phase 3 — validate()
# ---------------------------------------------------------------------------

def test_validate_passes_on_good_data(std_pubmed_df):
    report = validate(std_pubmed_df)
    assert report["passed"] is True
    assert all(c["passed"] for c in report["checks"])


def test_validate_report_structure(std_pubmed_df):
    report = validate(std_pubmed_df)
    assert "passed" in report
    assert "checks" in report
    assert len(report["checks"]) == 3
    for check in report["checks"]:
        assert "name" in check
        assert "passed" in check
        assert "problem_columns" in check


def test_validate_raises_on_missing_column(std_pubmed_df):
    broken = std_pubmed_df.drop(columns=["TI"])
    with pytest.raises(ValidationError, match="TI"):
        validate(broken)


def test_validate_raises_on_nan_value(std_pubmed_df):
    broken = std_pubmed_df.copy()
    broken.loc[0, "SO"] = None
    with pytest.raises(ValidationError):
        validate(broken)


def test_validate_raises_on_non_list_field(std_pubmed_df):
    broken = std_pubmed_df.copy()
    broken["AU"] = broken["AU"].apply(lambda v: ";".join(v) if isinstance(v, list) else v)
    with pytest.raises(ValidationError, match="AU"):
        validate(broken)


# ===========================================================================
# FILE SOURCE TESTS
#
# Each test loads a real sample file, runs the pipeline, and asserts
# schema correctness. Marked @pytest.mark.file_sources so they can be
# excluded on machines without the sources/ directory.
# ===========================================================================

def _assert_standardized(df: pd.DataFrame, source_name: str) -> None:
    """Shared post-pipeline assertions for any standardized DataFrame."""
    assert isinstance(df, pd.DataFrame), f"{source_name}: result is not a DataFrame"
    assert len(df) > 0, f"{source_name}: pipeline returned empty DataFrame"

    # All mandatory columns present
    for col in MANDATORY_COLUMNS:
        assert col in df.columns, f"{source_name}: mandatory column missing: {col}"

    # No NaN / None in scalar columns
    scalar_cols = [c for c in df.columns if c not in LIST_FIELDS]
    total_nan = df[scalar_cols].isnull().sum().sum()
    assert total_nan == 0, f"{source_name}: {total_nan} NaN values found in scalar columns"

    # List fields must be list[str]
    for col in LIST_FIELDS:
        if col in df.columns:
            non_list = df[col].apply(lambda v: not isinstance(v, list))
            assert not non_list.any(), f"{source_name}: {col} has non-list values"

    # TC must be int
    assert df["TC"].apply(lambda v: isinstance(v, int)).all(), \
        f"{source_name}: TC has non-int values"

    # PY must be empty string or 4-digit string
    for val in df["PY"]:
        assert val == "" or (len(val) == 4 and val.isdigit()), \
            f"{source_name}: PY value {val!r} is not 4-digit year"

    # SR must be non-empty string
    assert df["SR"].apply(lambda v: isinstance(v, str) and len(v) > 0).all(), \
        f"{source_name}: SR has empty / non-string values"


# ---- Individual load_file() smoke tests ----

@pytest.mark.file_sources
@pytest.mark.skipif(not _SCOPUS_CSV.exists(), reason="Scopus CSV sample not found")
def test_load_file_scopus_csv():
    df, src = load_file(_SCOPUS_CSV)
    assert src == "SCOPUS_CSV"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "EID" in df.columns or "Authors" in df.columns


@pytest.mark.file_sources
@pytest.mark.skipif(not _SCOPUS_BIB.exists(), reason="Scopus BibTeX sample not found")
def test_load_file_scopus_bib():
    df, src = load_file(_SCOPUS_BIB)
    assert src == "SCOPUS_BIB"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "author" in df.columns or "AU" in df.columns


@pytest.mark.file_sources
@pytest.mark.skipif(not _WOS_TXT.exists(), reason="WoS TXT sample not found")
def test_load_file_wos_txt():
    df, src = load_file(_WOS_TXT)
    assert src == "WOS_TXT"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.file_sources
@pytest.mark.skipif(not _WOS_CIW.exists(), reason="WoS CIW sample not found")
def test_load_file_wos_ciw():
    df, src = load_file(_WOS_CIW)
    assert src == "WOS_TXT"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.file_sources
@pytest.mark.skipif(not _WOS_BIB.exists(), reason="WoS BibTeX sample not found")
def test_load_file_wos_bib():
    df, src = load_file(_WOS_BIB)
    assert src == "WOS_BIB"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "author" in df.columns or "AU" in df.columns


@pytest.mark.file_sources
@pytest.mark.skipif(not _DIMENSIONS.exists(), reason="Dimensions CSV sample not found")
def test_load_file_dimensions_csv():
    df, src = load_file(_DIMENSIONS)
    assert src == "DIMENSIONS"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # Dimensions header at row 1; "Publication ID" should appear after skiprows=1
    assert "Publication ID" in df.columns or "PubYear" in df.columns


@pytest.mark.file_sources
@pytest.mark.skipif(not _DIM_XLSX.exists(), reason="Dimensions XLSX sample not found")
def test_load_file_dimensions_xlsx():
    df, src = load_file(_DIM_XLSX)
    assert src == "DIMENSIONS"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.file_sources
@pytest.mark.skipif(not _LENS.exists(), reason="Lens CSV sample not found")
def test_load_file_lens():
    df, src = load_file(_LENS)
    assert src == "LENS"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Lens ID" in df.columns or "Author/s" in df.columns


@pytest.mark.file_sources
@pytest.mark.skipif(not _COCHRANE.exists(), reason="Cochrane TXT sample not found")
def test_load_file_cochrane():
    df, src = load_file(_COCHRANE)
    assert src == "COCHRANE"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.file_sources
@pytest.mark.skipif(not _PUBMED_FILE.exists(), reason="PubMed TXT sample not found")
def test_load_file_pubmed_file():
    df, src = load_file(_PUBMED_FILE)
    assert src == "PUBMED_FILE"
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # MEDLINE format must include PMID field
    assert "PMID" in df.columns


# ---- Explicit source override ----

@pytest.mark.file_sources
@pytest.mark.skipif(not _WOS_CIW.exists(), reason="WoS CIW sample not found")
def test_load_file_explicit_source_override():
    """load_file() must respect an explicit source= argument."""
    df, src = load_file(_WOS_CIW, source="WOS_TXT")
    assert src == "WOS_TXT"
    assert len(df) > 0


# ---- Parametrized full pipeline tests ----

_PIPELINE_PARAMS = [
    pytest.param("SCOPUS_CSV",   _SCOPUS_CSV,  id="scopus_csv"),
    pytest.param("SCOPUS_BIB",   _SCOPUS_BIB,  id="scopus_bib"),
    pytest.param("WOS_TXT",      _WOS_TXT,     id="wos_txt"),
    pytest.param("WOS_BIB",      _WOS_BIB,     id="wos_bib"),
    pytest.param("DIMENSIONS",   _DIMENSIONS,  id="dimensions_csv"),
    pytest.param("LENS",         _LENS,        id="lens_csv"),
    pytest.param("COCHRANE",     _COCHRANE,    id="cochrane_txt"),
    pytest.param("PUBMED_FILE",  _PUBMED_FILE, id="pubmed_file"),
]


@pytest.mark.file_sources
@pytest.mark.parametrize("source,filepath", _PIPELINE_PARAMS)
def test_pipeline_file_source(source: str, filepath: Path):
    """Load a file, run the full ETL pipeline, assert schema correctness."""
    if not filepath.exists():
        pytest.skip(f"Sample file not found: {filepath}")

    raw_df, detected_source = load_file(filepath, source=source)
    assert len(raw_df) > 0, f"{source}: loaded empty DataFrame"

    std_df = run_pipeline(raw_df, source=detected_source)
    _assert_standardized(std_df, source)


@pytest.mark.file_sources
@pytest.mark.parametrize("source,filepath", _PIPELINE_PARAMS)
def test_validate_file_output(source: str, filepath: Path):
    """Pipeline output for every file source must pass validate() without errors."""
    if not filepath.exists():
        pytest.skip(f"Sample file not found: {filepath}")

    raw_df, detected_source = load_file(filepath, source=source)
    std_df = run_pipeline(raw_df, source=detected_source)
    report = validate(std_df)

    assert report["passed"] is True, (
        f"{source}: validation failed. "
        f"Checks: {[c for c in report['checks'] if not c['passed']]}"
    )


# ---- Source-specific column-mapping spot checks ----

@pytest.mark.file_sources
@pytest.mark.skipif(not _SCOPUS_CSV.exists(), reason="Scopus CSV sample not found")
def test_scopus_csv_db_value():
    raw, src = load_file(_SCOPUS_CSV)
    df = run_pipeline(raw, source=src)
    assert (df["DB"] == "SCOPUS").all(), f"Expected DB='SCOPUS', got {df['DB'].unique()}"


@pytest.mark.file_sources
@pytest.mark.skipif(not _WOS_TXT.exists(), reason="WoS TXT sample not found")
def test_wos_txt_db_value():
    raw, src = load_file(_WOS_TXT)
    df = run_pipeline(raw, source=src)
    assert (df["DB"] == "WOS").all(), f"Expected DB='WOS', got {df['DB'].unique()}"


@pytest.mark.file_sources
@pytest.mark.skipif(not _DIMENSIONS.exists(), reason="Dimensions CSV sample not found")
def test_dimensions_csv_db_value():
    raw, src = load_file(_DIMENSIONS)
    df = run_pipeline(raw, source=src)
    assert (df["DB"] == "DIMENSIONS").all(), f"Expected DB='DIMENSIONS', got {df['DB'].unique()}"


@pytest.mark.file_sources
@pytest.mark.skipif(not _LENS.exists(), reason="Lens CSV sample not found")
def test_lens_db_value():
    raw, src = load_file(_LENS)
    df = run_pipeline(raw, source=src)
    assert (df["DB"] == "LENS").all(), f"Expected DB='LENS', got {df['DB'].unique()}"


@pytest.mark.file_sources
@pytest.mark.skipif(not _COCHRANE.exists(), reason="Cochrane TXT sample not found")
def test_cochrane_db_value():
    raw, src = load_file(_COCHRANE)
    df = run_pipeline(raw, source=src)
    assert (df["DB"] == "COCHRANE").all(), f"Expected DB='COCHRANE', got {df['DB'].unique()}"


@pytest.mark.file_sources
@pytest.mark.skipif(not _PUBMED_FILE.exists(), reason="PubMed TXT sample not found")
def test_pubmed_file_db_value():
    raw, src = load_file(_PUBMED_FILE)
    df = run_pipeline(raw, source=src)
    assert (df["DB"] == "PUBMED").all(), f"Expected DB='PUBMED', got {df['DB'].unique()}"


@pytest.mark.file_sources
@pytest.mark.skipif(not _SCOPUS_BIB.exists(), reason="Scopus BibTeX sample not found")
def test_scopus_bib_au_is_list():
    """BibTeX 'and'-delimited author string must be split into list[str]."""
    raw, src = load_file(_SCOPUS_BIB)
    df = run_pipeline(raw, source=src)
    assert df["AU"].apply(lambda v: isinstance(v, list)).all(), \
        "SCOPUS_BIB: AU field not converted to list[str]"
    # Ensure at least one record has >1 author (confirming 'and' split works)
    multi_author = df["AU"].apply(lambda v: len(v) > 1)
    # This is a soft check: if all records happen to be single-author, skip
    if not multi_author.any():
        pytest.skip("All Scopus BibTeX records are single-author; cannot verify 'and' split")


@pytest.mark.file_sources
@pytest.mark.skipif(not _WOS_BIB.exists(), reason="WoS BibTeX sample not found")
def test_wos_bib_ut_starts_with_wos():
    """WoS BibTeX entry keys (e.g. 'WOS:001...') must become the UT field."""
    raw, src = load_file(_WOS_BIB)
    df = run_pipeline(raw, source=src)
    non_empty_ut = df["UT"][df["UT"] != ""]
    if len(non_empty_ut) == 0:
        pytest.skip("WoS BibTeX sample has no UT values")
    assert non_empty_ut.str.startswith("WOS:").any(), \
        "WOS_BIB: UT values do not start with 'WOS:'"


@pytest.mark.file_sources
@pytest.mark.skipif(not _DIMENSIONS.exists(), reason="Dimensions CSV sample not found")
def test_dimensions_py_4digit():
    """Dimensions PubYear must be extracted as a 4-digit string."""
    raw, src = load_file(_DIMENSIONS)
    df = run_pipeline(raw, source=src)
    non_empty = df["PY"][df["PY"] != ""]
    assert (non_empty.str.len() == 4).all(), \
        f"Dimensions: PY not 4-digit strings — sample: {non_empty.head().tolist()}"


# ---- export_to_csv round-trip ----

@pytest.mark.file_sources
@pytest.mark.skipif(not _SCOPUS_CSV.exists(), reason="Scopus CSV sample not found")
def test_export_to_csv_round_trip(tmp_path: Path):
    """CSV export must produce a file readable as a DataFrame with correct columns."""
    raw, src = load_file(_SCOPUS_CSV)
    std_df = run_pipeline(raw, source=src)
    csv_path = export_to_csv(std_df, src, output_dir=str(tmp_path))

    assert csv_path.exists(), "export_to_csv did not create the file"
    reloaded = pd.read_csv(csv_path)
    assert len(reloaded) == len(std_df), "Row count mismatch after CSV round-trip"
    for col in MANDATORY_COLUMNS:
        assert col in reloaded.columns, f"Mandatory column {col} missing from CSV"
