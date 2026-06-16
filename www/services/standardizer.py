"""
ETL Transform pipeline — Phase 2 + file loading.

Each function handles exactly one responsibility. run_pipeline() is the
only orchestrator; it calls each step in order and must not contain
any field-level logic itself.

The existing SR(M) function from metatagextraction.py is called by
add_calculated_fields() — it is never reimplemented here.
"""

import logging
import re
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd

from .mapping_dicts import (
    COCHRANE_MAP,
    DIMENSIONS_MAP,
    LENS_MAP,
    LIST_FIELDS,
    MANDATORY_COLUMNS,
    OPENALEX_MAP,
    PUBMED_FILE_MAP,
    PUBMED_MAP,
    SCALAR_FIELDS,
    SCOPUS_BIB_MAP,
    SCOPUS_CSV_MAP,
    SOURCE_TO_DB,
    WOS_BIB_MAP,
    WOS_TXT_MAP,
)

logger = logging.getLogger(__name__)

# All supported source identifiers (internal routing codes)
Source = Literal[
    "PUBMED",       # PubMed E-utilities API
    "OPENALEX",     # OpenAlex REST API
    "PUBMED_FILE",  # PubMed MEDLINE plaintext file
    "SCOPUS_CSV",   # Scopus CSV export
    "SCOPUS_BIB",   # Scopus BibTeX export
    "WOS_TXT",      # Web of Science TXT/CIW plaintext
    "WOS_BIB",      # Web of Science BibTeX export
    "DIMENSIONS",   # Dimensions CSV or XLSX export
    "LENS",         # Lens.org CSV export
    "COCHRANE",     # Cochrane CDSR TXT export
]

# Maps each Source code to the mapping dictionary for rename_columns()
_MAPPING_REGISTRY: dict[str, dict[str, str]] = {
    "PUBMED":      PUBMED_MAP,
    "OPENALEX":    OPENALEX_MAP,
    "PUBMED_FILE": PUBMED_FILE_MAP,
    "SCOPUS_CSV":  SCOPUS_CSV_MAP,
    "SCOPUS_BIB":  SCOPUS_BIB_MAP,
    "WOS_TXT":     WOS_TXT_MAP,
    "WOS_BIB":     WOS_BIB_MAP,
    "DIMENSIONS":  DIMENSIONS_MAP,
    "LENS":        LENS_MAP,
    "COCHRANE":    COCHRANE_MAP,
}

# Column fingerprints used by detect_source() fallback path
_PUBMED_SIGNATURE   = {"PMID", "TI", "AU", "FAU", "MH", "DP"}
_OPENALEX_SIGNATURE = {"display_name", "cited_by_count", "openalex_id", "author_names"}
_SCOPUS_CSV_SIG     = {"EID", "Authors", "Source title"}
_DIMENSIONS_SIG     = {"Publication ID", "PubYear", "Times cited"}
_LENS_SIG           = {"Lens ID", "Author/s", "Citing Works Count"}
_COCHRANE_SIG       = {"KY", "YR"}   # after parse_cochrane_data

# DB column value → default Source code (used by detect_source())
_DB_TO_SOURCE: dict[str, Source] = {
    "PUBMED":     "PUBMED",
    "OPENALEX":   "OPENALEX",
    "SCOPUS":     "SCOPUS_CSV",   # default to CSV when file format unspecified
    "WOS":        "WOS_TXT",      # default to TXT when file format unspecified
    "DIMENSIONS": "DIMENSIONS",
    "LENS":       "LENS",
    "COCHRANE":   "COCHRANE",
}

# Lazy import gate for text-file parsers (parsers.py → from .utils import *)
try:
    from .parsers import parse_cochrane_data, parse_pubmed_data, parse_wos_data
    _PARSERS_OK = True
except Exception:
    _PARSERS_OK = False
    logger.warning(
        "parsers module could not be imported; WoS TXT, PubMed TXT and "
        "Cochrane file sources will be unavailable until the environment "
        "is configured correctly."
    )
    parse_wos_data = parse_pubmed_data = parse_cochrane_data = None  # type: ignore[assignment]

# Pre-compiled patterns for source-specific field extraction
_CITED_BY_RE = re.compile(r"Cited\s+by:\s*(\d+)", re.IGNORECASE)
_EID_RE      = re.compile(r"eid=([\w.\-]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Phase 1 (file) — load raw DataFrame from a local file
# ---------------------------------------------------------------------------

def _detect_file_source(fp: Path, ext: str) -> Source:
    """
    Auto-detect the bibliographic source from file extension and content.

    Args:
        fp: Path to the file.
        ext: Lowercase file extension (e.g. ".csv").

    Returns:
        Source literal for the detected format.

    Raises:
        ValueError: If the source cannot be determined.
    """
    if ext in (".xlsx", ".xls"):
        return "DIMENSIONS"

    if ext == ".bib":
        text = fp.read_text(encoding="utf-8", errors="replace")[:3000]
        if re.search(r'\{\s*WOS:', text):
            return "WOS_BIB"
        if "abbrev_source_title" in text.lower():
            return "SCOPUS_BIB"
        # Heuristic: entry keys that look like "AuthorYear" → Scopus
        return "SCOPUS_BIB"

    if ext in (".txt", ".ciw"):
        with open(fp, encoding="utf-8", errors="replace") as fh:
            head = "".join(fh.readline() for _ in range(6))

        if re.search(r'^PMID-\s|^PMID  -\s', head, re.MULTILINE):
            return "PUBMED_FILE"
        if re.search(r'^Record #\d+', head, re.MULTILINE):
            return "COCHRANE"
        if re.search(r'^(FN|VR|PT)\s', head, re.MULTILINE) or ext == ".ciw":
            return "WOS_TXT"
        # Additional content check for PubMed files that don't start with PMID
        with open(fp, encoding="utf-8", errors="replace") as fh:
            body = fh.read(800)
        if "PMID-" in body or "PMID  -" in body:
            return "PUBMED_FILE"
        if "Record #" in body:
            return "COCHRANE"
        return "WOS_TXT"

    if ext == ".csv":
        try:
            hdr = pd.read_csv(fp, nrows=0)
            cols: set[str] = set(hdr.columns)
        except Exception:
            cols = set()

        if "EID" in cols:
            return "SCOPUS_CSV"
        if "Lens ID" in cols:
            return "LENS"
        if "Publication ID" in cols and "PubYear" in cols:
            return "DIMENSIONS"

        # Try Dimensions with skiprows=1 (row 0 is a title/export header)
        try:
            hdr2 = pd.read_csv(fp, skiprows=1, nrows=0)
            cols2: set[str] = set(hdr2.columns)
            if "Publication ID" in cols2 and "PubYear" in cols2:
                return "DIMENSIONS"
        except Exception:
            pass

        raise ValueError(
            f"Cannot detect CSV source from headers: {sorted(cols)}. "
            "Pass source= explicitly to load_file()."
        )

    raise ValueError(
        f"Unsupported file extension: {ext!r}. "
        "Supported: .csv, .bib, .txt, .ciw, .xlsx, .xls"
    )


def _load_bib(fp: Path) -> pd.DataFrame:
    """
    Parse a BibTeX file with bibtexparser v1.x and return a raw DataFrame.

    Args:
        fp: Path to the .bib file.

    Returns:
        DataFrame where each row is one BibTeX entry; field names are lowercase.

    Raises:
        ImportError: If bibtexparser is not installed.
        ValueError: If the file contains no entries.
    """
    try:
        import bibtexparser  # noqa: PLC0415
        from bibtexparser.bparser import BibTexParser  # noqa: PLC0415
        from bibtexparser.customization import convert_to_unicode  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "bibtexparser is required for BibTeX file support. "
            "Install it with: pip install bibtexparser==1.4.1"
        ) from exc

    with open(fp, encoding="utf-8", errors="replace") as fh:
        parser = BibTexParser()
        parser.customization = convert_to_unicode
        db = bibtexparser.load(fh, parser=parser)

    if not db.entries:
        raise ValueError(f"No BibTeX entries found in {fp}")

    return pd.DataFrame(db.entries)


def _normalize_wos_txt(records: list[dict]) -> pd.DataFrame:
    """
    Convert parse_wos_data() output to a flat DataFrame.

    parse_wos_data() returns dicts where each value is a list. Multi-element
    lists (AU, CR, etc.) are joined with ";" so that _split_to_list() in
    enforce_types can reconstruct them. Single-element lists (TI, PY, etc.)
    are unwrapped to strings.

    Args:
        records: List of record dicts from parse_wos_data().

    Returns:
        Flat DataFrame ready for rename_columns() and enforce_types().
    """
    flat: list[dict] = []
    for rec in records:
        row: dict = {}
        for key, value in rec.items():
            if isinstance(value, list):
                row[key] = ";".join(str(v) for v in value if str(v).strip())
            elif value is None:
                row[key] = ""
            else:
                row[key] = str(value)
        flat.append(row)
    return pd.DataFrame(flat)


def load_file(
    filepath: "str | Path",
    source: "Source | None" = None,
) -> "tuple[pd.DataFrame, Source]":
    """
    Load a bibliographic export file into a raw DataFrame.

    Supported formats:
        Scopus CSV, Scopus BibTeX, WoS TXT/CIW, WoS BibTeX,
        Dimensions CSV/XLSX, Lens.org CSV, Cochrane CDSR TXT,
        PubMed MEDLINE TXT.

    The returned DataFrame uses source-native column names; downstream
    rename_columns() translates them to WoS Field Tags using the
    appropriate mapping dictionary.

    Args:
        filepath: Path (or str) to the file to load.
        source: Optional explicit source override. When None, the source is
            auto-detected from the file extension and content fingerprint.

    Returns:
        Tuple of (raw_df, source_label) where source_label is a Source literal
        that should be passed directly to run_pipeline().

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If source cannot be determined or format is unsupported.
        ImportError: If a required optional library (bibtexparser) is absent.
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")

    ext = fp.suffix.lower()

    if source is None:
        source = _detect_file_source(fp, ext)

    if source == "SCOPUS_CSV":
        df = pd.read_csv(fp, encoding="utf-8", on_bad_lines="skip", low_memory=False)

    elif source == "SCOPUS_BIB":
        df = _load_bib(fp)

    elif source == "WOS_TXT":
        if not _PARSERS_OK:
            raise ImportError("parsers module unavailable; cannot load WoS TXT files")
        records = parse_wos_data(str(fp))
        df = _normalize_wos_txt(records)

    elif source == "WOS_BIB":
        df = _load_bib(fp)

    elif source == "DIMENSIONS":
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(fp, skiprows=1, engine="openpyxl")
        else:
            # Dimensions CSV has a title row as row 0; actual header is row 1
            try:
                df = pd.read_csv(fp, skiprows=1, encoding="utf-8",
                                 on_bad_lines="skip", low_memory=False)
                if "Publication ID" not in df.columns:
                    # Fallback: header was already row 0
                    df = pd.read_csv(fp, encoding="utf-8",
                                     on_bad_lines="skip", low_memory=False)
            except Exception:
                df = pd.read_csv(fp, encoding="utf-8",
                                 on_bad_lines="skip", low_memory=False)

    elif source == "LENS":
        df = pd.read_csv(fp, encoding="utf-8", on_bad_lines="skip", low_memory=False)

    elif source == "COCHRANE":
        if not _PARSERS_OK:
            raise ImportError("parsers module unavailable; cannot load Cochrane TXT files")
        records = parse_cochrane_data(str(fp))
        df = pd.DataFrame(records)

    elif source == "PUBMED_FILE":
        if not _PARSERS_OK:
            raise ImportError("parsers module unavailable; cannot load PubMed TXT files")
        records = parse_pubmed_data(str(fp))
        df = pd.DataFrame(records)

    else:
        raise ValueError(f"Unsupported source: {source!r}")

    if df.empty:
        logger.warning("load_file: zero records loaded from %s (source=%s)", fp.name, source)
    else:
        logger.info("load_file: %d records from %s (source=%s)", len(df), fp.name, source)

    return df, source


# ---------------------------------------------------------------------------
# Step 2a — detect source
# ---------------------------------------------------------------------------

def detect_source(df: pd.DataFrame) -> Source:
    """
    Identify the data source from the DB column or column name fingerprint.

    Checks the DB column first (set by load_file() or the API retriever);
    falls back to column signature matching for API DataFrames that have
    not yet been processed.

    Args:
        df: Raw or pre-processed DataFrame.

    Returns:
        Source literal string.

    Raises:
        ValueError: If the source cannot be determined.
    """
    if "DB" in df.columns and len(df) > 0:
        db_val = str(df["DB"].iloc[0]).upper()
        if db_val in _DB_TO_SOURCE:
            return _DB_TO_SOURCE[db_val]

    cols = set(df.columns)

    if _OPENALEX_SIGNATURE & cols:
        return "OPENALEX"
    if {"LID", "FAU", "PMID"} <= cols:
        return "PUBMED_FILE"
    if {"AID", "FAU", "PMID"} <= cols:
        return "PUBMED"
    if _PUBMED_SIGNATURE & cols:
        return "PUBMED"
    if _SCOPUS_CSV_SIG & cols:
        return "SCOPUS_CSV"
    if _DIMENSIONS_SIG & cols:
        return "DIMENSIONS"
    if _LENS_SIG & cols:
        return "LENS"
    if _COCHRANE_SIG & cols:
        return "COCHRANE"

    raise ValueError(
        f"Cannot detect source. Columns present: {sorted(cols)}. "
        "Call run_pipeline(df, source='PUBMED') or similar to specify explicitly."
    )


# ---------------------------------------------------------------------------
# Step 2a — rename columns
# ---------------------------------------------------------------------------

def rename_columns(df: pd.DataFrame, source: Source) -> pd.DataFrame:
    """
    Rename source-native columns to WoS Field Tags using the mapping
    dictionary for the given source.

    Columns not in the mapping are kept as-is. Pre-existing columns whose
    name matches a rename target (e.g. PubMed raw "SO" vs "JT"→"SO") are
    dropped before renaming to prevent duplicate columns.

    Args:
        df: Raw DataFrame with source-native column names.
        source: Source literal determining which mapping dict to use.

    Returns:
        DataFrame with columns renamed to WoS Field Tags.

    Raises:
        KeyError: If source is not in the mapping registry.
    """
    mapping = _MAPPING_REGISTRY[source]
    effective_map = {k: v for k, v in mapping.items() if k in df.columns}

    # Drop raw columns whose name already equals a rename target.
    # This prevents duplicate columns when e.g. PubMed raw "SO" (source citation
    # field) would collide with the renamed "JT"→"SO".
    target_names = set(effective_map.values())
    cols_to_drop = [
        c for c in df.columns
        if c in target_names and c not in effective_map
    ]
    if cols_to_drop:
        logger.debug("rename_columns (%s): dropping shadowed cols %s", source, cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    renamed = df.rename(columns=effective_map)

    # Safety net: de-duplicate any remaining double columns (keep first).
    if renamed.columns.duplicated().any():
        renamed = renamed.loc[:, ~renamed.columns.duplicated(keep="first")]

    logger.debug("rename_columns (%s): applied %d renames", source, len(effective_map))
    return renamed


# ---------------------------------------------------------------------------
# Step 2b — type-enforcement helpers
# ---------------------------------------------------------------------------

def _split_to_list(value: object, sep: str = ";") -> list[str]:
    """Split a delimited string into list[str]; return [] for nulls."""
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    return [x.strip() for x in s.split(sep) if x.strip()]


def _split_bib_authors(value: object) -> list[str]:
    """
    Split a BibTeX author string into a list of individual author names.

    BibTeX format: "Surname, Name and Surname2, Name2 and Surname3, Name3".
    Falls back to semicolon splitting for non-BibTeX formats.
    """
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    if " and " in s:
        return [x.strip() for x in s.split(" and ") if x.strip()]
    return [x.strip() for x in s.split(";") if x.strip()]


def _extract_year(value: object) -> str:
    """Extract a 4-digit year from a date string; return '' if none found."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    m = re.search(r"\b(\d{4})\b", s)
    return m.group(1) if m else ""


def _clean_doi(value: object) -> str:
    """
    Extract a clean DOI from various raw formats.

    Handles PubMed AID/LID suffix format ("10.1234/abc [doi]"),
    bare DOIs, and OpenAlex doi strings.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return ""
    # PubMed AID/LID: pick the entry tagged [doi]
    m = re.search(r"(10\.\S+?)\s*\[doi\]", raw)
    if m:
        return m.group(1).strip()
    # Bare DOI starting with the registrant prefix
    if raw.startswith("10."):
        return raw.split(";")[0].strip()
    return ""


def _extract_ep_from_pages(bp_value: object) -> "tuple[str, str]":
    """
    Split a page range into (begin_page, end_page).

    Handles "100-110", "100 - 110" (Scopus BibTeX), and "e12345" formats.

    Args:
        bp_value: Raw page string.

    Returns:
        Tuple (BP, EP); EP is '' if no range separator is present.
    """
    raw = str(bp_value).strip() if not (isinstance(bp_value, float) and pd.isna(bp_value)) else ""
    m = re.match(r'^(\S+)\s*-\s*(\S+)$', raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw, ""


def _unwrap_list_scalar(value: object) -> object:
    """Unwrap single-element lists (produced by parse_wos_data) to scalars."""
    if isinstance(value, list):
        return value[0] if len(value) == 1 else (";".join(str(v) for v in value) if value else "")
    return value


def _extract_dim_affiliations(value: object) -> list[str]:
    """
    Extract institution strings from Dimensions 'Authors (Raw Affiliation)' format.

    Format: "Surname, Name (Institution, City, Country); Surname2, Name2 (...)".
    Returns a list of the institution strings inside the parentheses.
    """
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return []
    # Extract content inside parentheses as affiliation strings
    found = re.findall(r'\(([^)]+)\)', raw)
    if found:
        return [a.strip() for a in found if a.strip()]
    # Fallback: split on ";"
    return [x.strip() for x in raw.split(";") if x.strip()]


# ---------------------------------------------------------------------------
# Step 2b — enforce type contracts
# ---------------------------------------------------------------------------

def enforce_types(df: pd.DataFrame, source: "Source | None" = None) -> pd.DataFrame:
    """
    Apply all WoS schema type contracts to the renamed DataFrame.

    Contracts applied:
    - Source-specific pre-processing (BibTeX author split, Scopus TC/UT,
      Dimensions affiliation extraction).
    - LIST_FIELDS  → list[str] (split on ";").
    - SCALAR_FIELDS → str ("" for missing/null).
    - PY           → 4-digit year string.
    - TC           → int (0 if unparseable).
    - DI           → clean DOI string.
    - BP / EP      → derived from page range when EP is absent.

    Args:
        df: DataFrame after rename_columns().
        source: Source literal; used for source-specific pre-processing.

    Returns:
        DataFrame with enforced types. Input is not modified.
    """
    df = df.copy()

    # ---- Source-specific pre-processing ----

    # BibTeX sources: split "Author A and Author B" strings for AU/AF
    if source in ("SCOPUS_BIB", "WOS_BIB"):
        if "AU" in df.columns:
            df["AU"] = df["AU"].apply(_split_bib_authors)
        if "AF" not in df.columns:
            # BibTeX has only one author field; use AU for both AU and AF
            df["AF"] = df["AU"].copy() if "AU" in df.columns else [[] for _ in range(len(df))]

    # Scopus BibTeX: TC comes from "note" field ("Cited by: N; ...")
    if source == "SCOPUS_BIB" and "TC" in df.columns:
        def _extract_bib_tc(v: object) -> str:
            m = _CITED_BY_RE.search(str(v)) if v and not (isinstance(v, float) and pd.isna(v)) else None
            return m.group(1) if m else "0"
        df["TC"] = df["TC"].apply(_extract_bib_tc)

    # Scopus BibTeX: UT comes from Scopus URL ("...?eid=2-s2.0-...")
    if source == "SCOPUS_BIB" and "UT" in df.columns:
        def _extract_eid(v: object) -> str:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return ""
            m = _EID_RE.search(str(v))
            return m.group(1) if m else str(v).strip()
        df["UT"] = df["UT"].apply(_extract_eid)

    # Dimensions: C1 contains "Author (Affiliation)" format
    if source == "DIMENSIONS" and "C1" in df.columns:
        df["C1"] = df["C1"].apply(_extract_dim_affiliations)

    # ---- LIST_FIELDS — split to list[str] ----

    for col in LIST_FIELDS:
        if col in df.columns:
            # BibTeX AU/AF already handled above — respect existing lists
            if source in ("SCOPUS_BIB", "WOS_BIB") and col in ("AU", "AF"):
                df[col] = df[col].apply(
                    lambda v: v if isinstance(v, list) else _split_to_list(v)
                )
            elif source == "DIMENSIONS" and col == "C1":
                df[col] = df[col].apply(
                    lambda v: v if isinstance(v, list) else _extract_dim_affiliations(v)
                )
            else:
                df[col] = df[col].apply(_split_to_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # ---- SCALAR_FIELDS — convert to str ----

    for col in SCALAR_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(_unwrap_list_scalar)
            df[col] = df[col].fillna("").astype(str).str.strip().replace("nan", "").replace("None", "")
        else:
            df[col] = ""

    # ---- PY — 4-digit year ----

    if "PY" in df.columns:
        df["PY"] = df["PY"].apply(_unwrap_list_scalar).apply(_extract_year)
    else:
        df["PY"] = ""

    # ---- TC — integer citation count ----

    if "TC" in df.columns:
        df["TC"] = pd.to_numeric(df["TC"], errors="coerce").fillna(0).astype(int)
    else:
        df["TC"] = 0

    # ---- DI — clean DOI ----

    if "DI" in df.columns:
        df["DI"] = df["DI"].apply(_unwrap_list_scalar).apply(_clean_doi)

    # ---- BP / EP — page range splitting ----

    if "BP" in df.columns:
        df["BP"] = df["BP"].apply(_unwrap_list_scalar)
        has_ep = "EP" in df.columns
        pages_split = df["BP"].apply(_extract_ep_from_pages)
        df["BP"] = pages_split.apply(lambda t: t[0])
        if not has_ep:
            df["EP"] = pages_split.apply(lambda t: t[1])
        else:
            df["EP"] = df["EP"].apply(_unwrap_list_scalar)
            derived_ep = pages_split.apply(lambda t: t[1])
            df["EP"] = df["EP"].fillna("").astype(str).str.strip().replace("nan", "")
            mask = df["EP"] == ""
            df.loc[mask, "EP"] = derived_ep[mask]
    else:
        df["BP"] = ""
        if "EP" not in df.columns:
            df["EP"] = ""

    logger.debug("enforce_types complete: shape=%s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 2c — null handling
# ---------------------------------------------------------------------------

def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace all remaining NaN / None values per field-type rules.

    Rules:
    - LIST_FIELDS → []
    - TC          → 0
    - All other columns → ''

    Args:
        df: DataFrame after enforce_types().

    Returns:
        DataFrame with zero NaN / None values. Input is not modified.
    """
    df = df.copy()

    for col in df.columns:
        if col in LIST_FIELDS:
            df[col] = df[col].apply(lambda v: v if isinstance(v, list) else [])
        elif col == "TC":
            df[col] = df[col].fillna(0).astype(int)
        else:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .replace("nan", "")
                .replace("None", "")
            )

    logger.debug("handle_nulls complete: NaN count=%d", df.isnull().sum().sum())
    return df


# ---------------------------------------------------------------------------
# Step 2d — calculated fields
# ---------------------------------------------------------------------------

def _compute_sr_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SR when metatagextraction.SR(M) cannot be imported.

    Mirrors the SR(M) logic from www/services/metatagextraction.py so that
    both the Shiny and Streamlit environments produce identical SR values.
    Called only when the primary import fails.

    Args:
        df: DataFrame with AU, JI, SO, PY, DB columns already set.

    Returns:
        DataFrame with SR and SR_FULL columns added in-place.
    """
    listAU = df["AU"].apply(lambda l: [str(x).strip() for x in l] if isinstance(l, list) else [])

    if df["DB"].iloc[0].lower() == "scopus":
        listAU = listAU.apply(
            lambda l: [x.replace(" ", ",").replace(",,", ",").replace(" ", "") for x in l]
        )

    first_authors = listAU.apply(lambda l: l[0] if l else "NA")
    first_authors = first_authors.str.replace(",", " ", regex=False)

    no_art = df["JI"] == ""
    df.loc[no_art, "JI"] = df.loc[no_art, "SO"]
    j9 = df["JI"].str.replace(".", " ", regex=False).str.strip()

    sr = first_authors + ", " + df["PY"].astype(str) + ", " + j9
    df["SR_FULL"] = sr.str.replace(r"\s+", " ", regex=True)

    # Disambiguate duplicate SR values by appending -a, -b, ...
    st_flag = i = 0
    while st_flag == 0:
        ind = sr.duplicated()
        if ind.any():
            i += 1
            sr[ind] = sr[ind] + "-" + chr(96 + i)
        else:
            st_flag = 1
    df["SR"] = sr.str.replace(r"\s+", " ", regex=True)
    return df


def add_calculated_fields(df: pd.DataFrame, source: Source) -> pd.DataFrame:
    """
    Set the DB column and compute the SR (Short Reference) field.

    Preferred path: calls SR(M) from www/services/metatagextraction.py.
    Falls back to _compute_sr_fallback() if that module cannot be loaded
    (e.g. Shiny-specific deps are absent in the Streamlit environment).

    Args:
        df: DataFrame after handle_nulls().
        source: Source literal; mapped to the DB column value via SOURCE_TO_DB.

    Returns:
        DataFrame with DB and SR columns populated.
    """
    df = df.copy()
    df["DB"] = SOURCE_TO_DB.get(source, source)

    # Ensure JI has a fallback value before SR computation
    if "JI" not in df.columns or (df["JI"].astype(str).str.strip() == "").all():
        df["JI"] = df.get("SO", pd.Series("", index=df.index))

    try:
        from .metatagextraction import SR as compute_sr  # noqa: PLC0415
        df = compute_sr(df)
    except ImportError:
        logger.info("metatagextraction unavailable (Shiny deps absent) — using SR fallback")
        df = _compute_sr_fallback(df)
    except Exception as exc:
        logger.warning("SR computation failed (%s) — using SR fallback", exc)
        df = _compute_sr_fallback(df)

    logger.debug("add_calculated_fields: DB=%s", df["DB"].iloc[0])
    return df


# ---------------------------------------------------------------------------
# Phase 4 — LOAD helper
# ---------------------------------------------------------------------------

def export_to_csv(
    df: pd.DataFrame,
    source: Source,
    output_dir: str = "data/outputs",
) -> Path:
    """
    Serialize the standardized DataFrame to CSV.

    Multi-value list fields are serialized with ";" as delimiter, matching
    the format expected by all analytical functions in www/services/.

    Args:
        df: Fully standardized and validated DataFrame.
        source: Source label; used to build the output filename.
        output_dir: Directory where the CSV will be written.

    Returns:
        Path to the written CSV file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()
    for col in LIST_FIELDS:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(
                lambda v: ";".join(v) if isinstance(v, list) else str(v)
            )

    filename = out / f"{source.lower()}_standardized.csv"
    export_df.to_csv(filename, index=False, encoding="utf-8")
    logger.info("export_to_csv: %d records → %s", len(export_df), filename)
    return filename


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame, source: "Source | None" = None) -> pd.DataFrame:
    """
    Orchestrate the full ETL transform pipeline (Phase 2).

    Steps executed in order:
        1. detect_source  — identify source (skipped if source is provided)
        2. rename_columns — apply mapping dict
        3. enforce_types  — apply all type contracts
        4. handle_nulls   — eliminate NaN / None
        5. add_calculated_fields — set DB, compute SR

    Args:
        df: Raw DataFrame from a fetch_*() function or load_file().
        source: Optional source override. If None, detect_source() is called.

    Returns:
        Fully standardized DataFrame ready for validate() and export_to_csv().
    """
    if source is None:
        source = detect_source(df)
    logger.info("run_pipeline: source=%s, input shape=%s", source, df.shape)

    df = rename_columns(df, source)
    df = enforce_types(df, source)
    df = handle_nulls(df)
    df = add_calculated_fields(df, source)

    # Ensure all mandatory columns are present
    for col in MANDATORY_COLUMNS:
        if col not in df.columns:
            df[col] = [] if col in LIST_FIELDS else ""  # type: ignore[assignment]

    logger.info("run_pipeline complete: output shape=%s", df.shape)
    return df
