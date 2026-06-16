"""
API retrieval module for PubMed (E-utilities) and OpenAlex REST API.

Each fetch_* function returns a raw pandas DataFrame whose column names
match the keys in the corresponding mapping dictionary in mapping_dicts.py.
No WoS-schema column names are used here; all renaming happens downstream
in standardizer.py.
"""

import logging
import re
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENALEX_BASE = "https://api.openalex.org/works"
OPENALEX_EMAIL = "bibliometrix-python@example.com"   # polite-pool identifier

_BACKOFF_BASE = 1.0      # seconds; doubled on each retry
_MAX_RETRIES = 5
_PUBMED_BATCH = 200      # efetch batch size
_OPENALEX_PAGE_SIZE = 200


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_with_backoff(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """
    HTTP GET with exponential-backoff retry on 429 / 5xx responses.

    Args:
        url: Endpoint URL.
        params: Query parameters dict.
        timeout: Per-request timeout in seconds.

    Returns:
        A successful requests.Response object.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    delay = _BACKOFF_BASE
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning("HTTP %s on attempt %d; retrying in %.1fs", resp.status_code, attempt, delay)
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(f"Request failed after {_MAX_RETRIES} attempts: {exc}") from exc
            logger.warning("Request error on attempt %d: %s; retrying in %.1fs", attempt, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

def _parse_medline_text(text: str) -> list[dict[str, str]]:
    """
    Parse a MEDLINE-format text block into a list of record dicts.

    Multi-value fields (AU, FAU, MH, OT, AD, PT) are joined with ";".
    Continuation lines (starting with 6 spaces) are appended to the
    current field value.

    Args:
        text: Raw MEDLINE text from efetch.

    Returns:
        List of dicts, one per PubMed record.
    """
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    current_key: str | None = None

    for line in text.splitlines():
        if line.strip() == "":
            if current:
                records.append(current)
                current = {}
                current_key = None
            continue

        if line.startswith("      "):
            # Continuation line — append to previous field
            if current_key and current_key in current:
                current[current_key] += " " + line.strip()
            continue

        match = re.match(r"^([A-Z]{2,4})\s*-\s*(.*)", line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()
            current_key = key
            if key in current:
                current[key] += ";" + value
            else:
                current[key] = value

    if current:
        records.append(current)

    return records


def _esearch_pmids(query: str, max_results: int) -> list[str]:
    """
    Search PubMed and return a list of PMIDs up to max_results.

    Args:
        query: PubMed search query string.
        max_results: Maximum number of PMIDs to retrieve.

    Returns:
        List of PMID strings.
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "usehistory": "y",
    }
    resp = _get_with_backoff(f"{PUBMED_BASE}/esearch.fcgi", params)
    data = resp.json()
    pmids: list[str] = data.get("esearchresult", {}).get("idlist", [])
    logger.info("esearch returned %d PMIDs for query: %s", len(pmids), query)
    return pmids


def _efetch_records(pmids: list[str]) -> list[dict[str, str]]:
    """
    Fetch full MEDLINE records for a list of PMIDs in batches.

    Args:
        pmids: List of PubMed IDs.

    Returns:
        List of parsed record dicts.
    """
    all_records: list[dict[str, str]] = []
    for start in range(0, len(pmids), _PUBMED_BATCH):
        batch = pmids[start: start + _PUBMED_BATCH]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "medline",
            "retmode": "text",
        }
        resp = _get_with_backoff(f"{PUBMED_BASE}/efetch.fcgi", params)
        records = _parse_medline_text(resp.text)
        all_records.extend(records)
        logger.info("Fetched %d records (batch %d-%d)", len(records), start, start + len(batch))
        if start + _PUBMED_BATCH < len(pmids):
            time.sleep(0.34)   # NCBI rate-limit: max 3 requests/second without API key
    return all_records


def fetch_pubmed(query: str, max_results: int = 100) -> pd.DataFrame:
    """
    Retrieve PubMed records via E-utilities and return a raw DataFrame.

    Column names match PUBMED_MAP keys from mapping_dicts.py.
    Multi-value fields are stored as ";"-joined strings — enforce_types
    will split them into list[str].

    Args:
        query: PubMed search query (supports full Entrez syntax).
        max_results: Maximum number of records to retrieve (default 100).

    Returns:
        Raw DataFrame with one row per article and PubMed field tags as columns.

    Raises:
        RuntimeError: If the API cannot be reached after retries.
        ValueError: If no results are found for the query.
    """
    logger.info("PubMed fetch: query=%r max_results=%d", query, max_results)
    pmids = _esearch_pmids(query, max_results)
    if not pmids:
        raise ValueError(f"No PubMed results for query: {query!r}")

    records = _efetch_records(pmids)
    df = pd.DataFrame(records)

    # Tag the source so the standardizer can identify it without ambiguity
    df["DB"] = "PUBMED"
    logger.info("fetch_pubmed complete: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """
    Reconstruct abstract text from OpenAlex abstract_inverted_index.

    The inverted index maps each word to a list of its positions in the
    abstract. Reconstruction sorts words by position to recover the original.

    Args:
        inverted_index: Dict mapping word -> [position, ...], or None.

    Returns:
        Reconstructed abstract string, or "" if index is missing/empty.
    """
    if not inverted_index:
        return ""
    position_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word[pos] = word
    return " ".join(position_word[i] for i in sorted(position_word))


def _flatten_openalex_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Flatten a single OpenAlex work JSON object into a flat dict.

    Produces column names that match OPENALEX_MAP keys in mapping_dicts.py.
    All multi-value fields are stored as lists at this stage; enforce_types
    will normalise them later.

    Args:
        record: A single OpenAlex work dict from the API response.

    Returns:
        Flat dict with one entry per column.
    """
    flat: dict[str, Any] = {}

    flat["display_name"] = record.get("title") or record.get("display_name") or ""
    flat["publication_year"] = str(record.get("publication_year") or "")
    flat["type"] = record.get("type") or ""
    flat["language"] = record.get("language") or ""
    flat["cited_by_count"] = record.get("cited_by_count") or 0

    # DOI — strip resolver prefix
    raw_doi = record.get("doi") or ""
    flat["doi"] = raw_doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

    # OpenAlex ID
    flat["openalex_id"] = record.get("id") or ""

    # PubMed ID
    ids = record.get("ids") or {}
    raw_pmid = ids.get("pmid") or ""
    flat["pmid"] = str(raw_pmid).replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip("/")

    # Authors
    authorships = record.get("authorships") or []
    author_names: list[str] = []
    author_full_names: list[str] = []
    affiliations: list[str] = []

    for a in authorships:
        author = a.get("author") or {}
        display = author.get("display_name") or ""
        author_names.append(display)
        author_full_names.append(display)   # OpenAlex provides full names only
        for inst in (a.get("institutions") or []):
            inst_name = inst.get("display_name") or ""
            if inst_name:
                affiliations.append(inst_name)

    flat["author_names"] = author_names
    flat["author_full_names"] = author_full_names
    flat["affiliations"] = affiliations
    flat["reprint_author"] = author_names[0] if author_names else ""

    # Journal / source
    primary = record.get("primary_location") or {}
    source = primary.get("source") or {}
    flat["source_title"] = source.get("display_name") or ""
    flat["source_abbr"] = source.get("abbreviated_title") or source.get("display_name") or ""

    # Bibliographic details
    biblio = record.get("biblio") or {}
    flat["volume"] = str(biblio.get("volume") or "")
    flat["issue"] = str(biblio.get("issue") or "")
    flat["first_page"] = str(biblio.get("first_page") or "")
    flat["last_page"] = str(biblio.get("last_page") or "")

    # Abstract
    flat["abstract"] = _reconstruct_abstract(record.get("abstract_inverted_index"))

    # Keywords (author keywords)
    kw_raw = record.get("keywords") or []
    flat["keywords"] = [k.get("keyword", "") for k in kw_raw if k.get("keyword")]

    # Concepts (index keywords)
    concepts_raw = record.get("concepts") or []
    flat["concepts"] = [c.get("display_name", "") for c in concepts_raw if c.get("display_name")]

    # Referenced works (as OpenAlex IDs — used as CR placeholders)
    flat["referenced_works"] = record.get("referenced_works") or []

    # DB tag added here for early source identification
    flat["DB"] = "OPENALEX"

    return flat


def fetch_openalex(query: str, max_results: int = 100) -> pd.DataFrame:
    """
    Retrieve works from OpenAlex REST API and return a raw DataFrame.

    Column names match OPENALEX_MAP keys from mapping_dicts.py.
    Multi-value fields are stored as Python lists at this stage.

    Args:
        query: Full-text search query string.
        max_results: Maximum number of records to retrieve (default 100).

    Returns:
        Raw DataFrame with one row per work and flattened OpenAlex fields.

    Raises:
        RuntimeError: If the API cannot be reached after retries.
        ValueError: If no results are found for the query.
    """
    logger.info("OpenAlex fetch: query=%r max_results=%d", query, max_results)
    records: list[dict[str, Any]] = []
    page = 1
    per_page = min(_OPENALEX_PAGE_SIZE, max_results)

    while len(records) < max_results:
        remaining = max_results - len(records)
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(per_page, remaining),
            "page": page,
            "mailto": OPENALEX_EMAIL,
        }
        resp = _get_with_backoff(OPENALEX_BASE, params)
        data = resp.json()
        results = data.get("results") or []

        if not results:
            break

        records.extend(results)
        logger.info("OpenAlex page %d: %d results (total so far: %d)", page, len(results), len(records))

        meta = data.get("meta") or {}
        count = meta.get("count") or 0
        if len(records) >= count or len(records) >= max_results:
            break

        page += 1
        time.sleep(0.1)   # polite-pool courtesy delay

    if not records:
        raise ValueError(f"No OpenAlex results for query: {query!r}")

    flat_records = [_flatten_openalex_record(r) for r in records[:max_results]]
    df = pd.DataFrame(flat_records)
    logger.info("fetch_openalex complete: %d rows, %d columns", len(df), len(df.columns))
    return df
