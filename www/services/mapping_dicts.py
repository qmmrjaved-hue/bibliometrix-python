"""
Column mapping dictionaries for each bibliographic source.

Each dictionary maps source-native field names to the WoS Field Tag schema.
Never reference column names directly elsewhere in the codebase — import and
use these dictionaries instead.
"""

from typing import Final

# ---------------------------------------------------------------------------
# PubMed — MEDLINE flat-tag format (E-utilities efetch rettype=medline)
# ---------------------------------------------------------------------------
# Multi-value fields (AU, FAU, MH, OT, AD, PT) are pre-joined with ";"
# by the API retriever before the DataFrame is built.
# AID contains entries like "10.1234/abc [doi]" — DOI extraction happens
# in enforce_types, not here.
# PG contains a page range like "100-110"; EP is derived in enforce_types.
# ---------------------------------------------------------------------------
PUBMED_MAP: Final[dict[str, str]] = {
    "PMID": "PMID",
    "TI":   "TI",
    "AU":   "AU",
    "FAU":  "AF",
    "AD":   "C1",
    "DP":   "PY",       # "2023 Jan 15" → 4-digit year extracted in enforce_types
    "TA":   "JI",
    "JT":   "SO",
    "PT":   "DT",
    "MH":   "ID",
    "OT":   "DE",
    "AB":   "AB",
    "VI":   "VL",
    "IP":   "IS",
    "PG":   "BP",       # full page string, EP derived in enforce_types
    "AID":  "DI",       # DOI candidate; cleaned in enforce_types
    "LA":   "LA",
    "GR":   "FU",
    "RP":   "RP",
    "EDAT": "UT",       # used as unique identifier fallback
}

# ---------------------------------------------------------------------------
# OpenAlex — flattened JSON field names produced by fetch_openalex()
# ---------------------------------------------------------------------------
# The API retriever flattens nested JSON into these scalar/list column names
# before the DataFrame is constructed. Nested extraction (authorships,
# biblio, primary_location, abstract_inverted_index) is done in the
# retriever, not here.
# ---------------------------------------------------------------------------
OPENALEX_MAP: Final[dict[str, str]] = {
    "display_name":       "TI",
    "author_names":       "AU",
    "author_full_names":  "AF",
    "affiliations":       "C1",
    "doi":                "DI",
    "publication_year":   "PY",
    "source_title":       "SO",
    "source_abbr":        "JI",
    "volume":             "VL",
    "issue":              "IS",
    "first_page":         "BP",
    "last_page":          "EP",
    "abstract":           "AB",
    "concepts":           "ID",
    "keywords":           "DE",
    "cited_by_count":     "TC",
    "type":               "DT",
    "language":           "LA",
    "referenced_works":   "CR",
    "openalex_id":        "UT",
    "pmid":               "PMID",
    "reprint_author":     "RP",
}

# ---------------------------------------------------------------------------
# PubMed file — MEDLINE plaintext export (uses LID instead of AID for DOI)
# ---------------------------------------------------------------------------
# IS in MEDLINE file format is ISSN (not issue number!). IP is issue number.
# LID contains the DOI with "[doi]" suffix, same cleaning as AID.
# Parsed by parse_pubmed_data() from parsers.py; multi-value fields joined ";".
# ---------------------------------------------------------------------------
PUBMED_FILE_MAP: Final[dict[str, str]] = {
    "PMID": "PMID",
    "TI":   "TI",
    "AU":   "AU",
    "FAU":  "AF",
    "AD":   "C1",
    "DP":   "PY",
    "TA":   "JI",
    "JT":   "SO",
    "PT":   "DT",
    "MH":   "ID",
    "OT":   "DE",
    "AB":   "AB",
    "VI":   "VL",
    "IP":   "IS",       # IP = Issue Number in MEDLINE
    "PG":   "BP",
    "LID":  "DI",       # DOI in file exports with "[doi]" suffix
    "IS":   "SN",       # IS = ISSN in MEDLINE file (not issue number!)
    "LA":   "LA",
    "GR":   "FU",
}

# ---------------------------------------------------------------------------
# Scopus CSV — column headers from the Scopus CSV export
# ---------------------------------------------------------------------------
SCOPUS_CSV_MAP: Final[dict[str, str]] = {
    "Title":                         "TI",
    "Authors":                       "AU",
    "Author full names":             "AF",
    "Abstract":                      "AB",
    "Source title":                  "SO",
    "Abbreviated Source Title":      "JI",
    "Year":                          "PY",
    "Volume":                        "VL",
    "Issue":                         "IS",
    "Page start":                    "BP",
    "Page end":                      "EP",
    "Cited by":                      "TC",
    "DOI":                           "DI",
    "References":                    "CR",
    "Affiliations":                  "C1",
    "Author Keywords":               "DE",
    "Index Keywords":                "ID",
    "ISSN":                          "SN",
    "Language of Original Document": "LA",
    "Document Type":                 "DT",
    "Correspondence Address":        "RP",
    "Funding Details":               "FU",
    "Funding Texts":                 "FX",
    "PubMed ID":                     "PMID",
    "Open Access":                   "OA",
    "EID":                           "UT",
    "Publisher":                     "PU",
    "Author(s) ID":                  "OI",
}

# ---------------------------------------------------------------------------
# Scopus BibTeX — fields returned by bibtexparser v1.x (all lowercase)
# ---------------------------------------------------------------------------
# author → "Surname, Name and Surname2, Name2" — split in enforce_types.
# note  → "Cited by: N; Export Date: ..." — TC integer extracted in enforce_types.
# url   → Scopus record URL; EID extracted via regex in enforce_types → UT.
# pages → "100 - 110" (with spaces) — EP derived in enforce_types.
# ---------------------------------------------------------------------------
SCOPUS_BIB_MAP: Final[dict[str, str]] = {
    "title":                   "TI",
    "author":                  "AU",
    "abstract":                "AB",
    "journal":                 "SO",
    "abbrev_source_title":     "JI",
    "year":                    "PY",
    "volume":                  "VL",
    "number":                  "IS",
    "pages":                   "BP",
    "doi":                     "DI",
    "affiliations":            "C1",
    "author_keywords":         "DE",
    "keywords":                "ID",
    "issn":                    "SN",
    "language":                "LA",
    "type":                    "DT",
    "correspondence_address":  "RP",
    "funding-acknowledgement": "FU",
    "funding-text":            "FX",
    "pmid":                    "PMID",
    "url":                     "UT",
    "note":                    "TC",
    "publisher":               "PU",
    "cited-references":        "CR",
}

# ---------------------------------------------------------------------------
# Web of Science TXT/CIW — parse_wos_data() already returns WoS tags.
# Only non-trivial renames are listed; all other fields keep their names.
# ---------------------------------------------------------------------------
WOS_TXT_MAP: Final[dict[str, str]] = {
    "PM": "PMID",   # PubMed ID field in WoS TXT
}

# ---------------------------------------------------------------------------
# Web of Science BibTeX — fields returned by bibtexparser v1.x (lowercase)
# ---------------------------------------------------------------------------
# The bibtexparser "ID" key holds the BibTeX entry key, which for WoS BibTeX
# exports is the WoS accession number (e.g. "WOS:001012311500028").
# author → "Surname, Name and Surname2, Name2" — split in enforce_types.
# Both "affiliation" and "affiliations" fields may be present; "affiliation"
# (per-author detail) is preferred for C1.
# ---------------------------------------------------------------------------
WOS_BIB_MAP: Final[dict[str, str]] = {
    "ID":                      "UT",
    "title":                   "TI",
    "author":                  "AU",
    "abstract":                "AB",
    "journal":                 "SO",
    "booktitle":               "SO",
    "journal-iso":             "JI",
    "year":                    "PY",
    "volume":                  "VL",
    "number":                  "IS",
    "pages":                   "BP",
    "doi":                     "DI",
    "affiliation":             "C1",
    "keywords":                "DE",
    "keywords-plus":           "ID",
    "issn":                    "SN",
    "language":                "LA",
    "type":                    "DT",
    "publisher":               "PU",
    "times-cited":             "TC",
    "research-areas":          "SC",
    "funding-acknowledgement": "FU",
    "funding-text":            "FX",
    "author-email":            "EM",
    "orcid-numbers":           "OI",
}

# ---------------------------------------------------------------------------
# Dimensions CSV/XLSX — column headers from Dimensions export (skiprows=1)
# ---------------------------------------------------------------------------
# Pagination: "100-110" — EP derived in enforce_types.
# Authors (Raw Affiliation): "Author (Affiliation); ..." — affiliation
# extraction handled in enforce_types for DIMENSIONS source.
# ---------------------------------------------------------------------------
DIMENSIONS_MAP: Final[dict[str, str]] = {
    "Title":                            "TI",
    "Authors":                          "AU",
    "Abstract":                         "AB",
    "Source title":                     "SO",
    "PubYear":                          "PY",
    "Volume":                           "VL",
    "Issue":                            "IS",
    "Pagination":                       "BP",
    "Times cited":                      "TC",
    "DOI":                              "DI",
    "Authors (Raw Affiliation)":        "C1",
    "Corresponding Authors":            "RP",
    "MeSH terms":                       "ID",
    "Fields of Research (ANZSRC 2020)": "SC",
    "Acknowledgements":                 "FX",
    "Funding":                          "FU",
    "PMID":                             "PMID",
    "Open Access":                      "OA",
    "Publication Type":                 "DT",
    "Publication ID":                   "UT",
}

# ---------------------------------------------------------------------------
# Lens.org CSV — column headers from the Lens CSV export
# ---------------------------------------------------------------------------
LENS_MAP: Final[dict[str, str]] = {
    "Title":               "TI",
    "Author/s":            "AU",
    "Abstract":            "AB",
    "Source Title":        "SO",
    "Publication Year":    "PY",
    "Volume":              "VL",
    "Issue Number":        "IS",
    "Start Page":          "BP",
    "End Page":            "EP",
    "Citing Works Count":  "TC",
    "DOI":                 "DI",
    "References":          "CR",
    "Keywords":            "DE",
    "MeSH Terms":          "ID",
    "ISSNs":               "SN",
    "Publication Type":    "DT",
    "PMID":                "PMID",
    "Lens ID":             "UT",
    "Publisher":           "PU",
    "Funding":             "FU",
    "Fields of Study":     "SC",
    "Open Access Colour":  "OA",
}

# ---------------------------------------------------------------------------
# Cochrane CDSR TXT — tags returned by parse_cochrane_data()
# ---------------------------------------------------------------------------
# "ID" in Cochrane format = Cochrane record ID (e.g., "CD006605").
# It is mapped to UT (unique identifier) to avoid collision with the WoS
# "ID" field (Index Keywords). The rename happens in rename_columns before
# enforce_types processes list fields.
# Multiple AU values are pre-joined with "; " by the parser.
# ---------------------------------------------------------------------------
COCHRANE_MAP: Final[dict[str, str]] = {
    "TI":  "TI",
    "AU":  "AU",
    "AB":  "AB",
    "SO":  "SO",
    "YR":  "PY",
    "SN":  "SN",
    "KY":  "DE",
    "DOI": "DI",
    "ID":  "UT",
    "PB":  "PU",
    "NO":  "IS",
}

# ---------------------------------------------------------------------------
# Source identifier → DB column value written to the output CSV
# ---------------------------------------------------------------------------
SOURCE_TO_DB: Final[dict[str, str]] = {
    "PUBMED":      "PUBMED",
    "OPENALEX":    "OPENALEX",
    "SCOPUS_CSV":  "SCOPUS",
    "SCOPUS_BIB":  "SCOPUS",
    "WOS_TXT":     "WOS",
    "WOS_BIB":     "WOS",
    "DIMENSIONS":  "DIMENSIONS",
    "LENS":        "LENS",
    "COCHRANE":    "COCHRANE",
    "PUBMED_FILE": "PUBMED",
}

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

MANDATORY_COLUMNS: Final[list[str]] = [
    "DB", "UT", "DI", "PMID", "TI", "SO", "JI", "PY", "DT", "LA",
    "TC", "AU", "AF", "C1", "RP", "CR", "DE", "ID", "AB", "VL",
    "IS", "BP", "EP", "SR",
]

# Fields that must be list[str] in the standardized DataFrame
LIST_FIELDS: Final[list[str]] = ["AU", "AF", "C1", "CR", "DE", "ID"]

# Fields that must be str ("" when missing)
SCALAR_FIELDS: Final[list[str]] = [
    "TI", "SO", "AB", "DI", "UT", "DT", "LA", "RP", "JI",
    "VL", "IS", "BP", "EP", "PMID", "DB", "SR", "SN",
]
