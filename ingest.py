"""Load SEC 10-K HTML filings, clean and chunk them, embed and persist to Chroma.

Run once (re-run to rebuild the index from scratch): python ingest.py
"""
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

FILES_DIR = Path("files")
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "sec_filings"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 100

FILENAME_RE = re.compile(r"([A-Z]+)_10K_\d{4}\.html")

# Full canonical statutory titles for standard 10-K items (Regulation S-K). We require the
# COMPLETE title (not a short keyword) immediately followed by a line break to identify a
# real heading. This matters because item titles get cross-referenced elsewhere in the
# document verbatim (e.g. Item 7's MD&A says "Refer to Item 1A. Risk Factors..." several
# times) — a cross-reference continues the sentence with a dash/comma after the title,
# while a real heading is its own DOM block and is followed by a newline before the next
# paragraph. Matching on the full title also rules out partial/incidental matches.
ITEM_TITLE_KEYWORDS = {
    "1": r"Business",
    "1A": r"Risk\s+Factors",
    "1B": r"Unresolved\s+Staff\s+Comments",
    "1C": r"Cybersecurity",
    "2": r"Properties",
    "3": r"Legal\s+Proceedings",
    "4": r"Mine\s+Safety\s+Disclosures",
    "5": r"Market\s+for\s+Registrant.s\s+Common\s+Equity,?\s+Related\s+Stockholder\s+Matters,?\s+and\s+Issuer\s+Purchases\s+of\s+Equity\s+Securities",
    "6": r"\[Reserved\]",
    "7": r"Management.s\s+Discussion\s+and\s+Analysis\s+of\s+Financial\s+Condition\s+and\s+Results\s+of\s+Operations",
    "7A": r"Quantitative\s+and\s+Qualitative\s+Disclosures?\s+[Aa]bout\s+Market\s+Risk",
    "8": r"Financial\s+Statements\s+and\s+Supplementary\s+Data",
    "9": r"Changes\s+in\s+and\s+Disagreements\s+[Ww]ith\s+Accountants\s+on\s+Accounting\s+and\s+Financial\s+Disclosure",
    "9A": r"Controls\s+and\s+Procedures",
    "9B": r"Other\s+Information",
    "9C": r"Disclosure\s+Regarding\s+Foreign\s+Jurisdictions\s+that\s+Prevent\s+Inspections",
    "10": r"Directors,?\s+Executive\s+Officers\s+and\s+Corporate\s+Governance",
    "11": r"Executive\s+Compensation",
    "12": r"Security\s+Ownership\s+of\s+Certain\s+Beneficial\s+Owners\s+and\s+Management\s+and\s+Related\s+Stockholder\s+Matters",
    "13": r"Certain\s+Relationships\s+and\s+Related\s+Transactions,?\s+and\s+Director\s+Independence",
    "14": r"Principal\s+Account(?:ant|ing)\s+Fees\s+and\s+Services",
    "15": r"Exhibits?(?:\s+and\s+Financial\s+Statement\s+Schedules)?",
    "16": r"Form\s+10-K\s+Summary",
}
ITEM_BOUNDARY_PATTERNS = {
    label: re.compile(rf"Item\s+{label}\.\s+{title}\s*\n", re.I)
    for label, title in ITEM_TITLE_KEYWORDS.items()
}

# Fallback for filers (e.g. Intel) that don't caption body sections with "Item N." at all —
# only their appendix cross-reference index uses that format. Their real headings are the
# bare title alone on its own line, sometimes wrapped in a single-cell styled table. This
# naturally excludes ToC rows too: a ToC row has a trailing page number on the same line
# ("Risk Factors | 31"), so nothing follows immediately by a newline the way a real heading
# does. Used only when the primary "Item N. <title>" pattern above finds zero matches.
ITEM_BARE_TITLE_PATTERNS = {
    label: re.compile(rf"\n(?:\[TABLE\]\s*\n)?({title})\s*(?:\n\s*\[/TABLE\])?\s*\n", re.I)
    for label, title in ITEM_TITLE_KEYWORDS.items()
}

# Some filers (Intel) use an abbreviated version of the statutory title as their real bare
# heading — e.g. "Management's Discussion and Analysis" without the "...of Financial
# Condition and Results of Operations" suffix. Tried only if neither pattern above matches.
ITEM_SHORT_TITLE_ALIASES = {
    "7": r"Management.s\s+Discussion\s+and\s+Analysis",
    "8": r"Financial\s+Statements\s+and\s+Supplemental(?:\s+Details|y\s+Data)",
}
ITEM_SHORT_TITLE_PATTERNS = {
    label: re.compile(rf"\n(?:\[TABLE\]\s*\n)?({title})\s*(?:\n\s*\[/TABLE\])?\s*\n", re.I)
    for label, title in ITEM_SHORT_TITLE_ALIASES.items()
}


def strip_non_content(soup: BeautifulSoup) -> None:
    """Remove scripts, styles, and the iXBRL header block (namespace/context definitions,
    never visible document content)."""
    for tag in soup.find_all(["script", "style", "head"]):
        tag.decompose()
    header = soup.find(re.compile(r"ix:header", re.I))
    if header:
        header.decompose()


def strip_hidden_tagging_spans(soup: BeautifulSoup) -> None:
    """Remove display:none elements. Must run AFTER extract_fiscal_metadata — the cover-page
    dei: facts (fiscal year, period end date) are frequently wrapped in a hidden span used
    only for XBRL tagging, not for visible rendering, so stripping them first would erase
    the very tags we need to read."""
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()


def table_to_text(table) -> str:
    """Flatten a table to pipe-delimited rows so a label stays attached to its values."""
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def inline_tables_as_text(soup: BeautifulSoup) -> None:
    """Replace each <table> with its flattened text, in place, so document order is preserved."""
    for table in soup.find_all("table"):
        table.replace_with("\n[TABLE]\n" + table_to_text(table) + "\n[/TABLE]\n")


def extract_fiscal_metadata(soup: BeautifulSoup) -> dict:
    """Read fiscal year and period end date from the filing's own XBRL tags (ground truth,
    not the filename — see NVDA_10K_2025/2026.html, whose filenames are swapped relative
    to the fiscal year they actually report). MUST run on the raw, unstripped soup: these
    dei: facts are sometimes tagged only inside a hidden ix:header/ix:hidden block with no
    visible on-page rendering, so stripping the header first would erase them."""
    fy_tag = soup.find(attrs={"name": re.compile(r"dei:DocumentFiscalYearFocus")})
    end_tag = soup.find(attrs={"name": re.compile(r"dei:DocumentPeriodEndDate")})
    fiscal_year = fy_tag.get_text(strip=True) if fy_tag else "UNKNOWN"
    period_end = end_tag.get_text(strip=True).replace("\xa0", " ") if end_tag else "UNKNOWN"
    return {"fiscal_year": fiscal_year, "period_end_date": period_end}


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Segment filing text by 'Item N. <canonical title>' headings. Each item's title
    typically appears twice (once in the table of contents, once at the real section) —
    we keep the last occurrence of each as the true boundary, since the ToC always comes
    first and running-header/cross-reference false positives are already excluded by
    requiring the exact statutory title."""
    # For each item, try three tiers in order of strictness: the exact "Item N. <title>"
    # caption, the bare title alone on its own line, then a shorter title alias some filers
    # use (e.g. Intel's abbreviated "Management's Discussion and Analysis"). Within
    # whichever tier first produces a match, take the LAST occurrence — the earlier
    # occurrence is virtually always the table of contents, since a genuine cross-reference
    # elsewhere restating the complete title immediately before a line break is rare.
    #
    # Note: we deliberately do NOT enforce that items appear in statutory order (1, 1A,
    # 1B, ... 16) here. That assumption holds for NVDA and AMD but not Intel, whose filing
    # explicitly states its "order and presentation of content... differs from the
    # traditional SEC Form 10-K format" — Intel's MD&A-equivalent content, for example,
    # appears earlier in the document than its Risk Factors section. Boundaries are
    # therefore sorted purely by position, which respects whatever true order a given
    # filer actually uses.
    boundaries = []
    for label in ITEM_TITLE_KEYWORDS:
        primary_hits = [m.start() for m in ITEM_BOUNDARY_PATTERNS[label].finditer(text)]
        bare_hits = [m.start(1) for m in ITEM_BARE_TITLE_PATTERNS[label].finditer(text)]
        short_pattern = ITEM_SHORT_TITLE_PATTERNS.get(label)
        short_hits = [m.start(1) for m in short_pattern.finditer(text)] if short_pattern else []
        candidates = primary_hits or bare_hits or short_hits
        if candidates:
            boundaries.append((max(candidates), label))
    boundaries.sort(key=lambda b: b[0])

    sections = []
    if not boundaries or boundaries[0][0] > 0:
        end = boundaries[0][0] if boundaries else len(text)
        sections.append(("Cover/Front Matter", text[:end]))
    for i, (start, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        sections.append((f"Item {label}", text[start:end]))
    return sections


def load_filing(path: Path) -> list[Document]:
    match = FILENAME_RE.match(path.name)
    ticker = match.group(1)

    html_text = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_text, "lxml")
    meta = extract_fiscal_metadata(soup)
    strip_non_content(soup)
    strip_hidden_tagging_spans(soup)
    inline_tables_as_text(soup)

    full_text = re.sub(r"[ \t]+", " ", soup.get_text("\n"))
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents = []
    for section_heading, section_text in split_into_sections(full_text):
        if not section_text.strip():
            continue
        for i, chunk in enumerate(splitter.split_text(section_text)):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "ticker": ticker,
                    "form_type": "10-K",
                    "fiscal_year": meta["fiscal_year"],
                    "period_end_date": meta["period_end_date"],
                    "section": section_heading,
                    "chunk_index": i,
                    "source_file": path.name,
                },
            ))
    return documents


def main():
    all_docs = []
    for path in sorted(FILES_DIR.glob("*.html")):
        docs = load_filing(path)
        print(f"{path.name}: {len(docs)} chunks")
        all_docs.extend(docs)
    print(f"Total chunks: {len(all_docs)}")

    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    existing_ids = vectorstore.get()["ids"]
    if existing_ids:
        vectorstore.delete(ids=existing_ids)

    for i in range(0, len(all_docs), EMBED_BATCH_SIZE):
        batch = all_docs[i:i + EMBED_BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Embedded {min(i + EMBED_BATCH_SIZE, len(all_docs))}/{len(all_docs)}")

    print("Done.")


if __name__ == "__main__":
    main()
