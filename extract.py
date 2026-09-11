"""Extract financial line items from the 10-K filings' inline XBRL (iXBRL) tags.

Numbers come from structured XBRL facts (e.g. <ix:nonFraction name="us-gaap:Revenues">),
not from an LLM reading a table — the LLM is not involved in this file at all. Every
extracted value carries the exact source row/sentence it came from, so it can be checked
by hand against the HTML, and a confidence tier (confirmed / low_confidence) rather than
a single trusted number.

Run: python extract.py
"""
import csv
import re
import warnings
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from ingest import extract_fiscal_metadata

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

FILES_DIR = Path("files")
OUTPUT_CSV = Path("data/metrics.csv")
FILENAME_RE = re.compile(r"([A-Z]+)_10K_\d{4}\.html")

# Different filers use different standard XBRL concept names for the same line item.
# We try each candidate; within one filing only one of these will actually be present.
# "duration" concepts (income/cash flow statement) need a ~1-year period ending on the
# filing's period end date; "instant" concepts (balance sheet) need a single as-of date
# equal to the period end date.
METRIC_CONCEPTS = {
    "revenue": ("duration", ["us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"]),
    "cogs": ("duration", ["us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold"]),
    "net_income": ("duration", ["us-gaap:NetIncomeLoss"]),
    "total_equity": ("instant", ["us-gaap:StockholdersEquity"]),
    "cash_from_operations": ("duration", ["us-gaap:NetCashProvidedByUsedInOperatingActivities"]),
    # NVIDIA doesn't use the standard PP&E-only capex tag — its equivalent line,
    # "Purchases related to property and equipment and intangible assets", bundles in
    # intangible asset purchases too. Same underlying idea (cash spent on long-lived
    # assets) but a slightly broader scope than AMD/Intel's figure — noted for metrics.py.
    "capex": ("duration", ["us-gaap:PaymentsToAcquirePropertyPlantAndEquipment", "us-gaap:PaymentsToAcquireProductiveAssets"]),
}

MIN_FULL_YEAR_DAYS = 350
MAX_FULL_YEAR_DAYS = 380
DEBT_RECONCILE_TOLERANCE = 1_000_000  # $1M, to allow for immaterial rounding between tags


def parse_date(text: str) -> date | None:
    text = text.replace("\xa0", " ").strip()
    try:
        return datetime.strptime(text, "%B %d, %Y").date()
    except ValueError:
        return None


def parse_contexts(soup: BeautifulSoup) -> dict:
    """Map context id -> {type, start, end, has_segment}. "duration" contexts (start+end
    date) back income/cash-flow-statement facts; "instant" contexts (single as-of date)
    back balance-sheet facts. Contexts with a segment/member dimension are tagged as such
    so callers can exclude segment/geographic breakdowns and keep only the consolidated
    whole-company figure."""
    contexts = {}
    for ctx in soup.find_all(re.compile(r"^xbrli:context$", re.I)):
        ctx_id = ctx.get("id")
        if not ctx_id:
            continue
        has_segment = ctx.find(re.compile(r"^xbrli:segment$", re.I)) is not None
        instant_tag = ctx.find(re.compile(r"^xbrli:instant$", re.I))
        start_tag = ctx.find(re.compile(r"^xbrli:startdate$", re.I))
        end_tag = ctx.find(re.compile(r"^xbrli:enddate$", re.I))
        try:
            if instant_tag is not None:
                instant = datetime.strptime(instant_tag.get_text(strip=True), "%Y-%m-%d").date()
                contexts[ctx_id] = {"type": "instant", "start": None, "end": instant, "has_segment": has_segment}
            elif start_tag is not None and end_tag is not None:
                start = datetime.strptime(start_tag.get_text(strip=True), "%Y-%m-%d").date()
                end = datetime.strptime(end_tag.get_text(strip=True), "%Y-%m-%d").date()
                contexts[ctx_id] = {"type": "duration", "start": start, "end": end, "has_segment": has_segment}
        except ValueError:
            continue
    return contexts


def context_matches(ctx: dict | None, period_end: date, period_type: str) -> bool:
    if ctx is None or ctx["has_segment"] or ctx["type"] != period_type or ctx["end"] != period_end:
        return False
    if period_type == "duration":
        duration_days = (ctx["end"] - ctx["start"]).days
        return MIN_FULL_YEAR_DAYS <= duration_days <= MAX_FULL_YEAR_DAYS
    return True


def clean_numeric_text(raw: str) -> float | None:
    cleaned = raw.replace("\xa0", "").replace(",", "").replace("$", "").strip()
    if not cleaned or cleaned in {"-", "—"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def resolve_value(tag) -> float | None:
    number = clean_numeric_text(tag.get_text())
    if number is None:
        return None
    scale = tag.get("scale")
    if scale is not None:
        number *= 10 ** int(scale)
    if tag.get("sign") == "-":
        number *= -1
    return number


def get_source_quote(tag) -> str:
    """Return the exact visible row/sentence the value came from, for manual verification
    against the HTML. Prefer the enclosing table row (label stays attached to its value);
    fall back to the nearest paragraph-level block."""
    row = tag.find_parent("tr")
    if row is not None:
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            return " | ".join(cells)
    block = tag.find_parent(["p", "div"])
    if block is not None:
        text = block.get_text(" ", strip=True)
        return text[:300]
    return tag.get_text(" ", strip=True)


def find_candidates(soup: BeautifulSoup, contexts: dict, period_end: date,
                     concepts: list[str], period_type: str) -> list[tuple]:
    """Return (value, tag, is_table_sourced) for every tag matching one of the given
    concepts under a clean, non-segment context for the filing's own period."""
    candidates = []
    for tag in soup.find_all(re.compile(r"^ix:nonfraction$", re.I)):
        if tag.get("name") not in concepts:
            continue
        ctx = contexts.get(tag.get("contextref"))
        if not context_matches(ctx, period_end, period_type):
            continue
        value = resolve_value(tag)
        if value is None:
            continue
        is_table = tag.find_parent("tr") is not None
        candidates.append((value, tag, is_table))
    return candidates


def resolve_candidates(candidates: list[tuple]) -> dict:
    """Turn a candidate list into a value + confidence tier. The same fact is often tagged
    more than once (e.g. an exact figure in the financial statement table AND a rounded
    restatement in MD&A prose, like Intel's "$3.7 billion" for a $3,729M line) — that is
    NOT genuine ambiguity. We prefer table-sourced values (the financial statement itself)
    and only fall back to prose if no table match exists; only distinct VALUES within the
    preferred pool count as a real conflict worth flagging."""
    if not candidates:
        return {"value": None, "unit": None, "confidence": "not_found", "source_quote": None}

    table_candidates = [c for c in candidates if c[2]]
    pool = table_candidates if table_candidates else candidates
    distinct_values = {v for v, _, _ in pool}

    if len(distinct_values) > 1:
        value, tag, _ = pool[0]
        quotes = "; ".join(f"{v} [{get_source_quote(t)}]" for v, t, _ in pool[:3])
        return {"value": value, "unit": "USD", "confidence": "low_confidence",
                "source_quote": f"CONFLICTING VALUES FOUND: {quotes}"}

    value, tag, _ = pool[0]
    return {"value": value, "unit": "USD", "confidence": "confirmed", "source_quote": get_source_quote(tag)}


def extract_metric(soup: BeautifulSoup, contexts: dict, period_end: date,
                    concepts: list[str], period_type: str) -> dict:
    candidates = find_candidates(soup, contexts, period_end, concepts, period_type)
    return resolve_candidates(candidates)


def extract_total_debt(soup: BeautifulSoup, contexts: dict, period_end: date) -> dict:
    """No single XBRL concept means "total debt" the way us-gaap:NetIncomeLoss means net
    income. We reconcile it explicitly from balance-sheet components, in priority order:

    1. If the filer tags an explicit combined total (e.g. AMD's
       "us-gaap:DebtLongtermAndShorttermCombinedAmount", labeled "Total debt (net)" on its
       own balance sheet) — use it directly.
    2. Otherwise sum the current portion + noncurrent portion of long-term debt. If a
       filer (e.g. NVIDIA, Intel) also separately tags an aggregate "us-gaap:LongTermDebt"
       figure, that's used as an independent cross-check on the sum, not as the source of
       truth by itself — if it disagrees with the sum of parts, that disagreement is
       surfaced as low_confidence rather than picked silently.

    The exact components used are always recorded in the returned source_quote."""
    combined = extract_metric(soup, contexts, period_end,
                               ["us-gaap:DebtLongtermAndShorttermCombinedAmount"], "instant")
    if combined["confidence"] == "confirmed":
        return {**combined, "source_quote": f"EXPLICIT COMBINED TAG us-gaap:DebtLongtermAndShorttermCombinedAmount="
                                              f"{combined['value']:,.0f} ({combined['source_quote']})"}

    noncurrent = extract_metric(soup, contexts, period_end, ["us-gaap:LongTermDebtNoncurrent"], "instant")
    current = extract_metric(soup, contexts, period_end,
                              ["us-gaap:LongTermDebtCurrent", "us-gaap:DebtCurrent"], "instant")

    if noncurrent["confidence"] == "not_found":
        return {"value": None, "unit": None, "confidence": "not_found",
                "source_quote": "No us-gaap:LongTermDebtNoncurrent or combined debt tag found"}
    if noncurrent["confidence"] == "low_confidence" or current["confidence"] == "low_confidence":
        return {"value": None, "unit": "USD", "confidence": "low_confidence",
                "source_quote": f"AMBIGUOUS COMPONENTS — noncurrent: {noncurrent['source_quote']}; "
                                 f"current: {current['source_quote']}"}

    current_value = current["value"] if current["confidence"] == "confirmed" else 0.0
    current_desc = (f"us-gaap:LongTermDebtCurrent/DebtCurrent={current_value:,.0f} ({current['source_quote']})"
                     if current["confidence"] == "confirmed" else "no current-portion tag found, treated as $0")
    total = noncurrent["value"] + current_value
    components_desc = (f"SUMMED: us-gaap:LongTermDebtNoncurrent={noncurrent['value']:,.0f} "
                        f"({noncurrent['source_quote']}) + {current_desc} = {total:,.0f}")

    crosscheck = extract_metric(soup, contexts, period_end, ["us-gaap:LongTermDebt"], "instant")
    if crosscheck["confidence"] == "confirmed":
        if abs(crosscheck["value"] - total) <= DEBT_RECONCILE_TOLERANCE:
            components_desc += f"; cross-checked against us-gaap:LongTermDebt={crosscheck['value']:,.0f} ({crosscheck['source_quote']}) — MATCHES"
            return {"value": total, "unit": "USD", "confidence": "confirmed", "source_quote": components_desc}
        else:
            components_desc += f"; cross-check us-gaap:LongTermDebt={crosscheck['value']:,.0f} ({crosscheck['source_quote']}) DOES NOT MATCH sum"
            return {"value": total, "unit": "USD", "confidence": "low_confidence", "source_quote": components_desc}

    return {"value": total, "unit": "USD", "confidence": "confirmed", "source_quote": components_desc}


def process_filing(path: Path) -> list[dict]:
    ticker = FILENAME_RE.match(path.name).group(1)
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    meta = extract_fiscal_metadata(soup)
    period_end = parse_date(meta["period_end_date"])
    contexts = parse_contexts(soup)

    rows = []
    for metric_name, (period_type, concepts) in METRIC_CONCEPTS.items():
        result = extract_metric(soup, contexts, period_end, concepts, period_type) if period_end else {
            "value": None, "unit": None, "confidence": "not_found", "source_quote": "period end date unresolved"
        }
        rows.append({
            "ticker": ticker,
            "fiscal_year": meta["fiscal_year"],
            "period_end_date": meta["period_end_date"],
            "metric": metric_name,
            "value": result["value"],
            "unit": result["unit"],
            "confidence": result["confidence"],
            "source_quote": result["source_quote"],
            "source_file": path.name,
        })

    debt_result = extract_total_debt(soup, contexts, period_end) if period_end else {
        "value": None, "unit": None, "confidence": "not_found", "source_quote": "period end date unresolved"
    }
    rows.append({
        "ticker": ticker,
        "fiscal_year": meta["fiscal_year"],
        "period_end_date": meta["period_end_date"],
        "metric": "total_debt",
        "value": debt_result["value"],
        "unit": debt_result["unit"],
        "confidence": debt_result["confidence"],
        "source_quote": debt_result["source_quote"],
        "source_file": path.name,
    })
    return rows


def main():
    all_rows = []
    for path in sorted(FILES_DIR.glob("*.html")):
        rows = process_filing(path)
        all_rows.extend(rows)
        for r in rows:
            print(f"{r['ticker']:5s} {r['fiscal_year']:>6s} {r['metric']:12s} "
                  f"{r['confidence']:14s} {r['value']}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
