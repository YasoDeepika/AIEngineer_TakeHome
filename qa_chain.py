"""RAG question-answering over the 10-K filings, plus a deterministic route for questions
about derived metrics (margins, growth, FCF, D/E) that answers from data/derived_metrics.csv
instead of asking the LLM to recompute or recite a number from memory.

Three independent safeguards against hallucination, all shown/testable, not just asserted:
1. A similarity-distance ceiling — if the best retrieved chunk is too dissimilar, refuse
   before ever calling the LLM.
2. A keyword-overlap floor — catches the case a similarity score alone misses: a
   topically-nearby but non-answering chunk (e.g. asking about a CEO's favorite programming
   language pulls back a company-overview chunk that never addresses the actual question).
3. A post-generation numeric grounding check — every number in the LLM's answer must appear
   in the retrieved context, or the answer is discarded in favor of a refusal.

Citations are assembled from chunk metadata (ticker, fiscal_year, section) and substituted
into the LLM's [SOURCE N] placeholders after generation — the LLM never produces the
citation text itself, so it can't get a citation wrong.
"""
import re
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "sec_filings"
DERIVED_METRICS_CSV = Path("data/derived_metrics.csv")
FISCAL_ALIGNMENT_CSV = Path("data/fiscal_alignment.csv")

RETRIEVAL_K = 8  # bumped from 4 after a recall miss on "Did AMD acquire any companies in 2025?"
DISTANCE_CEILING = 0.75  # best chunk must be at least this similar (lower = more similar)
KEYWORD_OVERLAP_FLOOR = 0.34  # at least a third of the question's content words must appear
STOPWORDS = {
    "what", "when", "where", "which", "who", "how", "why", "is", "are", "was", "were", "the",
    "a", "an", "of", "in", "on", "for", "to", "and", "or", "did", "does", "do", "its", "their",
    "compare", "with", "between", "than", "that", "this", "have", "has", "had",
}

REFUSAL_TEXT = "I cannot answer this from the filings provided."

SYSTEM_PROMPT = """You are a financial analyst assistant. Answer ONLY using the excerpts \
below, taken from SEC 10-K filings. Do not use any outside knowledge.

Rules:
- Every factual claim must cite its source using the exact tag shown, e.g. [SOURCE 1].
- If the excerpts do not contain enough information to answer the question, respond with \
exactly this sentence and nothing else: "{refusal}"
- Never estimate, infer, or state a number that is not explicitly written in the excerpts.
""".format(refusal=REFUSAL_TEXT)

USER_TEMPLATE = """Excerpts:
{context}

Question: {question}

Answer using only the excerpts above. Cite sources as [SOURCE N]."""

EXPLANATORY_PHRASES = ["why", "caused", "drove", "driven by", "reason for", "explain", "factors"]


def is_explanatory(question: str) -> bool:
    """A question asking WHY a metric moved needs the narrative text (MD&A commentary),
    not just the metric's number — even though it names the metric, it should never be
    routed to the structured no-LLM path. Missing this caused two false passes in eval.py:
    "Why did NVIDIA's gross margin change...?" was routed to the structured path and
    returned only "gross_margin = 72.72%" with no explanation at all, and "What did NVIDIA
    say caused the increase in Data Center revenue?" was routed to structured because it
    contains the substring "revenue"."""
    q = question.lower()
    return any(phrase in q for phrase in EXPLANATORY_PHRASES)


TICKER_ALIASES = {
    "NVDA": ["nvda", "nvidia"],
    "AMD": ["amd", "advanced micro devices"],
    "INTC": ["intc", "intel"],
}
METRIC_ALIASES = {
    "revenue_growth_yoy": ["revenue growth", "growth rate", "yoy growth", "revenue increase", "revenue decline"],
    "gross_margin": ["gross margin"],
    "net_margin": ["net margin", "profit margin"],
    "debt_to_equity": ["debt to equity", "debt-to-equity", "d/e ratio", "leverage"],
    "free_cash_flow": ["free cash flow", "fcf"],
}

# Core financial figures we've already extracted and confirmed via XBRL. Any question asking
# for one of these should be answered from metrics.csv directly, never from the LLM reading
# retrieved text — this is what fixes the "Intel net loss" failure by construction: the LLM
# is never the source of a headline number we already have a verified value for. Checked only
# after METRIC_ALIASES (derived metrics) so "revenue growth" still routes to growth, not raw
# revenue, since "revenue" is a substring concern otherwise.
RAW_METRIC_ALIASES = {
    "revenue": ["revenue", "net revenue", "total revenue", "sales"],
    "net_income": ["net income", "net loss", "net profit", "profit or loss"],
    "cogs": ["cost of revenue", "cost of goods", "cogs"],
    "total_equity": ["stockholders equity", "shareholders equity", "total equity"],
    "total_debt": ["total debt", "how much debt", "debt level"],
    "capex": ["capital expenditure", "capex"],
    "cash_from_operations": ["cash from operations", "operating cash flow", "cash flow from operations"],
}


def _get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")


def _get_llm():
    return ChatOllama(model="llama3.2", temperature=0)


def _get_vectorstore():
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=_get_embeddings(), persist_directory=CHROMA_DIR)


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS}


def detect_tickers(question: str) -> list[str]:
    q = question.lower()
    return [ticker for ticker, aliases in TICKER_ALIASES.items() if any(a in q for a in aliases)]


def detect_metric(question: str) -> str | None:
    q = question.lower()
    for metric, aliases in METRIC_ALIASES.items():
        if any(a in q for a in aliases):
            return metric
    return None


def detect_raw_metric(question: str) -> str | None:
    q = question.lower().replace("'", "")  # "stockholders' equity" -> "stockholders equity"
    for metric, aliases in RAW_METRIC_ALIASES.items():
        if any(a in q for a in aliases):
            return metric
    return None


def detect_fiscal_years(question: str) -> list[int]:
    return [int(y) for y in re.findall(r"20\d{2}", question)]


# --- Structured path: answer directly from data/derived_metrics.csv, no LLM call ---

def answer_from_derived_metrics(question: str) -> dict | None:
    tickers = detect_tickers(question)
    metric = detect_metric(question)
    if not tickers or not metric:
        return None

    derived = pd.read_csv(DERIVED_METRICS_CSV)
    alignment = pd.read_csv(FISCAL_ALIGNMENT_CSV)
    years = detect_fiscal_years(question)

    def latest_row(ticker):
        rows = derived[(derived["ticker"] == ticker) & (derived["metric"] == metric)]
        if years:
            rows = rows[rows["fiscal_year"].isin(years)]
        if rows.empty:
            return None
        return rows.sort_values("fiscal_year").iloc[-1]

    picked = {t: latest_row(t) for t in tickers}
    if any(r is None for r in picked.values()):
        return None  # fall through to RAG rather than answer with a gap

    lines = []
    for ticker, row in picked.items():
        value = row["value"]
        if pd.isna(value):
            lines.append(f"{ticker} FY{row['fiscal_year']}: {metric} is not available — {row['inputs']}")
            continue
        pct_metrics = {"revenue_growth_yoy", "gross_margin", "net_margin"}
        formatted = f"{value:.2%}" if metric in pct_metrics else f"{value:,.4f}" if metric == "debt_to_equity" else f"${value:,.0f}"
        lines.append(f"{ticker} FY{row['fiscal_year']}: {metric} = {formatted}\n"
                      f"    Formula: {row['formula']}\n"
                      f"    Inputs: {row['inputs']}" + (f"\n    Caveat: {row['caveat']}" if row["caveat"] and pd.notna(row["caveat"]) else ""))

    # Cross-company comparison: surface the fiscal-year offset instead of comparing silently.
    offset_note = None
    if len(picked) > 1:
        # Exact (ticker, fiscal_year) pairs actually being compared — not every alignment
        # row whose year happens to match ANY picked company's year (that previously pulled
        # in NVDA's unrelated FY2025 row when comparing NVDA FY2026 against AMD FY2025).
        picked_pairs = {(ticker, int(row["fiscal_year"])) for ticker, row in picked.items()}
        align_rows = alignment[alignment.apply(lambda r: (r["ticker"], int(r["fiscal_year"])) in picked_pairs, axis=1)]
        offsets = align_rows[["ticker", "fiscal_year", "offset_days_from_group_anchor"]].to_dict("records")
        nonzero = [o for o in offsets if o["offset_days_from_group_anchor"] != 0]
        if nonzero:
            offset_note = ("Note on fiscal-year alignment: " +
                            "; ".join(f"{o['ticker']} FY{o['fiscal_year']} ends "
                                      f"{o['offset_days_from_group_anchor']} days after the other company's period end"
                                      for o in nonzero) +
                            " — these are the closest-overlapping periods, not the same calendar period.")

    answer = "\n\n".join(lines)
    if offset_note:
        answer += "\n\n" + offset_note

    return {"answer": answer, "source": "computed from data/derived_metrics.csv (no LLM call)", "refused": False}


def answer_from_raw_metrics(question: str) -> dict | None:
    tickers = detect_tickers(question)
    metric = detect_raw_metric(question)
    if not tickers or not metric:
        return None

    metrics = pd.read_csv(Path("data/metrics.csv"))
    alignment = pd.read_csv(FISCAL_ALIGNMENT_CSV)
    years = detect_fiscal_years(question)

    def latest_row(ticker):
        rows = metrics[(metrics["ticker"] == ticker) & (metrics["metric"] == metric)]
        if years:
            rows = rows[rows["fiscal_year"].isin(years)]
        if rows.empty:
            return None
        return rows.sort_values("fiscal_year").iloc[-1]

    picked = {t: latest_row(t) for t in tickers}
    if any(r is None for r in picked.values()):
        return None  # fall through to RAG rather than answer with a gap

    lines = []
    for ticker, row in picked.items():
        if row["confidence"] != "confirmed":
            lines.append(f"{ticker} FY{row['fiscal_year']}: {metric} is '{row['confidence']}', not confirmed — "
                         f"refusing to state a number for it. {row['source_quote']}")
            continue
        value = row["value"]
        # "net loss" reads more naturally than "net income of -$X" for a negative value.
        if metric == "net_income" and value < 0:
            label = f"net loss of ${abs(value):,.0f}"
        else:
            label = f"{metric.replace('_', ' ')} of ${value:,.0f}"
        lines.append(f"{ticker} FY{row['fiscal_year']}: {label}\n"
                      f"    Source: {row['source_quote']} (from {row['source_file']}, confidence: confirmed)")

    offset_note = None
    if len(picked) > 1:
        picked_pairs = {(ticker, int(row["fiscal_year"])) for ticker, row in picked.items()}
        align_rows = alignment[alignment.apply(lambda r: (r["ticker"], int(r["fiscal_year"])) in picked_pairs, axis=1)]
        nonzero = [o for o in align_rows.to_dict("records") if o["offset_days_from_group_anchor"] != 0]
        if nonzero:
            offset_note = ("Note on fiscal-year alignment: " +
                            "; ".join(f"{o['ticker']} FY{o['fiscal_year']} ends "
                                      f"{o['offset_days_from_group_anchor']} days after the other company's period end"
                                      for o in nonzero) +
                            " — these are the closest-overlapping periods, not the same calendar period.")

    answer = "\n\n".join(lines)
    if offset_note:
        answer += "\n\n" + offset_note

    return {"answer": answer, "source": "computed from data/metrics.csv (XBRL-extracted, no LLM call)", "refused": False}


# --- RAG path ---

def retrieve(vectorstore, question: str, tickers: list[str]):
    filter_arg = {"ticker": tickers[0]} if len(tickers) == 1 else None
    return vectorstore.similarity_search_with_score(question, k=RETRIEVAL_K, filter=filter_arg)


def check_keyword_overlap(question: str, context_text: str) -> float:
    q_words = _content_words(question)
    if not q_words:
        return 1.0
    context_lower = context_text.lower()
    hits = sum(1 for w in q_words if w in context_lower)
    return hits / len(q_words)


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d[\d,]*\.?\d*", text)


def numbers_grounded(answer: str, context_text: str) -> bool:
    context_numbers = {n.replace(",", "") for n in extract_numbers(context_text)}
    for n in extract_numbers(answer):
        normalized = n.replace(",", "")
        if len(normalized) <= 1:  # ignore stray single digits (e.g. "Item 7")
            continue
        if normalized not in context_numbers:
            return False
    return True


def answer_from_rag(question: str) -> dict:
    vectorstore = _get_vectorstore()
    tickers = detect_tickers(question)
    results = retrieve(vectorstore, question, tickers)

    if not results or results[0][1] > DISTANCE_CEILING:
        return {"answer": REFUSAL_TEXT, "source": f"refused: best distance "
                f"{results[0][1]:.3f} exceeds ceiling {DISTANCE_CEILING}" if results else "refused: no results",
                "refused": True}

    context_text = "\n".join(doc.page_content for doc, _ in results)
    overlap = check_keyword_overlap(question, context_text)
    if overlap < KEYWORD_OVERLAP_FLOOR:
        return {"answer": REFUSAL_TEXT,
                "source": f"refused: keyword overlap {overlap:.2f} below floor {KEYWORD_OVERLAP_FLOOR}",
                "refused": True}

    context_blocks = []
    citations = {}
    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        context_blocks.append(f"[SOURCE {i}]\n{doc.page_content}")
        citations[i] = f"[{meta.get('ticker')} 10-K FY{meta.get('fiscal_year')}, {meta.get('section')}]"
    context = "\n\n".join(context_blocks)

    llm = _get_llm()
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
    raw_answer = llm.invoke(prompt).content

    if REFUSAL_TEXT in raw_answer:
        return {"answer": REFUSAL_TEXT, "source": "LLM declined — context did not support an answer", "refused": True}

    if not numbers_grounded(raw_answer, context_text):
        return {"answer": REFUSAL_TEXT,
                "source": "refused post-generation: answer contained a number not present in retrieved context",
                "refused": True}

    final_answer = raw_answer
    for i, citation in citations.items():
        final_answer = final_answer.replace(f"[SOURCE {i}]", citation)

    return {"answer": final_answer, "source": f"retrieved {len(results)} chunks, best distance {results[0][1]:.3f}",
            "refused": False}


def answer_question(question: str) -> dict:
    if is_explanatory(question):
        return answer_from_rag(question)
    structured = answer_from_derived_metrics(question)
    if structured is not None:
        return structured
    raw = answer_from_raw_metrics(question)
    if raw is not None:
        return raw
    return answer_from_rag(question)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What was NVIDIA's revenue growth driven by?"
    result = answer_question(q)
    print("Q:", q)
    print("A:", result["answer"])
    print("(", result["source"], ")")
