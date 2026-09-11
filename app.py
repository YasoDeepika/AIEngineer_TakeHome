"""Streamlit dashboard: Ask (RAG + structured Q&A), Compare (charts), Metrics detail (traceability).

Run: streamlit run app.py
"""
import pandas as pd
import streamlit as st

import qa_chain

st.set_page_config(page_title="SEC 10-K Analyst — NVDA / AMD / INTC", layout="wide")

METRICS_CSV = "data/metrics.csv"
DERIVED_CSV = "data/derived_metrics.csv"
ALIGNMENT_CSV = "data/fiscal_alignment.csv"

DERIVED_METRICS_TO_CHART = ["revenue_growth_yoy", "gross_margin", "net_margin", "debt_to_equity", "free_cash_flow"]
PERCENT_METRICS = {"revenue_growth_yoy", "gross_margin", "net_margin"}


@st.cache_data
def load_data():
    metrics = pd.read_csv(METRICS_CSV)
    derived = pd.read_csv(DERIVED_CSV)
    alignment = pd.read_csv(ALIGNMENT_CSV)
    return metrics, derived, alignment


def classify_path(result: dict) -> str:
    if result["refused"]:
        return "Refused"
    if "derived_metrics.csv" in result["source"]:
        return "Structured — derived_metrics.csv (no LLM call)"
    if "metrics.csv" in result["source"]:
        return "Structured — metrics.csv, XBRL-extracted (no LLM call)"
    return "RAG retrieval + LLM"


st.title("SEC 10-K Analyst — NVIDIA / AMD / Intel")

tab_ask, tab_compare, tab_detail = st.tabs(["Ask", "Compare", "Metrics detail"])

# --- Ask tab ---
with tab_ask:
    st.subheader("Ask a question about the filings")
    question = st.text_input("Question", placeholder="e.g. What was Intel's net loss in 2024?")
    if st.button("Ask") and question.strip():
        with st.spinner("Answering..."):
            result = qa_chain.answer_question(question)

        path_label = classify_path(result)
        if result["refused"]:
            st.warning(f"**Refused:** {result['answer']}")
        else:
            st.success(result["answer"])

        st.caption(f"**Path:** {path_label}")
        st.caption(f"**Detail:** {result['source']}")

# --- Compare tab ---
with tab_compare:
    metrics, derived, alignment = load_data()

    st.subheader("Fiscal-year alignment")
    st.caption(
        "NVIDIA's fiscal year ends in late January; AMD's and Intel's end in late December. "
        "Charts below use the aligned group (nearest-overlapping period), and NVIDIA's exact "
        "offset from the other two companies is shown here — it is never treated as the same "
        "calendar period."
    )
    st.dataframe(alignment, use_container_width=True, hide_index=True)

    def chart_metric(df: pd.DataFrame, metric: str, value_col: str, title: str, as_percent: bool = False):
        rows = df[df["metric"] == metric][["ticker", "fiscal_year", value_col]].copy()
        rows = rows.merge(alignment[["ticker", "fiscal_year", "aligned_group"]], on=["ticker", "fiscal_year"])
        if as_percent:
            rows[value_col] = rows[value_col] * 100
        pivot = rows.pivot(index="aligned_group", columns="ticker", values=value_col).sort_index()
        st.markdown(f"**{title}**")
        st.line_chart(pivot)

    st.subheader("Revenue ($)")
    chart_metric(metrics, "revenue", "value", "Revenue by aligned fiscal-year group")

    st.subheader("Derived metrics")
    chart_metric(derived, "revenue_growth_yoy", "value", "Revenue growth YoY (%)", as_percent=True)
    chart_metric(derived, "gross_margin", "value", "Gross margin (%)", as_percent=True)
    chart_metric(derived, "net_margin", "value", "Net margin (%)", as_percent=True)
    chart_metric(derived, "debt_to_equity", "value", "Debt-to-equity (ratio)")
    chart_metric(derived, "free_cash_flow", "value", "Free cash flow ($)")
    st.caption(
        "Free cash flow caveat: NVIDIA's capex figure (us-gaap:PaymentsToAcquireProductiveAssets) "
        "includes intangible-asset purchases, unlike AMD/Intel's PP&E-only capex tag — NVIDIA's FCF "
        "here is not scope-identical to the other two companies. See Metrics detail for the exact tags."
    )

# --- Metrics detail tab ---
with tab_detail:
    metrics, derived, alignment = load_data()

    st.subheader("Extracted figures (from XBRL)")
    st.caption("Every value's exact source row/sentence and confidence tier — trace any number back to its filing.")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("Derived metrics")
    st.caption("Formula and inputs for every computed metric — fully reproducible from the extracted figures above.")
    st.dataframe(derived, use_container_width=True, hide_index=True)

    st.subheader("Fiscal-year alignment")
    st.dataframe(alignment, use_container_width=True, hide_index=True)
