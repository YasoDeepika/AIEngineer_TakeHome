"""Compute derived metrics and the fiscal-year alignment table from data/metrics.csv.

Pure pandas — no LLM calls anywhere in this file. Every derived value shows its formula
and the exact inputs it was computed from; those inputs are traceable back to their XBRL
source quote via the same (ticker, fiscal_year, metric) key in data/metrics.csv. Only
confirmed inputs are used — if a required input isn't confirmed, the derived metric is
left unset rather than computed from a shaky number.

Run: python metrics.py
"""
from pathlib import Path

import pandas as pd

METRICS_CSV = Path("data/metrics.csv")
DERIVED_CSV = Path("data/derived_metrics.csv")
ALIGNMENT_CSV = Path("data/fiscal_alignment.csv")

NVDA_CAPEX_CAVEAT = (
    "NVIDIA's capex figure is us-gaap:PaymentsToAcquireProductiveAssets, labeled "
    "'Purchases related to property and equipment AND INTANGIBLE ASSETS' — a broader scope "
    "than AMD/Intel's PP&E-only capex tag. Any comparison involving NVDA capex (FCF, "
    "capex/revenue) is not scope-identical across companies."
)


def load_metrics() -> pd.DataFrame:
    df = pd.read_csv(METRICS_CSV, parse_dates=["period_end_date"])
    return df


def get_confirmed_value(df: pd.DataFrame, ticker: str, fiscal_year, metric: str):
    """Return (value, source_quote) for a confirmed metric, or (None, reason) if the input
    isn't available — callers must check for None before using the value."""
    match = df[(df["ticker"] == ticker) & (df["fiscal_year"] == fiscal_year) & (df["metric"] == metric)]
    if match.empty:
        return None, f"no {metric} row found for {ticker} FY{fiscal_year}"
    row = match.iloc[0]
    if row["confidence"] != "confirmed":
        return None, f"{metric} for {ticker} FY{fiscal_year} is '{row['confidence']}', not confirmed — refusing to derive from it"
    return row["value"], row["source_quote"]


def build_fiscal_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """NVDA's fiscal year ends in late January; AMD's and Intel's end in late December.
    Rather than compare NVDA FY2025 to AMD/Intel FY2025 as if they covered the same months
    (they don't — NVDA FY2025 ends Jan 26 2025, about a month after AMD/Intel FY2024 ends
    Dec 28 2024, and about 11 months before AMD/Intel FY2025 ends Dec 27 2025), each
    company's period end date is compared against every other company's, and matched to
    whichever is chronologically closest. The offset is always recorded, never hidden."""
    periods = df[["ticker", "fiscal_year", "period_end_date"]].drop_duplicates().reset_index(drop=True)

    # Anchor groups on AMD/Intel's calendar year (both use a Dec fiscal year end, so their
    # own fiscal_year label already matches within days of each other).
    anchors = periods[periods["ticker"].isin(["AMD", "INTC"])][["fiscal_year", "period_end_date"]].drop_duplicates()
    anchor_dates = dict(zip(anchors["fiscal_year"], anchors["period_end_date"]))

    rows = []
    for _, r in periods.iterrows():
        if r["ticker"] in ("AMD", "INTC"):
            group = r["fiscal_year"]
            offset_days = 0
        else:
            # NVDA: find whichever AMD/Intel calendar fiscal year end is closest in time.
            best_group, best_offset = None, None
            for group_year, anchor_date in anchor_dates.items():
                offset = (r["period_end_date"] - anchor_date).days
                if best_offset is None or abs(offset) < abs(best_offset):
                    best_group, best_offset = group_year, offset
            group, offset_days = best_group, best_offset
        rows.append({
            "ticker": r["ticker"],
            "fiscal_year": r["fiscal_year"],
            "period_end_date": r["period_end_date"].date(),
            "aligned_group": group,
            "offset_days_from_group_anchor": offset_days,
        })

    alignment = pd.DataFrame(rows).sort_values(["aligned_group", "ticker"]).reset_index(drop=True)
    return alignment


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    periods = df[["ticker", "fiscal_year", "period_end_date"]].drop_duplicates().sort_values(
        ["ticker", "period_end_date"])

    derived_rows = []

    def add_row(ticker, fiscal_year, metric, value, formula, inputs_desc, caveat=None):
        derived_rows.append({
            "ticker": ticker, "fiscal_year": fiscal_year, "metric": metric, "value": value,
            "formula": formula, "inputs": inputs_desc, "caveat": caveat or "",
        })

    for ticker, group in periods.groupby("ticker"):
        group = group.sort_values("period_end_date").reset_index(drop=True)
        prev_fiscal_year = None
        for _, row in group.iterrows():
            fy = row["fiscal_year"]

            revenue, revenue_src = get_confirmed_value(df, ticker, fy, "revenue")
            cogs, cogs_src = get_confirmed_value(df, ticker, fy, "cogs")
            net_income, ni_src = get_confirmed_value(df, ticker, fy, "net_income")
            total_debt, debt_src = get_confirmed_value(df, ticker, fy, "total_debt")
            total_equity, equity_src = get_confirmed_value(df, ticker, fy, "total_equity")
            cash_ops, cash_src = get_confirmed_value(df, ticker, fy, "cash_from_operations")
            capex, capex_src = get_confirmed_value(df, ticker, fy, "capex")

            # Revenue growth YoY — needs the immediately preceding period for this ticker.
            if prev_fiscal_year is None:
                add_row(ticker, fy, "revenue_growth_yoy", None,
                        "(revenue - prev_revenue) / prev_revenue",
                        "no prior period for this ticker in the corpus — cannot compute")
            elif revenue is None:
                add_row(ticker, fy, "revenue_growth_yoy", None,
                        "(revenue - prev_revenue) / prev_revenue", revenue_src)
            else:
                prev_revenue, prev_revenue_src = get_confirmed_value(df, ticker, prev_fiscal_year, "revenue")
                if prev_revenue is None:
                    add_row(ticker, fy, "revenue_growth_yoy", None,
                            "(revenue - prev_revenue) / prev_revenue", prev_revenue_src)
                else:
                    growth = (revenue - prev_revenue) / prev_revenue
                    add_row(ticker, fy, "revenue_growth_yoy", growth,
                            "(revenue - prev_revenue) / prev_revenue",
                            f"revenue={revenue:,.0f} ({ticker} FY{fy}, see metrics.csv); "
                            f"prev_revenue={prev_revenue:,.0f} ({ticker} FY{prev_fiscal_year}, see metrics.csv)")

            # Gross margin
            if revenue is None or cogs is None:
                reason = revenue_src if revenue is None else cogs_src
                add_row(ticker, fy, "gross_margin", None, "(revenue - cogs) / revenue", reason)
            else:
                gm = (revenue - cogs) / revenue
                add_row(ticker, fy, "gross_margin", gm, "(revenue - cogs) / revenue",
                        f"revenue={revenue:,.0f}; cogs={cogs:,.0f} ({ticker} FY{fy}, see metrics.csv)")

            # Net margin
            if revenue is None or net_income is None:
                reason = revenue_src if revenue is None else ni_src
                add_row(ticker, fy, "net_margin", None, "net_income / revenue", reason)
            else:
                nm = net_income / revenue
                add_row(ticker, fy, "net_margin", nm, "net_income / revenue",
                        f"net_income={net_income:,.0f}; revenue={revenue:,.0f} ({ticker} FY{fy}, see metrics.csv)")

            # Debt-to-equity
            if total_debt is None or total_equity is None:
                reason = debt_src if total_debt is None else equity_src
                add_row(ticker, fy, "debt_to_equity", None, "total_debt / total_equity", reason)
            else:
                dte = total_debt / total_equity
                add_row(ticker, fy, "debt_to_equity", dte, "total_debt / total_equity",
                        f"total_debt={total_debt:,.0f}; total_equity={total_equity:,.0f} ({ticker} FY{fy}, see metrics.csv)")

            # Free cash flow
            if cash_ops is None or capex is None:
                reason = cash_src if cash_ops is None else capex_src
                add_row(ticker, fy, "free_cash_flow", None, "cash_from_operations - capex", reason)
            else:
                fcf = cash_ops - capex
                caveat = NVDA_CAPEX_CAVEAT if ticker == "NVDA" else None
                add_row(ticker, fy, "free_cash_flow", fcf, "cash_from_operations - capex",
                        f"cash_from_operations={cash_ops:,.0f}; capex={capex:,.0f} ({ticker} FY{fy}, see metrics.csv)",
                        caveat)

            prev_fiscal_year = fy

    return pd.DataFrame(derived_rows)


def main():
    df = load_metrics()

    alignment = build_fiscal_alignment(df)
    ALIGNMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(ALIGNMENT_CSV, index=False)
    print("=== Fiscal-year alignment ===")
    print(alignment.to_string(index=False))

    derived = compute_derived_metrics(df)
    derived.to_csv(DERIVED_CSV, index=False)
    print("\n=== Derived metrics ===")
    for _, r in derived.iterrows():
        val = f"{r['value']:.4f}" if pd.notna(r["value"]) else "N/A"
        print(f"{r['ticker']:5s} FY{str(r['fiscal_year']):>5s} {r['metric']:20s} {val:>10s}  {r['formula']}")

    print(f"\nWrote {ALIGNMENT_CSV} and {DERIVED_CSV}")


if __name__ == "__main__":
    main()
