# Writeup notes — collected during build, for WRITEUP.md

Raw material for the "show one failure you found and diagnosed" and architecture-decisions
sections of WRITEUP.md. Written immediately after each finding so the account stays accurate,
not reconstructed from memory later.

## Bug 1: fiscal_year read as "UNKNOWN" for every chunk

**What happened:** `ingest.py`'s first pass extracted `dei:DocumentFiscalYearFocus` and
`dei:DocumentPeriodEndDate` from the same cleaned soup, after stripping `<script>`, `<style>`,
`<head>`, and the `ix:header` block (all assumed to be non-content noise). `period_end_date`
came back correct; `fiscal_year` came back `UNKNOWN` for all 9 filings.

**Diagnosis:** These 10-Ks are inline XBRL (iXBRL) documents — every financial fact is tagged
inline with `<ix:nonFraction>`/`<ix:nonNumeric>` elements carrying a `name` attribute like
`dei:DocumentFiscalYearFocus`. Checking the tag's ancestor chain showed it sits inside
`ix:header > ix:hidden` — a block used purely for machine-readable tagging with no visible
on-page rendering. `DocumentPeriodEndDate` happened to *also* be rendered visibly elsewhere on
the cover page, which is why only one of the two fields broke — the other tag has no visible
duplicate, so stripping `ix:header` erased the only copy.

**Fix:** Extract fiscal metadata from the fully raw, unstripped soup, before any cleaning runs.
Cleaning for chunking/embedding purposes happens only after metadata has already been read out.

**Why this is worth keeping in the writeup:** it's a good concrete example of a wrong-number
failure mode that had nothing to do with the LLM — it was a document-parsing ordering bug, and
it silently produced a wrong value (`UNKNOWN`) rather than an error, which is exactly the kind
of thing a confidence/verification layer needs to catch rather than a human noticing later.

## Bug 2: chunks from NVIDIA's real MD&A section were mislabeled `section: Item 1A`

**What happened:** Section boundaries were originally detected by regex-matching `Item N.`
followed by a title, then taking the *last* occurrence of each item number as the true
boundary (reasoning: the table of contents mentions each item first, the real section
second). Verification retrieval for "what factors drove NVIDIA's gross margin change"
returned genuinely relevant MD&A content, but tagged `section: Item 1A` (Risk Factors), not
`Item 7`.

**Diagnosis:** Printed every match position for `Item 1A` and found 8 occurrences, not 2.
NVIDIA's own Item 7 (MD&A) text contains multiple sentences like "Refer to Item 1A. Risk
Factors — Risks Related to Regulatory, Legal..." — cross-references to earlier items,
restating the item number and title verbatim, appearing *after* Item 7's real heading. "Last
occurrence" picked one of these cross-references instead of the real Item 1A heading, which
actually occurs earlier in the document. Cross-references were found on both sides of the
real heading (some even inside Item 1's own body, before Item 1A starts), so neither
"first occurrence" nor "last occurrence" alone is a safe heuristic.

**Fix:** Require the *complete* canonical statutory title (e.g., the full "Management's
Discussion and Analysis of Financial Condition and Results of Operations", not just "Item 7"),
anchored immediately before a line break. A cross-reference continues mid-sentence with a dash
or comma after the title; a real heading is its own DOM block and is followed by a newline
before the next paragraph starts. This discriminator is structural (DOM block boundary), not
just textual, which is why it holds up where pure text-matching didn't.

**Why this is worth keeping in the writeup:** direct evidence for "would you trust this in
front of an executive" — an unverified system would have silently cited Risk Factors language
as the explanation for a margin move, which is a materially different (and wrong) claim.

## Known limitation: Intel's 10-Ks don't follow standard Item-by-item structure

**What happened:** After fixing the above, section detection worked perfectly for NVIDIA and
AMD (24/24 statutory items correctly found and ordered in every filing) but found only 9-10 of
24 for Intel, and in a scrambled, non-statutory order.

**Diagnosis:** Intel's own filing text states outright: *"The order and presentation of
content in our Form 10-K differs from the traditional SEC Form 10-K format. Our format is
designed to improve readability and better present how we organize and manage our business."*
Concretely: (1) Intel's real section headings often use abbreviated titles instead of the full
statutory wording (e.g. "Management's Discussion and Analysis" without the "...of Financial
Condition and Results of Operations" suffix); (2) the standard SEC Item captions ("Item 1.
Business", etc.) appear *only* in an appendix cross-reference index near the end of the
document, not at the real section headings; (3) Intel's sections don't even appear in
statutory order in the document body — its MD&A-equivalent content appears earlier than its
Risk Factors section, for example. A monotonic "items must appear in 1, 1A, 1B... order"
constraint that was added to fix cross-reference false positives elsewhere actively broke
Intel further, since that assumption is simply false for this filer.

**Resolution:** Added a three-tier fallback per item (exact statutory caption → bare title
alone on its own line → known short alias) and stopped enforcing statutory ordering — boundaries
are sorted purely by where they're found in the document, which respects each filer's actual
structure instead of assuming one convention. This recovered Item 7 (MD&A) and Item 8
(Financial Statements) for Intel — the two sections that matter most for metrics and
quant-to-narrative linkage. Item 1 (Business) remains unrecovered for Intel: its business
description is genuinely dispersed across several company-specific thematic headings
("Fundamentals of Our Business", segment-by-segment "Market and Business Overview" sections)
with no single heading to anchor on. Rather than building Intel-specific parsing to chase this
one item, it was left as a known, disclosed gap — Intel chunks that would ideally cite "Item 1"
instead cite the coarser "Cover/Front Matter" bucket.

**Why this is worth keeping in the writeup:** this is the most honest answer to "handle the
messy reality of financial data" — different filers structure the same legally-mandated
document differently, and a system that assumes one convention will silently degrade for
filers that don't follow it. Disclosing the gap (9-10/24 sections for Intel vs. 24/24 for
NVDA/AMD, but including the two sections that matter most) is more useful than either hiding it
or spending unbounded time chasing full parity for a company whose own filing explains why
parity isn't the right goal.

---

# Session Log

Dated entries appended after every build step from here on. Each entry records what was
built, what was verified, and any bugs found. Attribution is marked explicitly:
**Automated (Claude)** = a script/tool ran and produced output I inspected; **Hand-verified
(user)** = the user independently opened the source HTML and confirmed a value or claim
themselves. Nothing here is invented — anything not actually run or checked is not listed
as verified.

## 2026-09-11 — Environment setup, ingest.py, extract.py (revenue + net income)

### Architecture: fully local, no API keys
- LLM: `ChatOllama` running `llama3.2` (3B, ~2GB) via a local Ollama service (installed this
  session via `winget install Ollama.Ollama`).
- Embeddings: `OllamaEmbeddings` running `nomic-embed-text` (274MB), also local.
- Vector store: `Chroma`, persisted to `data/chroma/` on disk — no hosted vector DB.
- Orchestration/retrieval: LangChain (`langchain`, `langchain-ollama`, `langchain-chroma`,
  `langchain-text-splitters`).
- UI: Streamlit (not yet built as of this entry — `ingest.py` and `extract.py` only so far).
- Why: constraint from the assignment brief and the user — no API keys, no paid services,
  small local models. Every model call in the system is a local HTTP call to `localhost:11434`
  (Ollama's default port); nothing leaves the machine.
- **Automated (Claude):** verified the local stack actually works before building anything —
  a real `llama3.2` generation and a real `nomic-embed-text` embedding call (768-dim vector
  returned), both shown to the user, before writing any application code.

### File structure so far
- `files/` — the 9 source 10-K HTML filings (user-provided, downloaded from SEC EDGAR):
  NVDA_10K_{2024,2025,2026}.html, AMD_10K_{2023,2024,2025}.html, INTC_10K_{2023,2024,2025}.html.
- `ingest.py` — cleans each filing's HTML (strips scripts/styles/hidden iXBRL metadata),
  segments it into SEC Item sections, chunks the text, embeds each chunk with
  `nomic-embed-text`, and persists everything to Chroma with metadata (`ticker`, `form_type`,
  `fiscal_year`, `period_end_date`, `section`, `source_file`). Run once (or re-run to rebuild).
- `extract.py` — a separate, deterministic pass that reads specific financial line items
  directly from each filing's inline XBRL tags (not from chat-style retrieval, not from an
  LLM reading rendered tables). Writes `data/metrics.csv` with one row per (ticker, fiscal
  year, metric), each row carrying its exact source quote and a confidence tier.
- `data/chroma/` — persisted vector store (gitignored, rebuildable from `files/` by re-running
  `ingest.py`).
- `data/metrics.csv` — extracted financial line items, machine- and human-readable.
- `writeup_notes.md` — this file: a running, dated build log kept as raw material for the
  final `WRITEUP.md`.
- Not yet built: `metrics.py` (derived metrics + fiscal-year alignment + conflict detection),
  `qa_chain.py` (RAG chat with citations + refusal), `app.py` (Streamlit dashboard),
  `eval.py` (labeled question set + scoring), `README.md`, `WRITEUP.md`.

### Key decision: pull hard numbers from XBRL tags, not from an LLM reading tables
These 10-Ks are inline XBRL (iXBRL) documents — every financial fact in the rendered tables
is *also* wrapped in a machine-readable tag, e.g. `<ix:nonFraction name="us-gaap:Revenues"
contextref="c-1" scale="6">215,938</ix:nonFraction>`. `extract.py` reads these tags directly:
finds the tag for a given concept (e.g. `us-gaap:NetIncomeLoss`), resolves its `contextref`
against the document's `<xbrli:context>` definitions to confirm it covers the filing's own
primary fiscal year (not a prior-year comparative or a segment breakdown), and applies the
`scale`/`sign` attributes to get the real number. No LLM call is involved in producing any
number in `metrics.csv`. This matters because the user's stated concern going in was that a
small local model (llama3.2, 3B) misreading a financial table would silently produce a wrong
number with no way to catch it — routing hard numbers through the filing's own structured
tags instead of LLM table-reading removes that failure mode entirely for anything with a
findable XBRL tag. The LLM's role in this system is reasoning over already-extracted numbers
and answering free-form questions with retrieval, never being the source of a number that
ends up in the metrics table.

### HTML chunking and citation without page numbers
`ingest.py` strips non-visible content (scripts, styles, the hidden `ix:header` block used
for XBRL tagging), flattens `<table>` elements to pipe-delimited rows (so a label stays
attached to its values instead of being split by generic character-count chunking), then
segments the remaining text by SEC Item headings before running
`RecursiveCharacterTextSplitter` (1000 chars, 150 overlap) within each section. Since HTML has
no page numbers, each chunk's citation metadata is `{ticker, form_type, fiscal_year,
period_end_date, section, source_file}` — e.g. "NVDA 10-K FY2025, Item 7" — assembled
programmatically from the chunk's own metadata, never transcribed by the LLM, so a citation
can't drift from its real source.

### Fiscal-year mismatch: surfaced, not resolved
NVIDIA's fiscal year ends in late January; AMD's and Intel's end in late December. Rather than
force NVDA's "FY2025" to line up with AMD/Intel's "FY2025" as if they were the same period
(they're offset by about a month), the system records each filing's exact `period_end_date`
from its own XBRL tags and treats alignment as something to display explicitly (e.g. a
~1-month-offset annotation when comparing across companies), not something to silently paper
over. This is the same principle applied to conflicting/restated figures generally: surface
the discrepancy, don't pick one value quietly.

### Two bugs found and fixed in ingest.py (full diagnosis above in this file)
Both found and fixed by **Automated (Claude)** verification — i.e., by actually running
retrieval queries and inspecting real output, not by reasoning about the code in the abstract:
1. `fiscal_year` metadata read as `"UNKNOWN"` for every chunk — the `dei:DocumentFiscalYearFocus`
   XBRL tag lives inside a hidden `ix:header/ix:hidden` block with no visible on-page
   rendering, and the original cleaning step stripped it before it was read. Fixed by
   extracting fiscal metadata from the raw, unstripped soup first.
2. Chunks containing NVIDIA's real MD&A commentary (about gross margin) were mislabeled
   `section: Item 1A` (Risk Factors) — caused by cross-references inside Item 7's own text
   ("Refer to Item 1A. Risk Factors...") matching the section-boundary regex. Fixed by
   requiring the complete canonical statutory title anchored immediately before a line break,
   which distinguishes a real heading (its own DOM block) from a cross-reference (continues
   mid-sentence with a dash/comma).

### Filename vs. true fiscal year mismatch
`NVDA_10K_2025.html`'s filename suggests fiscal year 2025, but it is actually NVIDIA's FY2026
10-K (period ended January 25, 2026); conversely `NVDA_10K_2026.html` is actually the FY2025
report (period ended January 26, 2025) — the two filenames are swapped relative to the fiscal
year they actually report. The system never trusts filenames for this: `fiscal_year` and
`period_end_date` are always read from the filing's own `dei:DocumentFiscalYearFocus` /
`dei:DocumentPeriodEndDate` XBRL tags, so this is self-correcting regardless of filename.
**Hand-verified (user):** confirmed independently against the filing's own "for the fiscal
year ended" cover-page line and the income statement's "Year Ended Jan 25, 2026" column
header — both matched the XBRL-derived FY2026 finding.

### Intel structural discovery (full diagnosis above in this file)
**Automated (Claude)** finding: Intel's 10-K explicitly restructures away from the standard
Item-by-item SEC format (the filing states this itself). Section detection reaches 24/24
statutory items for every NVDA and AMD filing, but only 9-10/24 for Intel — critically still
including Item 7 (MD&A) and Item 8 (Financial Statements), the two sections that matter most
for metrics and quant-to-narrative linkage. Item 1 (Business) is not recoverable for Intel via
text heuristics since its business description is genuinely dispersed across several
company-specific thematic headings with no single anchor. Disclosed as a known limitation
rather than chased further, per user's explicit instruction not to over-engineer this.

### Verification performed
- **Automated (Claude):** `ingest.py` produced 5,863 chunks across all 9 filings; retrieval
  spot-checks run for both an NVDA query ("what factors drove NVIDIA gross margin change" —
  returned correct `section: Item 7` chunks) and an Intel query ("why did Intel revenue
  decline" — returned a chunk stating "Our 2025 revenue was $52.9 billion, down $248 million
  from 2024," correctly tagged `fiscal_year: 2025, section: Item 7`).
- **Automated (Claude):** verified the $52.9B Intel quote appears verbatim in the raw
  `INTC_10K_2025.html` source (exact string match, not paraphrased).
- **Automated (Claude):** `extract.py` pulled revenue and net income for all 9 filings
  directly from XBRL tags (`us-gaap:Revenues` or `us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`
  for revenue depending on filer; `us-gaap:NetIncomeLoss` for net income, consistent across
  all three companies). All 18 values came back `confidence: confirmed` (single unambiguous
  XBRL tag per value, matched to the filing's own primary-year context, no conflicting
  duplicate-tag values found).
- **Hand-verified (user):** four of the extracted values checked directly against the filing
  HTML — NVDA FY2026 revenue ($215,938M), NVDA FY2025 revenue ($130,497M), NVDA FY2024 revenue
  ($60,922M), and Intel FY2024 net loss (-$18,756M) — all matched.

## 2026-09-11 — extract.py: total_equity, cash_from_operations, capex, total_debt

### What was built
Extended `extract.py` to handle balance-sheet ("instant") XBRL concepts, not just
income-statement ("duration") ones, and added four more metrics:
- `total_equity` — `us-gaap:StockholdersEquity`, a single consistent tag across all three
  companies.
- `cash_from_operations` — `us-gaap:NetCashProvidedByUsedInOperatingActivities`, also
  consistent across all three.
- `capex` — `us-gaap:PaymentsToAcquirePropertyPlantAndEquipment` for AMD/Intel. NVIDIA does
  not use this tag at all; its equivalent line is `us-gaap:PaymentsToAcquireProductiveAssets`,
  labeled "Purchases related to property and equipment **and intangible assets**" — a
  slightly broader scope (bundles in intangible-asset purchases) than AMD/Intel's PP&E-only
  figure. Recorded as a comparability caveat for `metrics.py`, not hidden.
- `total_debt` — no single XBRL concept means this the way `NetIncomeLoss` means net income.
  Implemented a dedicated reconciliation (`extract_total_debt`) tried in priority order:
  (1) an explicit filer-tagged combined total if one exists (AMD tags
  `us-gaap:DebtLongtermAndShorttermCombinedAmount`, literally labeled "Total debt (net)" on
  its own balance sheet — used directly); (2) otherwise sum
  `us-gaap:LongTermDebtNoncurrent` + `us-gaap:LongTermDebtCurrent`/`DebtCurrent`, then
  cross-check the sum against a separately-tagged `us-gaap:LongTermDebt` aggregate where one
  exists (NVIDIA, Intel) — if the cross-check disagrees with the sum, that disagreement is
  what gets surfaced as `low_confidence`, not picked silently. Every `total_debt` row's
  `source_quote` spells out exactly which components were summed and their individual source
  rows, not just the final number.

### A second table-vs-prose duplicate-tagging pattern, generalized
While investigating Intel's debt tags, found the same underlying fact tagged twice with
different precision: the balance sheet's exact "Short-term debt | 3,729" (from a real
`<tr>`) and an MD&A sentence rounding the same fact to "$3.7 billion" (from a `<p>`, not a
table). Naively treating these as two candidate values would have produced a false
`low_confidence` flag on a metric that isn't actually ambiguous. Generalized the fix beyond
this one case: `resolve_candidates()` now prefers table-sourced tag instances over
prose-sourced ones whenever both exist for the same concept, and only treats genuinely
different values *within* the preferred pool as a real conflict.

### Verification performed
- **Automated (Claude):** ran `extract.py` for all 6 metrics across all 9 filings (54 rows
  total). All 54 came back `confidence: confirmed`, including all 9 `total_debt`
  reconciliations, each of which cross-checked exactly (within $0, not just within
  tolerance) against an independently-tagged aggregate figure where one existed (6 of 9
  filings; AMD's 3 filings used the explicit combined tag instead, which needs no
  cross-check).
- Not yet hand-verified by the user as of this entry — the full 36-row table (equity, cash
  from operations, capex, total debt) plus debt reconciliation detail was presented for
  review immediately after this entry was written.

**Hand-verified (user), in a later message:** all 6 metrics spot-checked directly against
the HTML, including the harder ones — INTC FY2024 debt reconciliation (confirmed the note
shows Long-term debt 50,011 = noncurrent 46,282 + current 3,729, and that 46,282 is what
appears as "Debt" on the balance sheet), NVDA FY2026 total equity (157,293, balance sheet
dated Jan 25, 2026), INTC FY2025 capex (14,646, "Additions to property, plant and
equipment"), the NVIDIA capex scope caveat, and confirmation that Intel's *net* long-term
debt (50,011) was used rather than a *gross* pre-discount senior-notes figure (50,985) that
also appears in the filing — i.e., the user checked that the right one of two plausible
debt numbers was picked, not just that a number matching some tag was found.

## 2026-09-11 — extract.py: cogs; metrics.py: derived metrics + fiscal alignment

### What was built
- Added `cogs` to `extract.py` (`us-gaap:CostOfRevenue` for NVDA, `us-gaap:CostOfGoodsAndServicesSold`
  for AMD/Intel — same single-clean-tag pattern as revenue). All 9 values confirmed. User
  explicitly required gross margin not be computed unless COGS was itself properly
  extracted (not skipped or inferred) — it was extracted via the same XBRL method as
  everything else, not skipped.
- `metrics.py`: pure pandas, zero LLM calls. Computes `revenue_growth_yoy`, `gross_margin`,
  `net_margin`, `debt_to_equity`, `free_cash_flow` (cash_from_operations − capex) per
  (ticker, fiscal_year). Every derived row records its formula and the exact input values
  with a pointer back to `metrics.csv` (ticker+fiscal_year+metric) where the underlying
  XBRL source quote lives — nothing is a black box. Inputs are required to have
  `confidence: confirmed` in `metrics.csv`; if a required input isn't confirmed, the derived
  value is left as `None` with a stated reason rather than computed anyway.
- YoY growth is computed by sorting each ticker's periods chronologically by
  `period_end_date` and comparing to the immediately preceding period — not by fiscal-year
  label arithmetic — since NVDA's period-end dates don't line up with simple "current label
  minus 1" logic the way AMD/Intel's do. The earliest period per ticker has no prior period
  in the corpus and is explicitly left `N/A` rather than silently omitted.
- `free_cash_flow` rows for NVDA carry the NVIDIA capex-scope caveat (intangibles bundled
  into the capex figure) directly in a `caveat` column, not just as a comment somewhere else.
- Fiscal-year alignment table (`data/fiscal_alignment.csv`): NVDA's period end dates are
  matched to whichever AMD/Intel calendar fiscal year end is chronologically closest, with
  the exact offset in days always recorded (all three NVDA periods land 29 days after the
  corresponding AMD/Intel period end) — never treated as the same period.

### Verification performed
- **Automated (Claude):** ran `extract.py` (63 rows, all confirmed) and `metrics.py` (45
  derived-metric rows) successfully.
- **Automated (Claude), cross-check:** NVDA's computed gross margins (71.07% FY2026, 74.99%
  FY2025, 72.72% FY2024) match the filing's own reported percentages ("71.1%", "75.0%",
  "72.7%" respectively, found in the MD&A/summary tables during earlier `ingest.py`
  exploration) to within rounding — independent evidence the formula and inputs are correct,
  not just that the code ran without error.
- Not yet hand-verified by the user as of this entry — full derived-metrics table and
  alignment table presented for review immediately after.

**Hand-verified (user), in a later message:** reviewed `derived_metrics.csv`, confirmed the
gross-margin self-validation against NVIDIA's own reported percentages, confirmed the
fiscal-year alignment groups NVDA FY2026 with AMD/Intel FY2025 at the recorded +29-day
offset (not treated as the same period), and confirmed net margins (NVDA FY2026 55.6%, Intel
FY2024 -35.3%) check out.

## 2026-09-11 — qa_chain.py: RAG Q&A with dual refusal gates + structured metrics routing

### What was built
- `qa_chain.py` routes each question one of two ways:
  1. **Structured path** (no LLM call): if the question names a company and a derived
     metric (margin, growth, FCF, debt-to-equity), the answer is assembled directly from
     `data/derived_metrics.csv` — formula, inputs, and source pointer included verbatim, the
     same way it's stored. Zero hallucination risk for this path since no generation happens.
  2. **RAG path**: retrieve top-4 chunks from Chroma (filtered by ticker if one is named),
     answer via `llama3.2` using only the retrieved text, cite sources.
- Citations are never generated by the LLM. The prompt tells the model to reference
  `[SOURCE N]`; after generation, those placeholders are substituted with the real citation
  string (`[TICKER 10-K FYxxxx, Item N]`) assembled from the chunk's own metadata. The LLM
  cannot get a citation wrong because it never writes one.
- **Two independent pre-generation refusal gates**, deliberately not just one:
  1. Distance ceiling (0.75) — refuses before calling the LLM if the best retrieved chunk is
     too dissimilar.
  2. Keyword-overlap floor (0.34 of the question's content words must appear in the
     retrieved text) — added because the distance ceiling alone doesn't catch a real failure
     mode: a topically-nearby but non-answering chunk. Empirically checked before writing
     the gate: a wildly irrelevant query ("capital of France") scored 0.83-0.91 distance
     (caught by the ceiling); a plausible-sounding but unanswerable one ("NVIDIA CEO Jensen
     Huang's favorite programming language") scored 0.52-0.69 — inside what would look like
     an acceptable range on distance alone. That gap in the ceiling is exactly what the
     keyword-overlap floor covers.
  3. A third, post-generation check: every number in the LLM's answer must appear in the
     retrieved context, or the answer is discarded for the fixed refusal sentence.

### Bug found and fixed during testing
The first version of the cross-company comparison note pulled in an unrelated NVDA
alignment row: comparing NVDA FY2026 against AMD FY2025, the offset-note lookup filtered by
"any alignment row whose fiscal_year matches any picked company's year" (a set of
`{2025, 2026}`), which incorrectly also matched NVDA's own unrelated FY2025 row and printed
two offset lines instead of the one that was actually relevant. Fixed by filtering on the
exact (ticker, fiscal_year) pairs being compared, not the cross-product of tickers × years.
Caught by **Automated (Claude)** — running the actual test question and reading the output,
not by reasoning about the code.

### Verification performed — Automated (Claude)
- **Factual/RAG:** "What did NVIDIA say caused the increase in Data Center revenue?" →
  answered from 2 different fiscal years' Item 7 chunks (best distance 0.420), citing both.
  Checked the full retrieved chunk text directly: every specific claim in the answer
  ("strong demand for accelerated computing and AI solutions," "NVIDIA DGX Cloud and AI
  Foundations," "automotive data center processing demand") appears verbatim in the
  retrieved chunks — no fabrication.
- **Unanswerable/refusal:** "What is Jensen Huang's favorite programming language?" (no
  ticker named) → refused via the distance ceiling (0.754). Re-tested naming NVIDIA
  explicitly ("What is NVIDIA CEO Jensen Huang's favorite programming language?") to isolate
  the second gate → refused via keyword overlap (0.17, below the 0.34 floor) — confirms both
  gates function independently, not just the easier one.
- **Cross-company comparison:** "Compare NVIDIA and AMD gross margin" → routed to the
  structured path, answered NVDA FY2026 (71.07%) vs. AMD FY2025 (49.52%) with full
  formula/inputs for each, and correctly surfaced the +29-day fiscal offset for the actual
  pair being compared (after the bug fix above).
- Not yet hand-verified by the user as of this entry.

## 2026-09-11 — User-directed live stress test of qa_chain.py refusal behavior, 6 questions

The user ran their own set of 6 questions specifically designed to probe the hallucination
boundary before allowing `app.py` to be built — this is the most important verification
round so far, since it's what the assignment explicitly says will be sanity-checked. Two
real, distinct failures were found. Both are exactly the kind of finding the assignment's
writeup asks for directly.

### Failure 1: wrong number attributed to the right-sounding label (generation error)
**"What was Intel's net loss in 2024?"** → answered *"Intel's operating loss was $13.3
billion in 2024"* — wrong on two counts: (a) it's operating loss, not net loss (a materially
different, unasked-for figure), and (b) $13.3B doesn't match Intel's actual FY2024
consolidated net loss of $18.756B (confirmed via XBRL in an earlier `extract.py` step).
**Diagnosis (automated, Claude):** the retrieved chunks (checked directly) contain no
mention of "net loss" or "net income" at all — only segment-level operating figures,
including one chunk literally stating "Operating loss was $13.3 billion in 2024." The LLM
picked the closest-sounding number available and answered with it under the label the user
asked for, without flagging that it wasn't actually the requested metric. **Root cause:**
the post-generation numeric-grounding safeguard only checks that a number *appears* in the
retrieved context — it has no way to verify the number is attached to the *right concept*.
Grounding-by-presence is necessary but not sufficient. **Not yet fixed** — flagged, not
patched, pending the user's direction on whether to route "net income/net loss" questions
into the already-extracted, XBRL-confirmed `metrics.csv` value the same way margin/growth
questions already are.

### Failure 2: correct refusal on a question the corpus could actually answer (retrieval miss)
**"Did AMD acquire any companies in 2025?"** → refused. But AMD's FY2025 10-K states
directly: *"In March 2025, we acquired ZT Group Int'l, Inc. (ZT Systems)"* — an unambiguous
yes-answer that exists verbatim in the corpus. **Diagnosis (automated, Claude):** the top-4
retrieved chunks for this query (checked directly, all under the 0.75 distance ceiling)
were AMD's FY2024 chunk about *agreeing* to the acquisition (Aug 2024) plus boilerplate
Item 1 "Additional Information" content (state of incorporation, stock ticker) — the actual
FY2025 chunk confirming the March 2025 closing simply wasn't in the top 4 for this phrasing.
This is a pure recall failure: the LLM correctly refused given what it was shown (no
hallucination), but retrieval failed to surface the answering chunk. **Not yet fixed.**

### The other four questions, all correct
- "What was NVIDIA's total revenue in fiscal 2026?" → "$215.9 billion," cited [NVDA 10-K
  FY2026, Item 7]. **Automated (Claude) verified:** the exact sentence "Revenue for fiscal
  year 2026 was $215.9 billion" appears verbatim in that chunk.
- "What will NVIDIA's revenue be in 2027?" (future, unanswerable) → correctly refused (LLM
  itself declined per the prompt instruction, given retrieved historical-revenue content
  that couldn't answer a forward-looking question).
- "What is Lisa Su's salary?" → refused via the distance ceiling (0.809). **Automated
  (Claude) verified this is a correct refusal, not a retrieval miss:** grepped the source
  HTML directly — "Lisa" appears only in an exhibit-index reference to her employment
  agreement (filed as a separate exhibit, not in this document's text), and Item 11's body
  just incorporates executive compensation by reference to the proxy statement, as is
  standard for 10-Ks. The salary genuinely isn't in the corpus.
- "Compare Intel and NVIDIA net margin" → structured path, no LLM call, correct values for
  both (NVDA FY2026 55.60%, INTC FY2025 -0.51%) with the fiscal-offset note correctly
  scoped to just the two periods being compared (confirms the earlier bug fix held).

### Why this round matters for the writeup
This is the honest answer to "what is your hallucination rate, and would you trust this in
front of an executive": on a small, adversarially-chosen 6-question set, 4/6 were correct,
1/6 was a safe-direction failure (over-cautious refusal on an answerable question), and 1/6
was a genuine wrong-number error that a naive grounding check did not catch. The
wrong-number case is the one to fix first before trusting this in front of an executive —
it's the failure mode that actively misleads rather than merely under-delivers.

## 2026-09-11 — Fixing both stress-test failures, per user direction

### Fix for Failure 1 (wrong number under the right-sounding label)
Widened the structured (no-LLM) path in `qa_chain.py` with `answer_from_raw_metrics()`:
questions asking for a core financial figure we've already extracted and confirmed via
XBRL — revenue, net income/net loss, cogs, total equity, total debt, capex, cash from
operations — are now answered directly from `data/metrics.csv`, the same way margin/growth
questions already bypass the LLM via `data/derived_metrics.csv`. This fixes the Intel
net-loss error **by construction**, not by patching the specific case: the LLM is
structurally never the source of a headline number that's already been extracted and
verified. Router order: derived-metric aliases checked first (so "revenue growth" still
routes to growth, not raw revenue), then raw-metric aliases, then RAG as the final fallback.
A small readability touch: a negative `net_income` value is presented as "net loss of $X"
rather than "net income of -$X".

**Automated (Claude) verification:** re-ran "What was Intel's net loss in 2024?" → now
returns "net loss of $18,756,000,000" with the exact source row, no LLM call. Re-ran "What
was NVIDIA's total revenue in fiscal 2026?" → also now routes through the structured path
(a bonus — "total revenue" matches the new raw-metric aliases too), returning
$215,938,000,000 with its source row instead of an LLM-generated sentence.

### Fix attempt for Failure 2 (retrieval recall miss), per explicit user instruction: try one cheap fix, disclose rather than over-build
Bumped `RETRIEVAL_K` from 4 to 8. Explicitly did NOT build a re-ranker or query-rewriter —
user's direction was to try one cheap fix and log it as a known limitation if it didn't
fully resolve, rather than build retrieval infrastructure that couldn't be explained simply.

**Automated (Claude) verification:** re-ran "Did AMD acquire any companies in 2025?" → now
answers correctly: "Yes, AMD acquired ZT Systems on March 31, 2025, for a total purchase
consideration of $4.4 billion." Checked the actual retrieved chunk (distance 0.5788, would
have been outside the old k=4 window but within k=8) — verbatim match: *"On March 31, 2025
(the Acquisition Date), the Company completed the acquisition of all issued and outstanding
shares of ZT Systems... for a total purchase consideration of $4.4 billion."* The cheap fix
worked for this specific case.

**Known limitation, disclosed rather than chased further:** k=8 fixed this one instance, but
retrieval recall misses are a general risk that a fixed k cannot fully rule out — a corpus
of 5,863 chunks means any given fact has no guarantee of ranking in the top-k for every
possible phrasing of a question about it. The consequence of a miss is over-refusal (the
system says "I cannot answer" when the corpus actually could), not hallucination — the safe
direction, but still a real gap. A more robust fix (hybrid keyword+semantic search, a
re-ranking pass, or query rewriting/expansion) was deliberately not built, per direct user
instruction to avoid adding retrieval infrastructure the system couldn't be simply explained
by test time.

## 2026-09-11 — eval.py: labeled question set + evaluation framework

### What was built
`data/eval_questions.json` (20 questions: 4 factual, 4 derived, 3 comparison, 3 narrative,
1 event/acquisition, 5 unanswerable) with hand-verified ground truth pulled directly from
the current `metrics.csv`/`derived_metrics.csv` (not from memory), including the two
already-diagnosed failure modes as explicit regression tests (F2: Intel net loss; E1: AMD
acquisition). `eval.py` runs every question through `qa_chain.answer_question` and grades
with pure string matching against the labeled ground truth — deliberately not an
LLM-as-judge, since that would introduce a second unreliable model into the thing being
measured. Reports answer correctness and citation accuracy on the answerable subset, and
hallucination rate (non-refusal) on the unanswerable subset. Full detail written to
`data/eval_results.md`.

### The eval run itself found two NEW bugs — not the two we already knew about
First run: 93.3% correctness (14/15), 100% citation accuracy, 0% hallucination. Before
accepting that number, checked the one FAIL (N2) and, on a hunch, checked whether the one
narrative PASS (N1) had passed for the right reason.

**N1 was a false pass.** "Why did NVIDIA's gross margin change in fiscal 2024?" contains the
words "gross margin," which matched a *derived-metric* alias and routed to the structured
no-LLM path — which returned only `"gross_margin = 72.72%"` with no explanation at all. My
own substring grading ("72.7" ∈ expected answers) accepted this as correct because the bare
number happened to overlap with the percentage, without the question ever being answered.
**True baseline before any fix: 13/15 (86.7%), not 14/15 (93.3%).**

**N2 failed outright for the related reason.** "What did NVIDIA say caused the increase in
Data Center revenue?" contains the substring "revenue," which matched the *raw-metric*
alias and also routed to the structured path, returning NVIDIA's total revenue figure —
answering a completely different question than the one asked.

**Root cause (one fix for both):** the keyword router checked only for the *presence* of a
metric name, never whether the question was actually asking a numeric-lookup question versus
an explanatory one. Fixed by adding `is_explanatory()` — a check for "why / caused / drove /
driven by / reason for / explain / factors" — that forces the RAG path regardless of any
metric keyword also present. Also hardened `eval.py` itself so this class of false pass can't
recur silently: narrative-category questions are now graded incorrect if their answer somehow
came from a structured (no-LLM) path, independent of substring matching.

**First attempt at the fix was itself incomplete** — `is_explanatory()` initially checked for
the literal phrase "what caused," which doesn't match N2's actual phrasing "what did NVIDIA
**say caused**." Broadened to check for the standalone word "caused" instead of the full
phrase, re-verified it didn't accidentally flip routing for any of the other 19 questions.

### A test-design brittleness, also disclosed rather than quietly patched
After the router fix, N2 correctly routed to RAG and produced a genuinely correct, grounded
answer — but with `RETRIEVAL_K=8` (raised earlier for the AMD-acquisition fix) surfacing a
different valid set of chunks than the k=4 manual test this question was originally written
against, the answer was phrased around "accelerated computing / 162% growth" instead of
"DGX Cloud / Hopper," and failed my overly narrow expected-answer list. Checked the full
answer text and its citations directly — genuinely correct and grounded, not a system defect.
Broadened the expected-answer list to include the more robust, less phrasing-specific terms
rather than lock in one specific chunk's wording as the only acceptable answer.

### The user's follow-up: "make the eval honest by construction" — 5 novel adversarial questions
After seeing the clean 100%/100%/0% result, the user (correctly) didn't trust it as a
measure of general capability — two of the hardest cases were regression tests for bugs this
exact system had already been fixed to handle. Explicit instruction: add genuinely novel
hard questions never tested before, across five specific adversarial patterns, and **do not
tune anything to preserve the 100% — report whatever the real number is.**

Five questions added (`MIX1`, `OBS1`, `QN1`, `SUB1`, `OFF1`), each with ground truth verified
against the source HTML *before* running them through `qa_chain` even once (so the ground
truth couldn't be quietly shaped by seeing the system's actual behavior first). Grading
schema extended to support "grouped" expected-answer requirements (all groups must have a
match, not just any one substring) for the two compound questions (MIX1, QN1) where getting
only half right is a different, still-wrong answer, not partial credit.

**Result: 89.5% answer correctness (17/19), 100% citation accuracy (19/19), 0% hallucination
(0/6). Two new, real, previously-undiscovered failures — not tuned away:**

**MIX1 failed, worse than predicted.** "What is NVIDIA's net margin and what is AMD's
revenue growth, both for their most recent fiscal year?" — the router doesn't just conflate
the two metrics, it silently drops one company and one metric entirely: `detect_metric()`
returns the first alias match in dict-iteration order ("revenue growth," not "net margin"),
and the resulting answer is *only* `NVDA revenue_growth_yoy = 65.47%` — AMD is never
mentioned at all. Root cause: the single-metric-per-query architecture has no mechanism to
pair a specific company with a specific metric when more than one of each is named.

**OBS1 failed as a new instance of the exact bug class the F2/Intel-net-loss fix was
supposed to close.** "What was NVIDIA's Automotive revenue in fiscal 2024?" — "Automotive
revenue" contains the substring "revenue," matching the raw-metric alias, so the question
routed to the structured path and returned NVIDIA's *total* company revenue ($60,922M)
instead of the Automotive segment figure actually asked for ($1.1B, verified present in the
source). The earlier fix made the LLM structurally unable to state the wrong headline
number — but it didn't anticipate a *qualified* ask (a segment, not the whole company) being
swallowed by the same keyword match. This is a materially wrong number stated with full
confidence, not a refusal — the same danger class as the original Intel bug, just not yet
covered by the fix.

**The other three novel questions passed, including two designed to be hard:** OFF1 (NVDA
vs. Intel debt-to-equity, a pairing never tested before) correctly surfaced the fiscal-offset
note; QN1 (Intel revenue decline, a compound number+narrative question routed entirely
through RAG with no structured-path safety net for the number) got both the number and the
explanation right; SUB1 (AMD FY2015 free cash flow — sounds like a normal question about a
metric we do extract, but the year predates the corpus by 8 years) was correctly refused
rather than hallucinated.

**Not fixed as of this entry, per explicit user instruction to report the real number
first.** Both are logged as concrete, reproducible failures with root causes identified,
should the user want to prioritize a fix (most direct one: only route to structured metric
paths when the metric alias match is not immediately preceded by a qualifying segment/product
word, and give the mixed-metric case its own two-part parsing rather than a single
metric+ticker-list assumption).

### Final numbers, after fixing two real routing bugs and one test-design flaw (original 20-question set)
**100% answer correctness (15/15), 100% citation accuracy (15/15), 0% hallucination rate
(0/5).** These are honestly high, and it matters why: 20 questions is a small set, all 3
companies' data is fully extracted and verified, and both regression-test questions (F2, E1)
specifically target failures that were already found and fixed earlier in the build — this
number reflects a system that has now been through several rounds of adversarial testing on
this exact corpus, not a system with a low prior probability of failure on novel questions
or a larger/messier corpus. The honest caveat for WRITEUP.md: this eval set is small and
was partly built by the same process that already found and fixed its hardest cases, so
100% here is a reflection of iteration, not a claim that hallucination risk is fully
eliminated.

### Full 6-question re-run after both fixes
All 6 correct:
1. NVIDIA revenue FY2026 → $215,938,000,000, structured path (metrics.csv), source row shown.
2. Intel net loss 2024 → $18,756,000,000, structured path, source row shown. **(Failure 1, fixed.)**
3. NVIDIA revenue 2027 (future) → correctly refused.
4. AMD acquisitions in 2025 → correctly answered (ZT Systems, $4.4B, March 31 2025), citing
   [AMD 10-K FY2025, Item 8]. **(Failure 2, resolved by the k=8 bump for this case.)**
5. Lisa Su's salary → correctly refused (distance ceiling; confirmed genuinely absent from
   the corpus in the prior round).
6. Intel vs NVIDIA net margin → structured path, both values correct, fiscal-offset note
   correctly scoped.
