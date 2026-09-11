# SEC 10-K Analyst — NVIDIA / AMD / Intel

A small, local-only RAG dashboard that answers questions about three semiconductor
companies' 10-K filings, computes derived financial metrics from XBRL-extracted figures,
and cites every claim back to its source. No API keys, no paid services — everything runs
on your machine via Ollama.

## What it does

- **Ask** — natural-language Q&A over the filings. Core financial figures (revenue, net
  income, margins, growth, debt-to-equity, free cash flow) are answered directly from
  verified, extracted values with zero LLM involvement; open-ended questions are answered
  via retrieval-augmented generation with mandatory citations; questions the filings don't
  support are refused, not guessed at.
- **Compare** — charts of revenue, growth, margins, leverage, and free cash flow across all
  three companies over time, with NVIDIA's offset fiscal calendar (it reports on a
  January year-end; AMD and Intel report on a December year-end) explicitly surfaced rather
  than silently aligned.
- **Metrics detail** — every extracted figure and every derived metric, with its exact
  source quote, formula, and inputs, so any number in the dashboard can be traced back to
  the filing it came from.

## Install and run

**1. Install Ollama** and pull the two local models this project uses:

```
winget install Ollama.Ollama        # or download from ollama.com
ollama pull llama3.2
ollama pull nomic-embed-text
```

**2. Create a virtual environment and install dependencies:**

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**3. Build the vector store** (one-time; re-run any time `files/` changes):

```
.venv\Scripts\python ingest.py
```

This reads the 9 HTML filings in `files/`, chunks them, embeds them with
`nomic-embed-text`, and persists them to `data/chroma/`.

**4. Extract financial figures and compute derived metrics** (also one-time / re-run after
re-ingesting):

```
.venv\Scripts\python extract.py
.venv\Scripts\python metrics.py
```

**5. Run the dashboard:**

```
.venv\Scripts\python -m streamlit run app.py
```

Opens at `http://localhost:8501`.

**Optional — run the evaluation suite:**

```
.venv\Scripts\python eval.py
```

Runs the labeled question set in `data/eval_questions.json` and writes results to
`data/eval_results.md`.

## Document sources

All 9 filings are public 10-Ks obtained from SEC EDGAR. Fiscal year and period-end date
below are read from each filing's own XBRL tags (`dei:DocumentFiscalYearFocus` /
`dei:DocumentPeriodEndDate`), not inferred from the filename.

| Company | Ticker | CIK | Fiscal Year | Period End | Local file | EDGAR filings list |
|---|---|---|---|---|---|---|
| NVIDIA Corporation | NVDA | 0001045810 | FY2024 | 2024-01-28 | `NVDA_10K_2024.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&type=10-K) |
| NVIDIA Corporation | NVDA | 0001045810 | FY2025 | 2025-01-26 | `NVDA_10K_2026.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&type=10-K) |
| NVIDIA Corporation | NVDA | 0001045810 | FY2026 | 2026-01-25 | `NVDA_10K_2025.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001045810&type=10-K) |
| Advanced Micro Devices, Inc. | AMD | 0000002488 | FY2023 | 2023-12-30 | `AMD_10K_2023.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000002488&type=10-K) |
| Advanced Micro Devices, Inc. | AMD | 0000002488 | FY2024 | 2024-12-28 | `AMD_10K_2024.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000002488&type=10-K) |
| Advanced Micro Devices, Inc. | AMD | 0000002488 | FY2025 | 2025-12-27 | `AMD_10K_2025.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000002488&type=10-K) |
| Intel Corporation | INTC | 0000050863 | FY2023 | 2023-12-30 | `INTC_10K_2023.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000050863&type=10-K) |
| Intel Corporation | INTC | 0000050863 | FY2024 | 2024-12-28 | `INTC_10K_2024.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000050863&type=10-K) |
| Intel Corporation | INTC | 0000050863 | FY2025 | 2025-12-27 | `INTC_10K_2025.html` | [link](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000050863&type=10-K) |

**Note on filenames:** `NVDA_10K_2025.html` and `NVDA_10K_2026.html` are swapped relative to
the fiscal year they actually report (the "2025" file is FY2026 data; the "2026" file is
FY2025 data) — a mismatch between the download filename and the filing's own fiscal year
label, confirmed against the cover page and income statement headers. The pipeline reads
fiscal year from each filing's own XBRL tags, not the filename, so this is handled
correctly regardless; it's noted here so the filenames don't mislead a reader browsing
`files/` directly. CIKs above were read directly from each filing's own XBRL entity
identifier, not looked up separately.

## Design choices and why

- **Fully local, no API keys**: `ChatOllama` (llama3.2) for generation, `OllamaEmbeddings`
  (nomic-embed-text) for embeddings, `Chroma` for the vector store — everything is a local
  HTTP call to Ollama on `localhost:11434`.
- **Financial figures come from XBRL tags, not from an LLM reading tables.** These 10-Ks are
  inline XBRL documents — every number in the rendered tables is also machine-tagged (e.g.
  `<ix:nonFraction name="us-gaap:Revenues">`). `extract.py` reads these tags directly and
  resolves them against the filing's own period definitions, rather than asking a 3B-param
  local model to read a financial table and hope it transcribes the right cell. This was the
  single biggest accuracy decision in the project — see WRITEUP.md for the failure this
  prevents (and one it didn't fully prevent).
- **Citations are assembled from metadata, never transcribed by the LLM.** The prompt asks
  the model to reference `[SOURCE N]`; those placeholders are substituted with the real
  citation string built from chunk metadata after generation. The LLM cannot get a citation
  wrong because it never writes one.
- **Fiscal-year misalignment is surfaced, not resolved.** NVIDIA's fiscal year end and
  AMD/Intel's are about a month apart. Every cross-company comparison in the Compare tab and
  in `qa_chain.py` states the actual offset rather than treating differing periods as
  equivalent.
- **Two independent pre-generation refusal gates, not one**: a similarity-distance ceiling
  catches wildly irrelevant questions; a keyword-overlap floor catches the harder case a
  distance score alone misses — a topically-adjacent but non-answering retrieved chunk. A
  third, post-generation check discards any answer containing a number not present in the
  retrieved context.
- **Known, disclosed limitations, not chased into over-engineering**: Intel's 10-Ks
  restructure away from the standard SEC Item ordering (the filing says so itself), so
  section-level citation coverage for Intel is 9-10/24 items vs. 24/24 for NVIDIA and AMD —
  though it does include the two sections (MD&A, Financial Statements) that matter most.
  Retrieval recall on unusual phrasings can still miss an answering chunk that exists in the
  corpus; the failure direction is over-refusal, not hallucination. Full detail, including
  two real bugs found via adversarial testing (one fixed, one open), is in WRITEUP.md.
