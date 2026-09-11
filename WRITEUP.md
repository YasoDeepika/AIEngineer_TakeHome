# WRITEUP

## Architecture and files considered

Seven files, kept deliberately small: `ingest.py` (HTML → cleaned, section-tagged chunks →
Chroma), `extract.py` (a separate, deterministic pass that reads financial figures directly
from each filing's inline XBRL tags, not from chat-style retrieval), `metrics.py` (pure
pandas: derived metrics + fiscal-year alignment table, no LLM), `qa_chain.py` (RAG Q&A with
citation, refusal, and a structured no-LLM path for figures we've already extracted),
`app.py` (Streamlit: Ask / Compare / Metrics detail), `eval.py` (labeled question set +
scoring), plus `data/` for the pipeline's outputs. I considered folding extraction into the
same retrieval path used for chat, but split them deliberately: the dashboard's numbers
needed to be verifiable independent of whatever the chat LLM happens to retrieve for a given
phrasing.

## Hallucination rate, and would I trust this in front of an executive

On the original 20-question labeled set: 100% answer correctness, 100% citation accuracy,
0% hallucination. That number is real, but it is not the honest one to lead with — the set
is small and self-consistent, and two of its hardest cases were regression tests for
failures this exact system had already been fixed to handle. I added five genuinely novel
adversarial questions afterward specifically to avoid reporting a number that only measured
whether known bugs stayed fixed, and did not tune anything to preserve the higher score.
**On the harder 25-question set: 89.5% answer correctness (17/19 answerable), 100% citation
accuracy, 0% hallucination on the unanswerable subset.** Two new, real failures surfaced —
both wrong-number errors, not fabrications from nothing, and both fully diagnosed (below and
in `writeup_notes.md`).

Would I trust it in front of an executive? For a headline figure (revenue, margin, net
income) — yes, because those are now structurally incapable of being LLM-generated; they
come straight from XBRL tags or hand-verified formulas. For an open-ended or compound
question — not without a caveat, because the router's single-metric-per-question assumption
can silently answer a different, narrower question than the one asked. **What I'd fix
first**: the newest failure (below) — a segment-qualified question ("Automotive revenue")
getting swallowed by a whole-company keyword match and answered confidently wrong. It's the
same danger class as the bug that motivated the whole structured-routing redesign, just an
unanticipated variant of it.

## Most interesting insight, and my confidence in it

The three companies' trajectories diverge sharply and the filings' own numbers tell a
coherent story without needing any outside data. NVIDIA's revenue grew 114% then 65% over
the two most recent fiscal years while holding 71-75% gross margins — a company riding the
AI buildout at a scale that dwarfs the other two. AMD is executing a steadier, genuine
improvement: gross margin climbed from 46.1% (FY2023) to 49.5% (FY2025), revenue growth
accelerated to 34%, and leverage stayed low (debt-to-equity under 0.05 throughout). Intel is
the outlier in the other direction: revenue roughly flat-to-declining across all three years,
a $18.8B net loss in FY2024 driven by non-cash impairments, negative free cash flow in every
year in the corpus, and debt-to-equity 8-10x higher than its two competitors. **Confidence is
high in the numbers** — each is XBRL-extracted and, where possible, independently
cross-validated (e.g. computed gross margins matched the filings' own stated percentages to
within rounding, and total-debt reconciliations matched an independently-tagged aggregate
exactly). Confidence in the *causal story* (AI demand as the dividing line) is more moderate
— that's a reasonable read consistent with what each filing's own MD&A says drove its
numbers, but it's inference layered on top of verified figures, not itself a verified fact.

## One failure I found and diagnosed

Stress-testing `qa_chain.py` with "What was Intel's net loss in 2024?" returned *"Intel's
operating loss was $13.3 billion in 2024"* — wrong on two counts: operating loss, not net
loss, and the real consolidated net loss was $18.756B. The retrieved chunks contained no
mention of "net loss" at all, only a segment operating-loss figure; the LLM answered with
the closest-sounding number it had, under the label I asked for, without flagging the
mismatch. My numeric-grounding safeguard passed it because $13.3B *did* appear verbatim in
the context — grounding-by-presence doesn't verify the number is attached to the right
concept. Fixed by routing core financial figures (revenue, net income, cogs, equity, debt,
capex, cash from operations) through the already-extracted, XBRL-confirmed values directly,
the same way derived metrics already bypassed the LLM. Verified fixed and added as a
permanent regression test. The harder eval round later found a sibling of this bug that
isn't fixed yet: "Automotive revenue" matched the same "revenue" keyword and returned NVIDIA's
total revenue instead of the segment figure — the fix generalized to headline figures but not
to qualified/segment asks of the same figure.

## How I used AI tools, and where I overrode one

The AI's first cut of `qa_chain.py` routed only *derived* metrics (margins, growth) through
the no-LLM structured path, leaving core figures like revenue and net income to
retrieval-grounded generation. I overrode that design after the net-loss failure above: I
directed that every core figure we'd already extracted and verified via XBRL should bypass
the LLM entirely, not just ratios computed from them. That one architectural correction is
what actually fixed the failure — a better prompt would not have. Separately, when a
retrieval-recall miss surfaced (a real AMD acquisition not appearing in the top-4 chunks), I
explicitly declined the AI's implicit path toward building a re-ranker or query-rewriter and
told it to try one cheap fix (raise k) and disclose the limitation honestly if that didn't
fully resolve it — a smaller, more explainable system over a more complete one I couldn't
fully vouch for.

## Where LangChain helped, and where I dropped to raw calls

LangChain earned its place in the plumbing: `RecursiveCharacterTextSplitter`, the `Document`
abstraction, `Chroma` as a vector store, and `ChatOllama`/`OllamaEmbeddings` as clean,
swappable wrappers around the local Ollama server meant never hand-rolling HTTP calls or a
custom chunking format. I did not use LangChain's higher-level chain/LCEL composition for
the actual question-answering logic in `qa_chain.py` — the refusal gates, the citation
substitution, and the routing between structured lookup and RAG are all plain Python. Those
are precondition checks and non-LLM control flow ("refuse before generating," "answer this
from a CSV instead of the model"), and a generic chain abstraction had no natural way to
express them that would still be as easy to read and verify line-by-line as the explicit
`if`/`return` statements are now.
