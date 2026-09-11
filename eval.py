"""Evaluation framework: run the labeled question set through qa_chain.answer_question
and report answer correctness, citation accuracy, and hallucination rate on unanswerable
questions.

Grading is pure string matching against hand-verified ground truth (no LLM-as-judge —
that would just add a second unreliable model into the thing being measured).
"expected_answer_contains" supports two shapes: a flat list of strings means ANY one match
is sufficient (used when several phrasings of the same fact are all acceptable); a list of
lists ("groups") means at least one substring from EACH group must appear — used for
compound questions where multiple distinct facts must ALL be present (e.g. a question
asking for two companies' two different metrics, or a quant-to-narrative question where
both the number and the explanation must be right; getting only one of them isn't "mostly
right," it's a different, still-wrong answer). Citation is "correct" if the expected
ticker(s) and fiscal year(s) appear in the answer text; for comparison questions, an
expected offset-note requirement is also checked. Unanswerable questions are graded purely
on whether the system refused.

Run: python eval.py
"""
import json
from pathlib import Path

import pandas as pd

import qa_chain

QUESTIONS_PATH = Path("data/eval_questions.json")
RESULTS_PATH = Path("data/eval_results.md")
OFFSET_NOTE_MARKER = "fiscal-year alignment"


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def check_answer_correct(q: dict, answer_text: str) -> bool:
    expected = q["expected_answer_contains"]
    if not expected:
        return True  # unanswerable questions have no content to match
    lower = answer_text.lower()
    is_grouped = isinstance(expected[0], list)
    if is_grouped:
        # Every group must have at least one hit — compound questions where multiple
        # distinct facts must ALL be present, not just one of them.
        return all(any(sub.lower() in lower for sub in group) for group in expected)
    return any(sub.lower() in lower for sub in expected)


def check_citation_correct(q: dict, answer_text: str) -> bool:
    if not q["expected_sources"]:
        return True  # unanswerable questions have no source to check
    for src in q["expected_sources"]:
        if src["ticker"] not in answer_text:
            return False
        if src["fiscal_year"] is not None and str(src["fiscal_year"]) not in answer_text:
            return False
    return True


def check_offset_note_correct(q: dict, answer_text: str) -> bool:
    expected = q.get("expect_offset_note")
    if expected is None:
        return True  # not a comparison question, not applicable
    has_note = OFFSET_NOTE_MARKER in answer_text.lower()
    return has_note == expected


def evaluate_question(q: dict) -> dict:
    result = qa_chain.answer_question(q["question"])
    answer_text = result["answer"]
    refused = result["refused"]

    row = {
        "id": q["id"], "category": q["category"], "question": q["question"],
        "answerable": q["answerable"], "refused": refused, "answer": answer_text,
        "path": result["source"],
    }

    if not q["answerable"]:
        row["hallucinated"] = not refused
        row["correct"] = refused  # correct behavior for an unanswerable question is to refuse
        row["citation_correct"] = None
        return row

    row["hallucinated"] = None
    content_ok = check_answer_correct(q, answer_text) and not refused
    citation_ok = check_citation_correct(q, answer_text) and not refused
    offset_ok = check_offset_note_correct(q, answer_text)
    # Narrative ("why"/"what caused") questions need the actual MD&A commentary, not just a
    # metric's number — a structured no-LLM answer can coincidentally contain a matching
    # substring (e.g. a bare "gross_margin = 72.72%" satisfies a substring check for "72.7")
    # without ever explaining anything. Caught exactly this in the first eval run (N1).
    if q["category"] == "narrative" and "no LLM call" in result["source"]:
        content_ok = False
    row["correct"] = content_ok and offset_ok
    row["citation_correct"] = citation_ok
    return row


def main():
    questions = load_questions()
    rows = [evaluate_question(q) for q in questions]
    df = pd.DataFrame(rows)

    answerable = df[df["answerable"]]
    unanswerable = df[~df["answerable"]]

    answer_correctness_pct = 100 * answerable["correct"].mean() if len(answerable) else float("nan")
    citation_accuracy_pct = 100 * answerable["citation_correct"].mean() if len(answerable) else float("nan")
    hallucination_rate_pct = 100 * unanswerable["hallucinated"].mean() if len(unanswerable) else float("nan")

    summary_lines = [
        f"Questions total: {len(df)}  (answerable: {len(answerable)}, unanswerable: {len(unanswerable)})",
        f"Answer correctness (answerable subset): {answer_correctness_pct:.1f}% ({int(answerable['correct'].sum())}/{len(answerable)})",
        f"Citation accuracy (answerable subset):  {citation_accuracy_pct:.1f}% ({int(answerable['citation_correct'].sum())}/{len(answerable)})",
        f"Hallucination rate (unanswerable subset): {hallucination_rate_pct:.1f}% ({int(unanswerable['hallucinated'].sum())}/{len(unanswerable)})",
    ]

    print("=== Summary ===")
    for line in summary_lines:
        print(line)
    print()
    print("=== Per-question detail ===")
    for _, r in df.iterrows():
        if r["answerable"]:
            status = "PASS" if r["correct"] else "FAIL"
            cite = "OK" if r["citation_correct"] else "BAD"
            print(f"[{r['id']:3s}] {status:4s} (citation {cite:3s}) {r['category']:11s} {r['question']}")
        else:
            status = "PASS (refused)" if not r["hallucinated"] else "FAIL (hallucinated)"
            print(f"[{r['id']:3s}] {status:20s} {r['category']:11s} {r['question']}")
        if r["correct"] is False or (r["hallucinated"] is True):
            print(f"        -> answer: {r['answer'][:200]}")

    # Write full results to a markdown file.
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Evaluation results\n\n")
        f.write("## Summary\n\n")
        for line in summary_lines:
            f.write(f"- {line}\n")
        f.write("\n## Per-question detail\n\n")
        f.write("| ID | Category | Answerable | Correct | Citation OK | Hallucinated | Path | Question |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['id']} | {r['category']} | {r['answerable']} | {r['correct']} | "
                     f"{r['citation_correct']} | {r['hallucinated']} | {r['path']} | {r['question']} |\n")
        f.write("\n## Full answers\n\n")
        for _, r in df.iterrows():
            f.write(f"### {r['id']}: {r['question']}\n\n")
            f.write(f"- Answerable (ground truth): {r['answerable']}\n")
            f.write(f"- Refused: {r['refused']}\n")
            f.write(f"- Path: {r['path']}\n\n")
            f.write(f"```\n{r['answer']}\n```\n\n")

    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
