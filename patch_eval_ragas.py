"""Applies three fixes to src/retrieval/eval_ragas.py.

Run from the project root:
    docker compose exec dev python patch_eval_ragas.py

Safety: every edit needs its exact target line to appear exactly once. If any
edit cannot be applied, NOTHING is written. The patched file must compile
before it replaces the original, and the original is saved as eval_ragas.py.bak.
Running it twice is harmless (already-applied edits are skipped).

Fixes
  1. LocalEmbeddings stores its model as ._st, not .model. ragas logs
     getattr(embeddings, "model") into a field that must be a string, so a
     SentenceTransformer object there raised a ValidationError.
  2. evaluate() gets a RunConfig: one judge job at a time and a longer
     timeout, because 8 jobs hitting one local Ollama at once exceed ragas's
     default 180 s. New flags: --timeout and --workers.
  3. dict(result) replaced by summarize_scores(): dict(result) crashes on
     ragas's result object, and the new summary also reports how many rows the
     judge failed to score.
"""
import os
import py_compile
import shutil
import sys
import tempfile

TARGET = "src/retrieval/eval_ragas.py"

SUMMARY_FUNCS = r'''def summarize_scores(result):
    """Mean of the non-NaN scores per metric, plus how many rows the judge
    failed on. A NaN means the judge produced no usable output (timeout or
    malformed JSON), which is common with a small local judge, so it must be
    reported rather than silently averaged away."""
    import math

    summary = {}
    for name in result.scores[0].keys():
        vals = [s[name] for s in result.scores]
        ok = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        summary[name] = {
            "mean": (sum(ok) / len(ok)) if ok else None,
            "n_scored": len(ok),
            "n_failed": len(vals) - len(ok),
        }
    return summary


def print_summary(summary):
    print("\nRagas results (mean over the rows the judge actually scored):")
    for name, s in summary.items():
        mean = "n/a" if s["mean"] is None else format(s["mean"], ".3f")
        print(f"  {name:20s} {mean:>6}   scored={s['n_scored']}  judge_failed={s['n_failed']}")


def ground_truth_for(fault):'''

# (label, already-applied marker, old text, new text)
EDITS = [
    ("embeddings attr (init)", r'self._st = SentenceTransformer',
     r'self.model = SentenceTransformer(model_name)',
     r'self._st = SentenceTransformer(model_name)  # not ".model": ragas logs that attribute and needs a str'),
    ("embeddings attr (query)", r'self._st.encode(text,',
     r'return self.model.encode(text, normalize_embeddings=True).tolist()',
     r'return self._st.encode(text, normalize_embeddings=True).tolist()'),
    ("embeddings attr (documents)", r'self._st.encode(texts,',
     r'return self.model.encode(texts, normalize_embeddings=True).tolist()',
     r'return self._st.encode(texts, normalize_embeddings=True).tolist()'),
    ("cli flags", r'"--workers"',
     r'ap.add_argument("--out", default="data/eval/ragas_results.json")',
     'ap.add_argument("--out", default="data/eval/ragas_results.json")\n'
     '    ap.add_argument("--timeout", type=int, default=900, help="seconds allowed per judge job")\n'
     '    ap.add_argument("--workers", type=int, default=1, help="parallel judge jobs (keep 1 for local Ollama)")'),
    ("RunConfig import", r'from ragas.run_config import RunConfig',
     r'from ragas import evaluate',
     'from ragas import evaluate\n    from ragas.run_config import RunConfig'),
    ("run_config argument", r'run_config=RunConfig(',
     r'embeddings=judge_embeddings,',
     'embeddings=judge_embeddings,\n'
     '        run_config=RunConfig(timeout=args.timeout, max_workers=args.workers, max_retries=2),'),
    ("summary helpers", r'def summarize_scores',
     r'def ground_truth_for(fault):',
     SUMMARY_FUNCS),
    ("json summary", r'summary = summarize_scores(result)',
     r'json.dump({"summary": dict(result), "rows": rows}, f, indent=2)',
     'summary = summarize_scores(result)\n'
     '        json.dump({"summary": summary, "rows": rows, "per_row_scores": result.scores}, f, indent=2)'),
    ("print summary", 'print_summary(summary)\n    print(f',
     r'print(f"\nWritten to {args.out}")',
     'print_summary(summary)\n    print(f"\\nWritten to {args.out}")'),
]


def main():
    if not os.path.exists(TARGET):
        sys.exit(f"{TARGET} not found - run this from the project root (Bruviti_AI).")
    original = open(TARGET).read()
    text = original
    problems = []

    for label, marker, old, new in EDITS:
        if marker in text:
            print(f"  skip   {label} (already applied)")
            continue
        n = text.count(old)
        if n != 1:
            problems.append(f"{label}: expected 1 match, found {n} for: {old!r}")
            continue
        text = text.replace(old, new)
        print(f"  apply  {label}")

    if problems:
        print("\nNOTHING WAS CHANGED. Problems:")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    if text == original:
        print("\nAlready fully patched. Nothing to do.")
        return

    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(text)
    tmp.close()
    try:
        py_compile.compile(tmp.name, doraise=True)
    except py_compile.PyCompileError as e:
        os.unlink(tmp.name)
        sys.exit(f"Patched file does not compile, NOTHING WAS CHANGED:\n{e}")
    os.unlink(tmp.name)

    shutil.copy(TARGET, TARGET + ".bak")
    with open(TARGET, "w") as f:
        f.write(text)
    print(f"\nPATCHED OK. Original saved as {TARGET}.bak")


if __name__ == "__main__":
    main()
