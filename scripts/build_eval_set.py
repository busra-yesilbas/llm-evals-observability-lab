"""
build_eval_set.py — Generate and save the evaluation dataset.

Run from project root:
    python scripts/build_eval_set.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_evals_lab.config import load_config
from llm_evals_lab.data.evalset import EvalSetLoader, build_synthetic_eval_set
from llm_evals_lab.utils import setup_logging


def main() -> None:
    setup_logging()
    cfg = load_config()

    print("=" * 60)
    print("NovaSaaS Eval Set Builder")
    print("=" * 60)

    examples = build_synthetic_eval_set()
    loader = EvalSetLoader(cfg.eval_dir())
    path = loader.save(examples)

    print(f"\n✓ Saved {len(examples)} eval examples → {path}")

    # Print summary by category and difficulty
    from collections import Counter
    cats = Counter(e.category.value for e in examples)
    diffs = Counter(e.difficulty.value for e in examples)
    answerable = sum(1 for e in examples if e.is_answerable)

    print(f"\nCategory distribution:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    print(f"\nDifficulty distribution:")
    for diff, count in sorted(diffs.items()):
        print(f"  {diff}: {count}")

    print(f"\nAnswerable: {answerable} / {len(examples)}")
    print(f"Unanswerable: {len(examples) - answerable} / {len(examples)}")

    print("\nEval examples:")
    for e in examples:
        ans_flag = "✓" if e.is_answerable else "✗"
        print(f"  [{ans_flag}] {e.example_id} | {e.difficulty.value:6s} | {e.category.value:12s} | {e.question[:60]}...")

    print("\n✓ Eval set build complete.")


if __name__ == "__main__":
    main()
