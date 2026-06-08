"""
End-to-end grounding smoke test.
Run with: venv_actual/bin/python -m tests.smoke_generate

For each query, ask yourself:
  Could this response have come from anywhere other than the retrieved chunks printed above it?
  If yes — the grounding prompt needs tightening.
  If no  — grounding is working.
"""

from src.retrieve import retrieve
from src.generate import generate

SMOKE_QUERIES = [
    "What do students say about noise and security at The Wyatt?",
    "Which KMetro bus routes stop at WMU, and is it free for students?",
    "If my landlord keeps my security deposit after 30 days, what are my rights under Michigan law?",
]

_SEP = "=" * 70


def run_smoke() -> None:
    print(f"\n{_SEP}")
    print("  GROUNDED GENERATION SMOKE TEST")
    print(f"{_SEP}\n")

    for i, query in enumerate(SMOKE_QUERIES, 1):
        print(f"[Q{i}] {query}\n")

        chunks = retrieve(query, k=5)
        print("  Retrieved context:")
        for j, chunk in enumerate(chunks, 1):
            print(f"    [{j}] {chunk['source_file']} | {chunk['header']} | dist={chunk['distance']:.4f}")
        print()

        answer = generate(query, chunks)
        print("  Generated answer:")
        for line in answer.splitlines():
            print(f"    {line}")
        print()
        print(f"  {'─' * 66}")
        print()


if __name__ == "__main__":
    run_smoke()
