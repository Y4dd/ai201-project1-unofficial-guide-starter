from src.retrieve import retrieve

EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": "Which is cheaper, On-campus or Off-campus?",
        "expected": "Off-campus",
    },
    {
        "id": 2,
        "question": "What is the closest Off-Campus housing option to WMU?",
        "expected": "The Tate on Howard (note: ranks 7th — 'closest' vs 'walking distance / directly across Howard St' semantic mismatch; k=9 surfaces it)",
    },
    {
        "id": 3,
        "question": "How is Off-campus transportation for WMU students?",
        "expected": "Free with Bronco Card (KMetro buses)",
    },
    {
        "id": 4,
        "question": "Does Hunter's Ridge include utilities in the rent?",
        "expected": "Yes, except the electric bill",
    },
    {
        "id": 5,
        "question": "What is the most dangerous area to rent near WMU?",
        "expected": "Near downtown / Vine neighborhood",
    },
]

_SEP = "─" * 70


def run_eval(k: int = 9) -> None:
    print(f"\n{'=' * 70}")
    print(f"  RETRIEVAL EVALUATION — top-{k} chunks per question")
    print(f"{'=' * 70}\n")

    for q in EVAL_QUESTIONS:
        print(f"[Q{q['id']}] {q['question']}")
        print(f"  Expected answer hint: {q['expected']}")
        print()

        chunks = retrieve(q["question"], k=k)
        for i, chunk in enumerate(chunks, 1):
            dist_label = f"dist={chunk['distance']:.4f}"
            print(f"  [{i}] {chunk['source_file']} | {chunk['header']} | {dist_label}")
            preview = chunk["text"][:300].replace("\n", " ")
            if len(chunk["text"]) > 300:
                preview += f"  ... ({len(chunk['text']) - 300} more chars)"
            print(f"      {preview}")
            print()

        print(_SEP)
        print()


if __name__ == "__main__":
    run_eval()
