"""
make_questions.py
=================
Download a standard question-answering dataset and write it into the
question/answer JSONL format that extract_signals.py expects. This gives you a
few hundred to a few thousand real questions with known answers, instead of the
tiny 10-question sample.

It needs the Hugging Face 'datasets' package and an internet connection (the same
connection that just downloaded the model):

    pip install datasets

Default dataset is SciQ (clean science questions with short correct answers).

Examples (run inside the src folder):
    python make_questions.py --n 800
    python make_questions.py --dataset sciq --n 1500 --out ../data/questions.jsonl

After this, extract the signals using the file you just wrote:
    python extract_signals.py --data ../data/questions.jsonl --out ../data/signals.csv
    python run_demo.py
"""

import argparse
import json
import sys


def from_sciq(n, split="train"):
    """SciQ: build multiple-choice items with one correct answer and three
    distractors. Each item is {question, answer, options} where options is the
    four choices in shuffled order."""
    import random
    from datasets import load_dataset
    rng = random.Random(0)
    ds = load_dataset("allenai/sciq", split=split)
    items = []
    for row in ds:
        q = row["question"].strip()
        correct = row["correct_answer"].strip()
        distractors = [row["distractor1"].strip(), row["distractor2"].strip(),
                       row["distractor3"].strip()]
        if not q or not correct or not all(distractors):
            continue
        options = [correct] + distractors
        rng.shuffle(options)
        items.append({"question": q, "answer": correct, "options": options})
        if len(items) >= n:
            break
    return items


def from_triviaqa(n, split="validation"):
    """TriviaQA (rc.nocontext): question + answer.value."""
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    items = []
    for row in ds:
        q = row["question"].strip()
        a = row["answer"]["value"].strip()
        if q and a:
            items.append({"question": q, "answer": a})
        if len(items) >= n:
            break
    return items


BUILDERS = {"sciq": from_sciq, "triviaqa": from_triviaqa}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a questions.jsonl file.")
    ap.add_argument("--dataset", choices=list(BUILDERS), default="sciq")
    ap.add_argument("--n", type=int, default=800, help="how many questions")
    ap.add_argument("--out", default="../data/questions.jsonl")
    args = ap.parse_args(argv)

    print(f"[build] {args.dataset}, up to {args.n} questions")
    items = BUILDERS[args.dataset](args.n)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(items)} questions to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
