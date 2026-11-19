"""
demo.py
=======
Interactive demo. Type a multiple-choice question, and the system
shows the small model's chosen answer, a confidence rating from the trained fuzzy
gate, the decision (answer / abstain / escalate), and the fuzzy rule that produced
the rating.

Prerequisites:
  * the model is cached (same one used for extraction)
  * you have run run_demo.py at least once, so results/gate.npz exists

Run (inside the src folder):
    python demo.py

Then either type your own question and four options, or press Enter on the
question line to pull a random question from data/questions.jsonl (its correct
answer is shown so you can see whether the model got it right).

One-shot form:
    python demo.py --question "What gas do plants absorb?" \
                   --options "oxygen,carbon dioxide,nitrogen,helium"
"""

import argparse
import json
import os
import sys

import numpy as np

import extract_signals as E
from fuzzy_anfis import ANFIS, _sigmoid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GATE = os.path.join(ROOT, "results", "gate.npz")
QFILE = os.path.join(ROOT, "data", "questions.jsonl")

# decision thresholds on the fuzzy confidence
TAU_HIGH = 0.70   # at or above -> answer
TAU_LOW = 0.45    # at or below -> escalate to a larger model


def load_gate():
    if not os.path.exists(GATE):
        sys.exit("results/gate.npz not found. Run 'python run_demo.py' first.")
    d = np.load(GATE, allow_pickle=True)
    net = ANFIS(n_inputs=int(d["J"]), n_mf=int(d["M"]),
                first_order=bool(d["first_order"]))
    net.c, net.s, net.P, net.b = d["c"], d["s"], d["P"], d["b"]
    net.sel = d["sel"]
    net.R = net.sel.shape[0]
    features = [str(x) for x in d["features"]]
    return net, d["mu"], d["sd"], features


def confidence_and_rule(net, mu, sd, features, signals):
    """signals: dict with the four feature keys. Returns (confidence, rule_text)."""
    x = np.array([[signals[f] for f in features]])
    xs = (x - mu) / sd
    conf = float(net.predict_proba(xs)[0])
    net.forward(xs)                       # populate firing strengths
    wn = net._cache["wn"][0]
    r = int(np.argmax(wn))
    mf_names = ["low", "high"] if net.M == 2 else [f"MF{m}" for m in range(net.M)]
    antecedent = " AND ".join(f"{features[j]} is {mf_names[net.sel[r, j]]}"
                              for j in range(net.J))
    return conf, antecedent, float(wn[r])


def decide(conf):
    if conf >= TAU_HIGH:
        return "ANSWER (confident)"
    if conf <= TAU_LOW:
        return "ESCALATE to a larger model (low confidence)"
    return "ABSTAIN / flag for review (borderline)"


def run_one(model, tok, device, net, mu, sd, features, question, options, gold=None):
    item = {"question": question, "options": options}
    row, pick = E._run_mcq(model, tok, item, device)
    conf, rule, firing = confidence_and_rule(net, mu, sd, features, row)
    print("\n" + "=" * 64)
    print(f"Question : {question}")
    for i, o in enumerate(options):
        tag = "  <- model pick" if o == pick else ""
        star = " (correct)" if gold is not None and o == gold else ""
        print(f"   {E.LETTERS[i]}) {o}{star}{tag}")
    print("-" * 64)
    print(f"Model's answer      : {pick}")
    if gold is not None:
        print(f"Actually correct    : {'YES' if pick == gold else 'NO'}  (gold: {gold})")
    print(f"Fuzzy confidence    : {conf:.3f}")
    print(f"Decision            : {decide(conf)}")
    print(f"Signals             : " + ", ".join(f"{k}={row[k]:.3f}" for k in features))
    print(f"Top firing rule     : IF {rule}  (firing {firing:.2f})")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--question", default=None)
    ap.add_argument("--options", default=None, help="comma-separated options")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model} (this takes a moment)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", attn_implementation="eager")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    net, mu, sd, features = load_gate()
    print(f"[ready] gate loaded. Thresholds: answer>={TAU_HIGH}, escalate<={TAU_LOW}")

    # one-shot mode
    if args.question and args.options:
        opts = [o.strip() for o in args.options.split(",")]
        run_one(model, tok, device, net, mu, sd, features, args.question, opts)
        return

    # interactive mode
    pool = []
    if os.path.exists(QFILE):
        with open(QFILE, encoding="utf-8") as f:
            pool = [json.loads(l) for l in f if l.strip()]
    rng = np.random.default_rng()
    print("\nType a question, or press Enter for a random one. Ctrl+C to quit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not q:
            if not pool:
                print("no questions.jsonl to sample from; type your own.")
                continue
            item = pool[int(rng.integers(len(pool)))]
            run_one(model, tok, device, net, mu, sd, features,
                    item["question"], item["options"], item.get("answer"))
            continue
        raw = input("Four options, comma-separated> ").strip()
        opts = [o.strip() for o in raw.split(",") if o.strip()]
        if len(opts) < 2:
            print("need at least two options.")
            continue
        run_one(model, tok, device, net, mu, sd, features, q, opts)


if __name__ == "__main__":
    main()
