"""
extract_signals.py
==================
Run a small language model over a set of questions and record four internal
"confidence" signals plus whether the model's answer was correct.

Two modes, chosen automatically per question:
  * MULTIPLE CHOICE (preferred): if a question has an "options" list, the model is
    asked to choose A/B/C/D. This is the right way to evaluate a small model: the
    answer is unambiguous and the confidence signals are measured over the four
    options. One forward pass per question, so it is fast and there is no
    truncation.
  * OPEN ENDED (fallback): if there are no options, the model generates a short
    answer and correctness is checked by matching the gold answer text.

Run on any machine with the model cached (CPU is fine for a 0.5B model):
    pip install torch transformers datasets
    python extract_signals.py --data ../data/questions.jsonl --out ../data/signals.csv

Output columns: entropy, margin, attention_spread, layer_instability,
max_softmax, correct.

The four signals (computed without changing the model's weights)
---------------------------------------------------------------
1. entropy            : entropy of the answer distribution. In multiple choice
                        this is over the four options; high means the model
                        spreads probability across choices = less certain.
2. margin             : gap between the top-1 and top-2 answer probabilities.
3. attention_spread   : normalised entropy of the final layer's attention from the
                        last prompt token back over the prompt. Diffuse = no clear
                        evidence.
4. layer_instability  : via the logit lens, the fraction of layers whose predicted
                        token disagrees with the final layer's prediction. High =
                        the internal guess kept changing across depth.
"""

import argparse
import csv
import json
import re
import sys

LETTERS = ["A", "B", "C", "D", "E", "F"]


# ----------------------------------------------------------------- utilities
def normalise(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_correct_text(prediction, gold):
    p, g = normalise(prediction), normalise(gold)
    if not g:
        return 0
    return int(g in p or p in g)


def load_questions(path):
    items = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    elif path.endswith(".csv"):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                items.append(row)
    else:
        raise ValueError("data file must be .jsonl or .csv with question/answer")
    return items


def _entropy(prob, eps=1e-12):
    import numpy as np
    return float(-np.sum(prob * np.log(prob + eps)))


# ------------------------------------------------- shared signal computation
def _structure_signals(model, out, prompt_len):
    """Attention spread and layer instability at the last prompt position.
    Shared by both modes. `out` is a forward pass with attentions + hidden states.
    """
    import numpy as np
    import torch

    pos = prompt_len - 1

    # Attention spread: final layer, last token's attention over the prompt.
    last_att = out.attentions[-1][0].mean(0)              # (seq, seq)
    row = last_att[pos, :prompt_len].float().cpu().numpy()
    if row.sum() > 0:
        row = row / row.sum()
        att_spread = _entropy(row) / float(np.log(len(row) + 1e-12))
    else:
        att_spread = 1.0

    # Layer instability via logit lens at the last prompt position.
    final_logits = out.logits[0, pos].float()
    final_tok = int(torch.argmax(final_logits))
    head = model.get_output_embeddings()
    norm = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    disagree, n_layers = 0, len(out.hidden_states) - 1
    for h in out.hidden_states[1:]:
        vec = h[0, pos]
        if norm is not None:
            vec = norm(vec)
        if int(torch.argmax(head(vec).float())) != final_tok:
            disagree += 1
    layer_instability = disagree / max(n_layers, 1)
    return float(att_spread), float(layer_instability)


# --------------------------------------------------------- multiple-choice
def _letter_token_ids(tok):
    """Token id for each option letter, trying a leading space and bare forms."""
    ids = {}
    for L in LETTERS:
        cands = []
        for form in (f" {L}", L):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                cands.append(enc[0])
        ids[L] = cands
    return ids


def _mcq_prompt(tok, question, options):
    lines = [f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options)]
    body = ("Answer the multiple choice question with the single letter of the "
            "correct option.\n\nQuestion: " + question + "\nOptions:\n"
            + "\n".join(lines) + "\n")
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": body}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        return prompt + "Answer: ("
    return body + "Answer: ("


def _run_mcq(model, tok, item, device):
    import numpy as np
    import torch

    question, options = item["question"], item["options"]
    gold = item.get("answer")
    k = len(options)
    prompt = _mcq_prompt(tok, question, options)
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        out = model(**enc, output_attentions=True, output_hidden_states=True)

    next_logits = out.logits[0, -1].float()
    letter_ids = _letter_token_ids(tok)
    # Logit for each option = best matching letter-token logit.
    opt_logits = np.full(k, -1e9)
    for i in range(k):
        cands = letter_ids[LETTERS[i]]
        if cands:
            opt_logits[i] = max(float(next_logits[c]) for c in cands)
    # Softmax over the k options.
    z = opt_logits - opt_logits.max()
    probs = np.exp(z)
    probs = probs / probs.sum()

    order = np.argsort(probs)[::-1]
    pred_idx = int(order[0])
    top1, top2 = float(probs[order[0]]), float(probs[order[1]])

    entropy = _entropy(probs)
    margin = top1 - top2
    max_softmax = top1
    att_spread, layer_instability = _structure_signals(model, out, prompt_len)
    correct = int(options[pred_idx] == gold) if gold is not None else 0
    return dict(entropy=entropy, margin=margin, attention_spread=att_spread,
                layer_instability=layer_instability, max_softmax=max_softmax,
                correct=correct), options[pred_idx]


# --------------------------------------------------------------- open ended
def _run_open(model, tok, item, device, max_new_tokens):
    import numpy as np
    import torch

    question, gold = item["question"], item.get("answer", "")
    if getattr(tok, "chat_template", None):
        msgs = [{"role": "user", "content": f"Answer in a few words. {question}"}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    else:
        prompt = f"Question: {question}\nAnswer:"
    enc = tok(prompt, return_tensors="pt").to(device)
    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    ent, marg, mx = [], [], []
    for logits in gen.scores:
        p = torch.softmax(logits[0].float(), dim=-1).cpu().numpy()
        top = np.sort(p)[::-1]
        ent.append(_entropy(p)); marg.append(float(top[0] - top[1]))
        mx.append(float(top[0]))
    answer = tok.decode(gen.sequences[0, prompt_len:], skip_special_tokens=True)

    with torch.no_grad():
        out = model(gen.sequences, output_attentions=True,
                    output_hidden_states=True)
    att_spread, layer_instability = _structure_signals(model, out, prompt_len)
    return dict(entropy=float(np.mean(ent)), margin=float(np.mean(marg)),
                attention_spread=att_spread, layer_instability=layer_instability,
                max_softmax=float(np.mean(mx)),
                correct=is_correct_text(answer, gold)), answer


# -------------------------------------------------------------------- driver
def extract(model_name, data_path, out_path, max_new_tokens=8, limit=None):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    # eager attention is required so the model can return attention weights
    # (the faster sdpa backend used by default in transformers 5.x cannot).
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", attn_implementation="eager")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"[device] {device}")

    items = load_questions(data_path)
    if limit:
        items = items[:limit]
    mode = "multiple-choice" if items and "options" in items[0] else "open-ended"
    print(f"[data] {len(items)} questions ({mode})")

    rows = []
    for k, item in enumerate(items):
        if "options" in item:
            row, ans = _run_mcq(model, tok, item, device)
        else:
            row, ans = _run_open(model, tok, item, device, max_new_tokens)
        rows.append({kk: (round(vv, 6) if isinstance(vv, float) else vv)
                     for kk, vv in row.items()})
        if (k + 1) % 20 == 0:
            acc = np.mean([r["correct"] for r in rows])
            print(f"[{k+1}/{len(items)}] pick={ans!r} correct={row['correct']} "
                  f"running_acc={acc:.3f}")

    cols = ["entropy", "margin", "attention_spread", "layer_instability",
            "max_softmax", "correct"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    acc = float(np.mean([r["correct"] for r in rows])) if rows else 0.0
    print(f"[done] wrote {out_path}  ({len(rows)} rows, base accuracy {acc:.3f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract SLM confidence signals.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True, help=".jsonl or .csv")
    ap.add_argument("--out", default="../data/signals.csv")
    ap.add_argument("--max_new_tokens", type=int, default=8,
                    help="open-ended mode only")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    extract(args.model, args.data, args.out, args.max_new_tokens, args.limit)


if __name__ == "__main__":
    sys.exit(main())
