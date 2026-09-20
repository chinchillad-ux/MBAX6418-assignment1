#!/usr/bin/env python3
"""
LLM-as-judge binary sentiment classifier for Amazon Gift_Cards reviews.

Classifies each review as POSITIVE/NEGATIVE from its TITLE + TEXT via an
OpenAI-compatible endpoint (here a vLLM server). Evaluates against a
rating-derived ground truth (1-3 -> NEGATIVE, 4-5 -> POSITIVE).

Reads the .jsonl.gz directly (streaming) so no decompressed copy is needed.

Usage:
    .venv/bin/python classify.py [--limit N] [--out predictions.csv]
"""
import argparse
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------- config ----
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://dobolyi.com:9001/v1")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
MAX_TOKENS = 2048  # reasoning model needs headroom for its thinking trace

DATA = Path(__file__).parent / "Gift_Cards.jsonl.gz"

SYSTEM_PROMPT = (
    "You are a sentiment classifier for Amazon product reviews. "
    "Classify the sentiment expressed in the review's TITLE and TEXT as "
    "POSITIVE or NEGATIVE. Respond with exactly one line of the form "
    "'Sentiment: POSITIVE' or 'Sentiment: NEGATIVE' and nothing else."
)

LABEL_FROM_RATING = lambda r: "POSITIVE" if r >= 4 else "NEGATIVE"  # 1-3 NEG, 4-5 POS


# ------------------------------------------------------------ inference ----
def classify(client, title: str, text: str):
    """Send one review to the model, return the predicted label string."""
    user_prompt = f"TITLE:\n{title}\n\nTEXT:\n{text}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    # vLLM Qwen3 puts the thinking trace here; answer may land in either.
    reasoning = (
        getattr(msg, "reasoning", None)
        or getattr(msg, "reasoning_content", None)
        or ""
    )
    combined = f"{reasoning}\n{content}"
    # Take the LAST mention of a label: the reasoning trace may contain both.
    matches = re.findall(r"\b(POSITIVE|NEGATIVE)\b", combined.upper())
    return matches[-1] if matches else "UNKNOWN"


# ------------------------------------------------------------- evaluation ----
def evaluate(predictions):
    y_true = [p["true"] for p in predictions]
    y_pred = [p["pred"] for p in predictions]
    known = [(t, p) for t, p in zip(y_true, y_pred) if p != "UNKNOWN"]
    acc = sum(t == p for t, p in known) / len(known) if known else 0.0
    # confusion: rows[true][pred]  -> matrix[0]=NEG, matrix[1]=POS
    m = {"NEGATIVE": {"NEGATIVE": 0, "POSITIVE": 0},
         "POSITIVE": {"NEGATIVE": 0, "POSITIVE": 0}}
    for t, p in known:
        m[t][p] += 1
    return acc, m


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    if not API_KEY:
        ap.error("OPENAI_API_KEY must be set in the environment")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=180)

    out = Path(args.out)
    first = out.suffix == ".csv"
    f = open(out, "w")
    if first:
        f.write("rating,true_sentiment,title,text,predicted_sentiment\n")

    predictions = []
    processed = errors = unknown = 0
    t0 = time.time()

    with gzip.open(DATA, "rt", encoding="utf-8") as gz:
        for line in gz:
            if processed >= args.limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = (rec.get("title") or "").strip()
            text = (rec.get("text") or "").strip()
            rating = float(rec.get("rating", 0))
            if not title and not text:
                continue
            try:
                pred = classify(client, title, text)
            except Exception as e:  # per-review tolerance
                errors += 1
                pred = "UNKNOWN"
                print(f"  !! error review #{processed}: {e}", file=sys.stderr)
            if pred == "UNKNOWN":
                unknown += 1
            true = LABEL_FROM_RATING(rating)
            predictions.append(
                {"true": true, "pred": pred, "rating": rating,
                 "title": title, "text": text}
            )
            f.write(
                f"{rating},{true},{json.dumps(title)},{json.dumps(text)},{pred}\n"
            )
            f.flush()
            processed += 1
            if processed % 10 == 0:
                print(f"  {processed}/{args.limit}  "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr)

    f.close()
    acc, cm = evaluate(predictions)

    print("\n=============== RESULTS ===============")
    print(f"Reviews processed : {processed}")
    print(f"API errors        : {errors}")
    print(f"Unknown labels    : {unknown}")
    print(f"Accuracy (vs rating label) : {acc:.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"               pred NEG   pred POS")
    print(f" true NEG   :   {cm['NEGATIVE']['NEGATIVE']:>7}  {cm['NEGATIVE']['POSITIVE']:>7}")
    print(f" true POS   :   {cm['POSITIVE']['NEGATIVE']:>7}  {cm['POSITIVE']['POSITIVE']:>7}")
    if cm['POSITIVE']['NEGATIVE'] + cm['NEGATIVE']['POSITIVE'] > 0:
        tp, fn, fp, tn = (cm['POSITIVE']['POSITIVE'], cm['POSITIVE']['NEGATIVE'],
                          cm['NEGATIVE']['POSITIVE'], cm['NEGATIVE']['NEGATIVE'])
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        print(f"\nPrecision (POS)   : {prec:.3f}")
        print(f"Recall (POS)      : {rec:.3f}")
        print(f"F1 (POS)          : {2*prec*rec/(prec+rec) if (prec+rec) else 0:.3f}")
    print(f"\nPredictions written to: {out.resolve()}")


if __name__ == "__main__":
    main()
