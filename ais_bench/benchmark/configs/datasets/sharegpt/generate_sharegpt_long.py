#!/usr/bin/env python3
"""Generate tokenizer-verified user messages; no inference requests are sent."""
import argparse
import json
import random
import uuid
from pathlib import Path


def exact_text(tokenizer, source, target):
    ids = tokenizer.encode(source, add_special_tokens=False)
    if len(ids) < target + 64:
        raise ValueError("Source is too short")
    # Decoding and re-encoding can change token boundaries. Check actual text.
    lengths = [target]
    for delta in range(1, 65):
        lengths.extend([target - delta, target + delta])
    for size in lengths:
        if size <= 0:
            continue
        text = tokenizer.decode(
            ids[:size], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "\ufffd" not in text and len(tokenizer.encode(text, add_special_tokens=False)) == target:
            return text
    raise RuntimeError(f"Cannot construct exactly {target} tokens with this tokenizer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conversations", type=int, default=1)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--first-user-tokens", type=int, default=20000)
    parser.add_argument("--next-user-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=400,
                        help="Budget estimate only; set AISBench max_out_len separately")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    for key in ("conversations", "turns", "first_user_tokens", "next_user_tokens", "output_tokens"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    output = Path(args.output).resolve()
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        parser.error("Output or manifest already exists; choose another output path")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code,
    )
    rng = random.Random(args.seed)
    nouns = ["storage", "network", "scheduler", "database", "memory", "service"]
    records, counts = [], []
    for conv in range(args.conversations):
        messages, sizes = [], []
        for turn in range(args.turns):
            target = args.first_user_tokens if turn == 0 else args.next_user_tokens
            # Unique identifier is early in the prefix to limit cross-conversation hits.
            marker = uuid.UUID(int=rng.getrandbits(128)).hex
            source = (f"Record {marker}. Conversation {conv}, turn {turn + 1}. "
                      "Read the following experimental notes and explain the observations.\n")
            chunks = [source]
            row = 0
            while True:
                for _ in range(max(32, target // 16)):
                    chunks.append(
                        f"Observation {row}: the {rng.choice(nouns)} processed "
                        f"{rng.randrange(100, 10000)} records while the measured delay was "
                        f"{rng.randrange(1, 999)} milliseconds. The result needs comparison "
                        "with the preceding measurement.\n"
                    )
                    row += 1
                if len(tokenizer.encode("".join(chunks), add_special_tokens=False)) >= target + 64:
                    break
            text = exact_text(tokenizer, "".join(chunks), target)
            messages.extend([
                {"from": "human", "value": text},
                {"from": "gpt", "value": "placeholder"},
            ])
            sizes.append(len(tokenizer.encode(text, add_special_tokens=False)))
        records.append({"id": f"long-context-{conv:05d}", "conversations": messages})
        counts.append({"id": records[-1]["id"], "user_tokens": sizes})
        print(f"{records[-1]['id']}: user_tokens={sizes}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    final_budget = (args.first_user_tokens + (args.turns - 1) * args.next_user_tokens
                    + args.turns * args.output_tokens)
    manifest.write_text(json.dumps({
        "arguments": vars(args), "verified_counts": counts,
        "token_count_scope": "user content only; add_special_tokens=False",
        "last_turn_input_plus_output_excluding_chat_template": final_budget,
        "note": "Actual outputs and chat template must be measured at runtime."
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset: {output}\nManifest: {manifest}")
    print(f"Final context estimate: {final_budget} + chat template/system overhead")


if __name__ == "__main__":
    main()
