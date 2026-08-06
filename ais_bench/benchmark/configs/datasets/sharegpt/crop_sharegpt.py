#!/usr/bin/env python3
"""Create a deterministic, multi-turn ShareGPT subset for prefix-cache tests."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select complete ShareGPT conversations while preserving consecutive "
            "human/GPT turns for PD-disaggregated prefix-cache benchmarks."
        )
    )
    parser.add_argument("--input", required=True, help="Source ShareGPT JSON file")
    parser.add_argument("--output", required=True, help="Output subset JSON file")
    parser.add_argument(
        "--model-path", required=True, help="Qwen tokenizer/model directory"
    )
    parser.add_argument("--target", type=int, default=256)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=128,
        help="Estimated generated tokens per assistant turn",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "target",
        "min_turns",
        "max_turns",
        "min_tokens",
        "max_tokens",
        "output_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_turns > args.max_turns:
        raise ValueError("--min-turns cannot exceed --max-turns")
    if args.min_tokens > args.max_tokens:
        raise ValueError("--min-tokens cannot exceed --max-tokens")


def extract_pairs(conversations: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return strict, consecutive human/GPT pairs or an empty list."""
    if not isinstance(conversations, list):
        return []

    pairs = []
    for index in range(0, len(conversations) - 1, 2):
        human = conversations[index]
        assistant = conversations[index + 1]
        if not isinstance(human, dict) or not isinstance(assistant, dict):
            return []
        if human.get("from") != "human" or assistant.get("from") != "gpt":
            return []
        if not isinstance(human.get("value"), str):
            return []
        if not isinstance(assistant.get("value"), str):
            return []
        pairs.append((human, assistant))
    return pairs


def estimate_final_prompt_tokens(
    tokenizer: Any,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    output_tokens: int,
) -> int:
    """Estimate the final request length when prior model outputs are fixed-size."""
    messages = []
    assistant_placeholder = " token" * output_tokens
    for turn_index, (human, _) in enumerate(pairs):
        messages.append({"role": "user", "content": human["value"]})
        if turn_index < len(pairs) - 1:
            messages.append(
                {"role": "assistant", "content": assistant_placeholder}
            )

    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)
    except (AttributeError, TypeError, ValueError):
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        return len(tokenizer.encode(rendered, add_special_tokens=True))


def main() -> None:
    args = parse_args()
    validate_args(args)

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required to tokenize Qwen prompts; install the "
            "AISBench model dependencies before generating the subset"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    with Path(args.input).open("r", encoding="utf-8") as input_file:
        source = json.load(input_file)
    if not isinstance(source, list):
        raise ValueError("ShareGPT input must be a top-level JSON list")

    candidates: list[tuple[int, dict[str, Any]]] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        pairs = extract_pairs(item.get("conversations"))
        if len(pairs) < args.min_turns:
            continue

        pairs = pairs[: args.max_turns]
        while len(pairs) >= args.min_turns:
            token_count = estimate_final_prompt_tokens(
                tokenizer, pairs, args.output_tokens
            )
            if token_count <= args.max_tokens:
                break
            pairs.pop()
        if len(pairs) < args.min_turns:
            continue

        token_count = estimate_final_prompt_tokens(
            tokenizer, pairs, args.output_tokens
        )
        if token_count < args.min_tokens:
            continue

        conversations = [message for pair in pairs for message in pair]
        candidates.append(
            (
                token_count,
                {"id": item.get("id"), "conversations": conversations},
            )
        )

    random_generator = random.Random(args.seed)
    random_generator.shuffle(candidates)
    selected = candidates[: args.target]
    if len(selected) < args.target:
        raise RuntimeError(
            f"Only {len(selected)} conversations matched the filters; "
            f"requested {args.target}. Relax the token/turn limits."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            [item for _, item in selected],
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    selected_token_counts = [token_count for token_count, _ in selected]
    selected_items = [item for _, item in selected]
    total_requests = sum(len(item["conversations"]) // 2 for item in selected_items)
    print(f"Selected conversations: {len(selected_items)}")
    print(f"Total inference rounds: {total_requests}")
    print(
        "Estimated final-prompt tokens: "
        f"min={min(selected_token_counts)}, "
        f"max={max(selected_token_counts)}, "
        f"avg={sum(selected_token_counts) / len(selected_token_counts):.1f}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
