"""Offline time-shift evaluation of already trained trial-3 checkpoints.

The reference may shift after the final voiced input frame. This removes the
arbitrary target-onset penalty while preserving frame-level content/duration.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from trial2 import StreamMemory, batch_examples, onset, runs


def best_alignment(template, predicted, earliest_start):
    best = None
    for start in range(earliest_start, len(predicted) - len(template) + 1):
        reference = [0] * len(predicted)
        reference[start:start + len(template)] = template
        matches = sum(a == b for a, b in zip(predicted, reference))
        if best is None or matches > best["matching_frames"]:
            best = {"aligned_start": start, "matching_frames": matches,
                    "frame_accuracy": matches / len(predicted),
                    "full_frame_exact": predicted == reference,
                    "active_frame_accuracy": sum(a == b for a, b in zip(predicted, reference)
                                                 if b != 0) / sum(b != 0 for b in reference)}
    if best is None:
        raise ValueError("No causal alignment fits the evaluation window")
    return best


def aligned_score(source, target, predicted):
    last_input = max(i for i, value in enumerate(source) if value != 0)
    first_output = onset(predicted)
    no_interruption = first_output is not None and first_output > last_input
    # Primary measure: compare against the original input, including its pauses.
    input_match = best_alignment(source[:last_input + 1], predicted, last_input + 1)
    # Diagnostic: training used a canonical repeat with one-frame pauses.
    first_target = onset(target)
    last_target = max(i for i, value in enumerate(target) if value != 0)
    target_match = best_alignment(target[first_target:last_target + 1], predicted,
                                  last_input + 1)
    return {"no_interruption": no_interruption,
            "last_input_frame": last_input,
            "first_output_frame": first_output,
            "input_aligned_start": input_match["aligned_start"],
            "input_frame_accuracy": input_match["frame_accuracy"] if no_interruption else 0.0,
            "input_active_frame_accuracy": input_match["active_frame_accuracy"] if no_interruption else 0.0,
            "input_full_frame_exact": input_match["full_frame_exact"] if no_interruption else False,
            "target_aligned_start": target_match["aligned_start"],
            "target_frame_accuracy": target_match["frame_accuracy"] if no_interruption else 0.0,
            "target_full_frame_exact": target_match["full_frame_exact"] if no_interruption else False}


@torch.no_grad()
def evaluate(model, device, seed, n, gain, use_plastic):
    model.eval()
    rng = random.Random(seed)
    groups = {}
    examples = []
    for index in range(n):
        x, y, mask, items = batch_examples(rng, 1, 3, 6, device)
        source, target, words = items[0]
        logits = model(x, gain, rng.randint(2, 5), use_plastic=use_plastic)
        predicted = logits.argmax(-1)[0, :len(target)].tolist()
        score = aligned_score(source, target, predicted)
        repeated = len(set(words)) < len(words)
        keys = ("overall", f"length_{len(words)}", "repeated" if repeated else "distinct")
        for key in keys:
            if key not in groups:
                groups[key] = {"n": 0, "no_interruption": 0, "sequence_exact": 0,
                               "input_full_frame_exact": 0, "target_full_frame_exact": 0,
                               "input_frame_acc_sum": 0.0, "input_active_acc_sum": 0.0,
                               "target_frame_acc_sum": 0.0}
            stats = groups[key]
            stats["n"] += 1
            stats["no_interruption"] += int(score["no_interruption"])
            stats["sequence_exact"] += int(score["no_interruption"] and runs(predicted) == words)
            stats["input_full_frame_exact"] += int(score["input_full_frame_exact"])
            stats["target_full_frame_exact"] += int(score["target_full_frame_exact"])
            stats["input_frame_acc_sum"] += score["input_frame_accuracy"]
            stats["input_active_acc_sum"] += score["input_active_frame_accuracy"]
            stats["target_frame_acc_sum"] += score["target_frame_accuracy"]
        if index < 5:
            examples.append({"words": words, "predicted_words": runs(predicted),
                             "original_target_start": onset(target), **score})
    result = {}
    for key, stats in groups.items():
        count = stats["n"]
        result[key] = {"n": count,
                       "no_interruption_rate": stats["no_interruption"] / count,
                       "sequence_exact_no_interruption": stats["sequence_exact"] / count,
                       "input_aligned_full_frame_exact": stats["input_full_frame_exact"] / count,
                       "input_aligned_frame_accuracy": stats["input_frame_acc_sum"] / count,
                       "input_aligned_active_frame_accuracy": stats["input_active_acc_sum"] / count,
                       "target_aligned_full_frame_exact": stats["target_full_frame_exact"] / count,
                       "target_aligned_frame_accuracy": stats["target_frame_acc_sum"] / count}
    return result, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Drive trial3_results directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"count_per_condition": args.count, "device": str(device), "variants": {}}
    for variant in ("plastic", "recurrent"):
        checkpoint = args.input / variant / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        model = StreamMemory().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        report["variants"][variant] = {}
        for gain in (0.0, 0.15, 0.35):
            metrics, examples = evaluate(model, device, 5151, args.count, gain,
                                         use_plastic=(variant == "plastic"))
            report["variants"][variant][str(gain)] = metrics
            print(variant, gain, metrics["overall"], flush=True)
            if gain == 0.15:
                (args.output / f"{variant}_examples.json").write_text(
                    json.dumps(examples, indent=2), encoding="utf-8")
    (args.output / "aligned_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
