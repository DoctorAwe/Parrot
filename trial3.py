"""Trial 3: independently train plastic and recurrent-only variants.

Uses the same synthetic frequency-frame task as trial 2. All metrics are
computed on a held-out deterministic set in the Colab notebook.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from trial2 import StreamMemory, batch_examples, edit_distance, onset, runs, training_loss, write_wave


def group_stats():
    return {"n": 0, "sequence_matches": 0, "frame_matches": 0,
            "frame_correct": 0, "frame_total": 0, "edit_sum": 0.0,
            "early_onsets": 0, "late_onsets": 0, "onset_error_sum": 0.0,
            "missing_onsets": 0, "silent_false": 0, "silent_total": 0,
            "active_correct": 0, "active_total": 0}


def finalize(s):
    n = max(s["n"], 1)
    return {"n": s["n"],
            "sequence_exact": s["sequence_matches"] / n,
            "full_frame_exact": s["frame_matches"] / n,
            "frame_accuracy": s["frame_correct"] / max(s["frame_total"], 1),
            "normalized_edit_distance": s["edit_sum"] / n,
            "early_onset_rate": s["early_onsets"] / n,
            "late_onset_rate": s["late_onsets"] / n,
            "mean_absolute_onset_error_frames": s["onset_error_sum"] / n,
            "missing_onset_rate": s["missing_onsets"] / n,
            "silent_false_positive_rate": s["silent_false"] / max(s["silent_total"], 1),
            "active_frame_accuracy": s["active_correct"] / max(s["active_total"], 1)}


@torch.no_grad()
def evaluate(model, device, seed, count, gain, use_plastic):
    model.eval()
    rng = random.Random(seed)
    groups = defaultdict(group_stats)
    examples = []
    for index in range(count):
        x, target, mask, items = batch_examples(rng, 1, 3, 6, device)
        source, gold, words = items[0]
        logits = model(x, gain, rng.randint(2, 5), use_plastic=use_plastic)
        predicted = logits.argmax(-1)[0, :len(gold)].tolist()
        guessed_words = runs(predicted)
        repeated = len(set(words)) < len(words)
        groups_for_item = ("overall", f"length_{len(words)}", "repeated" if repeated else "distinct")
        for group in groups_for_item:
            s = groups[group]
            s["n"] += 1
            s["sequence_matches"] += int(guessed_words == words)
            s["frame_matches"] += int(predicted == gold)
            s["frame_correct"] += sum(a == b for a, b in zip(predicted, gold))
            s["frame_total"] += len(gold)
            s["edit_sum"] += edit_distance(guessed_words, words) / len(words)
            gold_start, guess_start = onset(gold), onset(predicted)
            if guess_start is None:
                s["missing_onsets"] += 1
                s["onset_error_sum"] += len(gold)
            else:
                offset = guess_start - gold_start
                s["early_onsets"] += int(offset < 0)
                s["late_onsets"] += int(offset > 0)
                s["onset_error_sum"] += abs(offset)
            for a, b in zip(predicted, gold):
                if b == 0:
                    s["silent_total"] += 1
                    s["silent_false"] += int(a != 0)
                else:
                    s["active_total"] += 1
                    s["active_correct"] += int(a == b)
        if index < 5:
            examples.append({"words": words, "predicted_words": guessed_words,
                             "input_frames": source, "target_frames": gold,
                             "predicted_frames": predicted,
                             "target_onset": onset(gold), "predicted_onset": onset(predicted)})
    return {key: finalize(value) for key, value in groups.items()}, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("plastic", "recurrent"))
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--eval", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    use_plastic = args.variant == "plastic"
    random.seed(3030)
    torch.manual_seed(3030)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StreamMemory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = random.Random(3030)
    total = 2 if args.smoke else args.steps
    history = []
    print(json.dumps({"variant": args.variant, "device": str(device), "steps": total}), flush=True)
    for step in range(1, total + 1):
        model.train()
        if step <= total // 3:
            lo, hi = 2, 3
        elif step <= 2 * total // 3:
            lo, hi = 2, 4
        else:
            lo, hi = 3, 6
        x, target, mask, _ = batch_examples(rng, args.batch, lo, hi, device)
        gain = rng.choice((0.0, 0.15, 0.35))
        logits = model(x, gain, rng.randint(2, 5), use_plastic=use_plastic)
        loss = training_loss(logits, target, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == total:
            row = {"step": step, "loss": float(loss.item()), "min_words": lo, "max_words": hi}
            if step % 300 == 0 or step == total:
                validation, _ = evaluate(model, device, 4141, 5 if args.smoke else 50,
                                         0.15, use_plastic)
                row["validation_sequence_exact"] = validation["overall"]["sequence_exact"]
                row["validation_full_frame_exact"] = validation["overall"]["full_frame_exact"]
            history.append(row)
            with (args.out / "train_log.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("step", "loss", "min_words", "max_words",
                                                         "validation_sequence_exact", "validation_full_frame_exact"))
                writer.writeheader()
                writer.writerows(history)
            print(json.dumps(row), flush=True)
    results = {"variant": args.variant, "device": str(device), "steps": total,
               "conditions": {}}
    for gain in (0.0, 0.15, 0.35):
        metrics, examples = evaluate(model, device, 5151, 3 if args.smoke else args.eval,
                                     gain, use_plastic)
        results["conditions"][str(gain)] = metrics
        if gain == 0.15:
            (args.out / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
            first = examples[0]
            write_wave(args.out / "input.wav", first["input_frames"])
            write_wave(args.out / "target.wav", first["target_frames"])
            write_wave(args.out / "model.wav", first["predicted_frames"])
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(results["conditions"]["0.15"]["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
