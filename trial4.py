"""Trial 4: train with shift-invariant frame loss after the last voiced input.

Two variants use identical architecture, seeds, and original-input templates.
Only their training loss differs: soft best-shift vs fixed random onset.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from trial2 import StreamMemory, batch_examples, onset, runs, write_wave
from trial3_align_eval import aligned_score


def candidate_targets(items, padded_length, device, objective):
    """Build a reference sequence for every permitted start, with no mode bit."""
    all_starts = []
    for source, target, _ in items:
        last_input = max(i for i, value in enumerate(source) if value != 0)
        template_length = last_input + 1
        if objective == "aligned":
            starts = list(range(last_input + 1, len(target) - template_length + 1))
        else:
            starts = [onset(target)]
        if not starts or max(starts) + template_length > len(target):
            raise ValueError("Input template does not fit in output window")
        all_starts.append(starts)
    max_candidates = max(map(len, all_starts))
    batch_size = len(items)
    labels = torch.zeros(batch_size, max_candidates, padded_length, dtype=torch.long, device=device)
    frame_mask = torch.zeros(batch_size, 1, padded_length, device=device)
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool, device=device)
    for b, ((source, target, _), starts) in enumerate(zip(items, all_starts)):
        last_input = max(i for i, value in enumerate(source) if value != 0)
        template = torch.tensor(source[:last_input + 1], dtype=torch.long, device=device)
        frame_mask[b, 0, :len(target)] = 1.0
        for k, start in enumerate(starts):
            labels[b, k, start:start + len(template)] = template
            candidate_mask[b, k] = True
    return labels, frame_mask, candidate_mask


def sequence_loss(logits, items, objective, temperature=0.08):
    labels, frame_mask, candidate_mask = candidate_targets(
        items, logits.shape[1], logits.device, objective)
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = torch.gather(
        log_probs.unsqueeze(1).expand(-1, labels.shape[1], -1, -1),
        dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    weights = torch.where(labels == 0, 1.0, 3.0) * frame_mask
    costs = -(chosen * weights).sum(-1) / weights.sum(-1).clamp_min(1)
    if objective == "fixed":
        return costs[:, 0].mean()
    costs = costs.masked_fill(~candidate_mask, float("inf"))
    count = candidate_mask.sum(-1)
    # Differentiable best-start approximation; equals hard minimum as tau -> 0.
    soft_best = -temperature * (torch.logsumexp(-costs / temperature, dim=-1)
                                - count.log())
    return soft_best.mean()


@torch.no_grad()
def evaluate(model, device, seed, count, gain):
    model.eval()
    rng = random.Random(seed)
    grouped = {}
    examples = []
    for index in range(count):
        x, _, _, items = batch_examples(rng, 1, 3, 6, device)
        source, target, words = items[0]
        logits = model(x, gain, rng.randint(2, 5))
        predicted = logits.argmax(-1)[0, :len(target)].tolist()
        aligned = aligned_score(source, target, predicted)
        repeated = len(set(words)) < len(words)
        for name in ("overall", f"length_{len(words)}", "repeated" if repeated else "distinct"):
            if name not in grouped:
                grouped[name] = {"n": 0, "safe": 0, "word_exact": 0, "frame_exact": 0,
                                 "input_frame_sum": 0.0, "input_active_sum": 0.0}
            g = grouped[name]
            g["n"] += 1
            g["safe"] += int(aligned["no_interruption"])
            g["word_exact"] += int(aligned["no_interruption"] and runs(predicted) == words)
            g["frame_exact"] += int(aligned["input_full_frame_exact"])
            g["input_frame_sum"] += aligned["input_frame_accuracy"]
            g["input_active_sum"] += aligned["input_active_frame_accuracy"]
        if index < 5:
            examples.append({"words": words, "predicted_words": runs(predicted), **aligned,
                             "input_frames": source, "predicted_frames": predicted})
    result = {}
    for name, g in grouped.items():
        n = g["n"]
        result[name] = {"n": n, "no_interruption_rate": g["safe"] / n,
                        "sequence_exact": g["word_exact"] / n,
                        "input_aligned_full_frame_exact": g["frame_exact"] / n,
                        "input_aligned_frame_accuracy": g["input_frame_sum"] / n,
                        "input_aligned_active_frame_accuracy": g["input_active_sum"] / n}
    return result, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("aligned", "fixed"), required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--eval", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(4040)
    torch.manual_seed(4040)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StreamMemory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = random.Random(4040)
    total = 2 if args.smoke else args.steps
    history = []
    print(json.dumps({"objective": args.objective, "device": str(device),
                      "steps": total}), flush=True)
    for step in range(1, total + 1):
        model.train()
        if step <= total // 3:
            lo, hi = 2, 3
        elif step <= 2 * total // 3:
            lo, hi = 2, 4
        else:
            lo, hi = 3, 6
        x, _, _, items = batch_examples(rng, args.batch, lo, hi, device)
        gain = rng.choice((0.0, 0.15, 0.35))
        logits = model(x, gain, rng.randint(2, 5))
        loss = sequence_loss(logits, items, args.objective)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == total:
            row = {"step": step, "loss": float(loss.item()), "min_words": lo, "max_words": hi}
            if step % 300 == 0 or step == total:
                val, _ = evaluate(model, device, 4242, 5 if args.smoke else 50, 0.15)
                row["validation_sequence_exact"] = val["overall"]["sequence_exact"]
                row["validation_frame_accuracy"] = val["overall"]["input_aligned_frame_accuracy"]
            history.append(row)
            with (args.out / "train_log.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("step", "loss", "min_words", "max_words",
                                                         "validation_sequence_exact", "validation_frame_accuracy"))
                writer.writeheader()
                writer.writerows(history)
            print(json.dumps(row), flush=True)
    result = {"objective": args.objective, "device": str(device), "steps": total,
              "conditions": {}}
    for gain in (0.0, 0.15, 0.35):
        metrics, examples = evaluate(model, device, 5252, 3 if args.smoke else args.eval, gain)
        result["conditions"][str(gain)] = metrics
        if gain == 0.15:
            (args.out / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
            first = examples[0]
            write_wave(args.out / "input.wav", first["input_frames"])
            write_wave(args.out / "model.wav", first["predicted_frames"])
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(result["conditions"]["0.15"]["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
