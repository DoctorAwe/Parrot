"""Colab trial 2: curriculum, diagnostics, and memory ablation.

Synthetic frequency-frame task only; no claim of natural speech performance.
"""

import argparse
import csv
import json
import random
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trial1 import N_TONES, example, frame_to_wave


def batch_examples(rng, count, lo, hi, device):
    items = [example(rng, lo, hi) for _ in range(count)]
    length = max(len(item[1]) for item in items)
    x = torch.zeros(count, length, N_TONES + 1, device=device)
    x[:, :, 0] = 1.0
    target = torch.zeros(count, length, dtype=torch.long, device=device)
    mask = torch.zeros(count, length, device=device)
    for b, (source, gold, _) in enumerate(items):
        idx = torch.arange(len(source), device=device)
        x[b, idx, 0] = 0.0
        x[b, idx, torch.tensor(source, device=device)] = 1.0
        target[b, :len(gold)] = torch.tensor(gold, device=device)
        mask[b, :len(gold)] = 1.0
    return x, target, mask, items


class StreamMemory(nn.Module):
    def __init__(self, width=64, plastic=16):
        super().__init__()
        self.width, self.plastic = width, plastic
        self.encoder = nn.Linear(N_TONES + 1, width)
        self.recurrent = nn.ModuleList([nn.Linear(width * 3, width) for _ in range(3)])
        self.keep_logit = nn.Parameter(torch.tensor([0.0, 2.0, 4.0]))
        self.key = nn.Linear(width * 3, plastic)
        self.value = nn.Linear(width * 3, plastic)
        self.query = nn.Linear(width * 3, plastic)
        self.write_gate = nn.Linear(width * 3, 1)
        self.p_keep_logit = nn.Parameter(torch.tensor(3.5))
        self.output = nn.Sequential(nn.Linear(width * 3 + plastic, width), nn.Tanh(),
                                    nn.Linear(width, N_TONES + 1))

    def forward(self, external, feedback_gain, delay, use_plastic=True, clear_h_at=None):
        b, steps, _ = external.shape
        h = [external.new_zeros(b, self.width) for _ in range(3)]
        p = external.new_zeros(b, self.plastic, self.plastic)
        # Bounded environment propagation queue, outside the learned memory.
        environment = [external.new_zeros(b, N_TONES + 1) for _ in range(delay)]
        out = []
        for t in range(steps):
            if clear_h_at is not None and t == clear_h_at:
                h = [torch.zeros_like(v) for v in h]
            heard = external[:, t] + feedback_gain * environment.pop(0)
            old = torch.cat(h, dim=-1)
            drive = self.encoder(heard)
            keep = torch.sigmoid(self.keep_logit)
            h = [keep[i] * h[i] + (1 - keep[i]) * torch.tanh(drive + self.recurrent[i](old))
                 for i in range(3)]
            current = torch.cat(h, dim=-1)
            if use_plastic:
                # Continuous acoustic-energy gating: received self-sound may also write.
                energy = heard[:, 1:].sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
                gate = torch.sigmoid(self.write_gate(current)) * energy
                key = F.normalize(torch.tanh(self.key(current)), dim=-1)
                value = torch.tanh(self.value(current))
                update = torch.bmm(value.unsqueeze(2), key.unsqueeze(1))
                p = torch.sigmoid(self.p_keep_logit) * p + gate.unsqueeze(2) * update
                p = torch.tanh(p)
                read = torch.bmm(p, torch.tanh(self.query(current)).unsqueeze(2)).squeeze(2)
            else:
                read = current.new_zeros(b, self.plastic)
            logits = self.output(torch.cat((current, read), dim=-1))
            out.append(logits)
            acoustic = F.softmax(logits, dim=-1)
            environment.append(torch.cat((acoustic[:, :1] * 0, acoustic[:, 1:]), dim=-1))
        return torch.stack(out, dim=1)


def training_loss(logits, target, mask):
    frame_loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    weight = torch.where(target == 0, 1.0, 3.0)
    return (frame_loss * weight * mask).sum() / (weight * mask).sum()


def runs(frames):
    result, prev = [], 0
    for frame in frames:
        frame = int(frame)
        if frame and frame != prev:
            result.append(frame)
        prev = frame
    return result


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        row = [i]
        for j, other in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + int(token != other)))
        prev = row
    return prev[-1]


def onset(frames):
    return next((i for i, v in enumerate(frames) if v), None)


@torch.no_grad()
def evaluate(model, device, seed, count, gain, lo, hi, use_plastic=True, clear_h=False):
    model.eval()
    rng = random.Random(seed)
    exact = 0
    distances = []
    onset_errors = []
    active_ok = active_n = silent_false = silent_n = 0
    examples = []
    for index in range(count):
        x, target, mask, items = batch_examples(rng, 1, lo, hi, device)
        source, gold, words = items[0]
        reset_at = None
        if clear_h:
            reset_at = next((i for i in range(1, len(source)) if source[i] == 0 and all(v == 0 for v in source[i:])), None)
        logits = model(x, gain, rng.randint(2, 5), use_plastic=use_plastic, clear_h_at=reset_at)
        predicted = logits.argmax(-1)[0, :len(gold)].tolist()
        decoded = runs(predicted)
        exact += int(decoded == words)
        distances.append(edit_distance(decoded, words) / len(words))
        if onset(gold) is not None and onset(predicted) is not None:
            onset_errors.append(abs(onset(predicted) - onset(gold)))
        for guess, truth in zip(predicted, gold):
            if truth:
                active_n += 1
                active_ok += int(guess == truth)
            else:
                silent_n += 1
                silent_false += int(guess != 0)
        if index < 5:
            examples.append({"source_words": words, "predicted_words": decoded,
                             "target_onset_frame": onset(gold), "predicted_onset_frame": onset(predicted),
                             "source_frames": source, "target_frames": gold, "predicted_frames": predicted})
    return {"sequence_exact": exact / count,
            "normalized_edit_distance": sum(distances) / count,
            "active_frame_accuracy": active_ok / max(1, active_n),
            "silent_false_positive_rate": silent_false / max(1, silent_n),
            "mean_onset_error_frames": sum(onset_errors) / len(onset_errors) if onset_errors else None}, examples


def write_wave(path, frames):
    pcm = (np.clip(frame_to_wave(frames), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--eval", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("trial2_results"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(2028)
    np.random.seed(2028)
    torch.manual_seed(2028)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    model = StreamMemory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = random.Random(2028)
    steps = 2 if args.smoke else args.steps
    history = []
    for step in range(1, steps + 1):
        model.train()
        if step <= steps // 3:
            lo, hi = 2, 3
        elif step <= 2 * steps // 3:
            lo, hi = 2, 4
        else:
            lo, hi = 3, 6
        x, target, mask, _ = batch_examples(rng, args.batch, lo, hi, device)
        gain = rng.choice((0.0, 0.15, 0.35))
        logits = model(x, gain, rng.randint(2, 5))
        loss = training_loss(logits, target, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            record = {"step": step, "training_loss": float(loss.item()), "train_min": lo, "train_max": hi}
            if step % 300 == 0 or step == steps:
                val, _ = evaluate(model, device, 4040, 12 if args.smoke else 30, 0.15, 3, 6)
                record["validation_exact"] = val["sequence_exact"]
                record["validation_edit"] = val["normalized_edit_distance"]
            history.append(record)
            print(json.dumps(record), flush=True)
            # Keep partial progress on Drive if Colab disconnects mid-run.
            with (args.out / "train_log.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["step", "training_loss", "train_min", "train_max",
                                                          "validation_exact", "validation_edit"])
                writer.writeheader()
                writer.writerows(history)
    report = {"device": str(device), "steps": steps, "conditions": {}}
    for gain in (0.0, 0.15, 0.35):
        metrics, examples = evaluate(model, device, 9090, 3 if args.smoke else args.eval, gain, 3, 6)
        report["conditions"][str(gain)] = metrics
        if gain == 0.15:
            (args.out / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
            first = examples[0]
            write_wave(args.out / "input.wav", first["source_frames"])
            write_wave(args.out / "target.wav", first["target_frames"])
            write_wave(args.out / "model.wav", first["predicted_frames"])
    if not args.smoke:
        for label, kwargs in (("without_plastic", {"use_plastic": False}),
                              ("cleared_activity", {"clear_h": True})):
            metrics, _ = evaluate(model, device, 9090, args.eval, 0.15, 3, 6, **kwargs)
            report["conditions"][label] = metrics
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
