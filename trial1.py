"""Colab trial 1: causal frequency-frame echo with acoustic feedback.

This is a synthetic sequence-memory experiment, not natural-language speech.
"""

import argparse
import json
import math
import random
import wave
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


N_TONES = 8
FRAME_S = 0.04
FRAME_SAMPLES = 320  # 8 kHz, 40 ms


def example(rng, min_words=3, max_words=6):
    """Return one frequency-frame input, desired output, and source sequence."""
    words = [rng.randrange(1, N_TONES + 1) for _ in range(rng.randint(min_words, max_words))]
    source = []
    for i, word in enumerate(words):
        source.extend([word, word])
        if i + 1 < len(words):
            source.extend([0] * rng.randint(1, 2))
    source.extend([0] * rng.randint(5, 8))
    repeat = []
    for i, word in enumerate(words):
        repeat.extend([word, word])
        if i + 1 < len(words):
            repeat.append(0)
    target = [0] * len(source) + repeat + [0] * 5
    source.extend([0] * (len(target) - len(source)))
    return source, target, words


def make_batch(batch_size, rng, device, fixed=None):
    items = [example(rng) if fixed is None else fixed for _ in range(batch_size)]
    length = max(len(x[1]) for x in items)
    x = torch.zeros(batch_size, length, N_TONES + 1, device=device)
    y = torch.zeros(batch_size, length, dtype=torch.long, device=device)
    mask = torch.zeros(batch_size, length, device=device)
    for b, (source, target, _) in enumerate(items):
        x[b, torch.arange(len(source), device=device), torch.tensor(source, device=device)] = 1.0
        y[b, : len(target)] = torch.tensor(target, device=device)
        mask[b, : len(target)] = 1.0
    return x, y, mask, items


class MultiTimescaleMemory(nn.Module):
    """No language labels, phase switch, output-copy input, or recording buffer."""

    def __init__(self, width=64, plastic=16):
        super().__init__()
        self.width = width
        self.plastic = plastic
        self.input_map = nn.Linear(N_TONES + 1, width)
        self.recurrent = nn.ModuleList([nn.Linear(width * 3, width) for _ in range(3)])
        self.write_key = nn.Linear(width * 3, plastic)
        self.write_value = nn.Linear(width * 3, plastic)
        self.read_query = nn.Linear(width * 3, plastic)
        self.output_map = nn.Sequential(
            nn.Linear(width * 3 + plastic, width), nn.Tanh(), nn.Linear(width, N_TONES + 1)
        )
        # Different learned retention ranges, rather than named linguistic units.
        self.logit_keep = nn.Parameter(torch.tensor([1.0, 2.5, 4.0]))
        self.plastic_keep = nn.Parameter(torch.tensor(3.0))
        self.plastic_gain = nn.Parameter(torch.tensor(-2.0))

    def forward(self, external, feedback_gain=0.0, delay=3):
        batch, steps, _ = external.shape
        h = [external.new_zeros(batch, self.width) for _ in range(3)]
        p = external.new_zeros(batch, self.plastic, self.plastic)
        outputs = []
        # Environment queue holds only the bounded propagation delay.
        env = [external.new_zeros(batch, N_TONES + 1) for _ in range(delay)]
        for t in range(steps):
            heard = external[:, t] + feedback_gain * env.pop(0)
            h_cat = torch.cat(h, dim=-1)
            drive = self.input_map(heard)
            keep = torch.sigmoid(self.logit_keep)
            h = [
                keep[i] * h[i] + (1 - keep[i]) * torch.tanh(drive + self.recurrent[i](h_cat))
                for i in range(3)
            ]
            h_cat = torch.cat(h, dim=-1)
            key = torch.tanh(self.write_key(h_cat))
            value = torch.tanh(self.write_value(h_cat))
            update = torch.bmm(value.unsqueeze(2), key.unsqueeze(1)) / self.plastic
            p = torch.sigmoid(self.plastic_keep) * p + torch.sigmoid(self.plastic_gain) * update
            read = torch.bmm(p, torch.tanh(self.read_query(h_cat)).unsqueeze(2)).squeeze(2)
            logits = self.output_map(torch.cat((h_cat, read), dim=-1))
            outputs.append(logits)
            # Echo of the generated sound, not a second internal model input.
            acoustic = F.softmax(logits, dim=-1)
            acoustic = torch.cat((acoustic[:, :1] * 0, acoustic[:, 1:]), dim=-1)
            env.append(acoustic)
        return torch.stack(outputs, dim=1)


def loss_fn(logits, target, mask):
    loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    weight = torch.where(target == 0, 1.0, 2.5)
    return (loss * weight * mask).sum() / (weight * mask).sum()


def word_runs(frames):
    """Collapse continuous runs, ignoring silence; used for evaluation only."""
    result, last = [], 0
    for frame in frames:
        if frame != 0 and frame != last:
            result.append(int(frame))
        last = int(frame)
    return result


def frame_to_wave(frames):
    """Render model's frequency-bin decisions to an audible 8 kHz waveform."""
    pieces = []
    phase = 0.0
    for index in frames:
        t = np.arange(FRAME_SAMPLES) / 8000.0
        if index:
            frequency = 300.0 + 115.0 * (int(index) - 1)
            segment = 0.18 * np.sin(2 * np.pi * frequency * t + phase)
            phase = float((phase + 2 * np.pi * frequency * FRAME_S) % (2 * np.pi))
        else:
            segment = np.zeros(FRAME_SAMPLES)
        pieces.append(segment)
    return np.concatenate(pieces)


def write_wave(path, frames):
    samples = (np.clip(frame_to_wave(frames), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(samples.tobytes())


@torch.no_grad()
def evaluate(model, seed, device, gain, n=100):
    model.eval()
    rng = random.Random(seed)
    exact, active_correct, active_total = 0, 0, 0
    case = None
    for _ in range(n):
        x, y, mask, items = make_batch(1, rng, device)
        logits = model(x, feedback_gain=gain, delay=rng.randint(2, 5))
        pred = logits.argmax(-1)[0, : int(mask.sum().item())].tolist()
        gold = y[0, : len(pred)].tolist()
        exact += int(word_runs(pred) == items[0][2])
        active_correct += sum(int(a == b) for a, b in zip(pred, gold) if b != 0)
        active_total += sum(b != 0 for b in gold)
        if case is None:
            case = (items[0], pred)
    return {"sequence_exact": exact / n, "active_frame_accuracy": active_correct / max(active_total, 1)}, case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("trial1_results"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    model = MultiTimescaleMemory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = random.Random(1234)
    steps = 2 if args.smoke else args.steps
    for step in range(1, steps + 1):
        model.train()
        x, y, mask, _ = make_batch(args.batch, rng, device)
        gain = rng.choice([0.0, 0.15, 0.35])
        logits = model(x, feedback_gain=gain, delay=rng.randint(2, 5))
        loss = loss_fn(logits, y, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            print(f"step={step} loss={loss.item():.4f}", flush=True)
    results = {}
    for gain in (0.0, 0.15, 0.35):
        metrics, case = evaluate(model, 9876, device, gain, n=args.eval if not args.smoke else 2)
        results[str(gain)] = metrics
        if gain == 0.15:
            source, target, words = case[0]
            pred = case[1]
            results["example"] = {"words": words, "predicted_runs": word_runs(pred)}
            write_wave(args.out / "heard_input.wav", source)
            write_wave(args.out / "target_output.wav", target)
            write_wave(args.out / "model_output.wav", pred)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
