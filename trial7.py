"""Trial 7: online event binding in a finite associative state.

This is a representation experiment. The model receives one frame at a time;
the diagnostic reader receives only the final state, never the input history.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TONES = 8
DURATIONS = tuple(range(3, 9))


def make_pair(rng, same_tone=False):
    first = rng.randint(1, TONES)
    second = first if same_tone else rng.choice([k for k in range(1, TONES + 1) if k != first])
    d1, d2 = rng.sample(DURATIONS, 2)
    gap = rng.randint(2, 4)
    frames = [first] * d1 + [0] * gap + [second] * d2 + [0] * 4
    return frames, (first, d1, second, d2)


def batch(rng, size, device, fixed=None):
    examples = [make_pair(rng, same_tone=rng.random() < 0.5) for _ in range(size)]
    if fixed is not None:
        examples = fixed
    length = max(len(frames) for frames, _ in examples)
    x = torch.zeros(len(examples), length, TONES + 1)
    x[:, :, 0] = 1
    for row, (frames, _) in enumerate(examples):
        for t, tone in enumerate(frames):
            x[row, t, 0] = 0
            x[row, t, tone] = 1
    labels = torch.tensor([target for _, target in examples])
    return x.to(device), labels.to(device)


class AssociativeEventMemory(nn.Module):
    def __init__(self, key_dim=32, duration_dim=8, ordered=True):
        super().__init__()
        self.ordered = ordered
        self.key_dim = key_dim
        self.duration_dim = duration_dim
        self.register_buffer("decay", torch.tensor([0.20, 0.45, 0.65, 0.78, 0.86, 0.91, 0.95, 0.975]))
        self.position_start = nn.Parameter(torch.randn(key_dim) * 0.1)
        self.position_step = nn.Linear(key_dim, key_dim, bias=False)
        self.key_map = nn.Linear(key_dim, key_dim)
        self.value_map = nn.Linear(TONES * duration_dim, 64)
        self.read0 = nn.Parameter(torch.randn(key_dim) * 0.1)
        self.read1 = nn.Parameter(torch.randn(key_dim) * 0.1)
        self.reader = nn.Sequential(nn.Linear(64, 96), nn.Tanh(), nn.Linear(96, TONES + 6))

    def initial_state(self, count, device):
        position = self.position_start.unsqueeze(0).expand(count, -1)
        memory = torch.zeros(count, self.key_dim, 64, device=device)
        duration = torch.zeros(count, self.duration_dim, device=device)
        previous = torch.zeros(count, TONES + 1, device=device)
        return position, memory, duration, previous

    def step(self, frame, state):
        position, memory, duration, previous = state
        energy = frame[:, 1:].sum(-1, keepdim=True)
        previous_energy = previous[:, 1:].sum(-1, keepdim=True)
        # Falling acoustic activity supplies a continuous write strength.
        gate = previous_energy * (1 - energy)
        content = previous[:, 1:]
        joint = (content.unsqueeze(-1) * duration.unsqueeze(1)).flatten(1)
        value = torch.tanh(self.value_map(joint))
        key = F.normalize(self.key_map(position), dim=-1)
        memory = 0.999 * memory + gate.unsqueeze(-1) * key.unsqueeze(-1) * value.unsqueeze(1)
        position = position + (gate if self.ordered else gate * 0) * torch.tanh(self.position_step(position))
        duration = energy * (self.decay * duration + (1 - self.decay))
        return position, memory, duration, frame

    def forward(self, x, wait=0):
        state = self.initial_state(x.shape[0], x.device)
        for t in range(x.shape[1]):
            state = self.step(x[:, t], state)
        silence = F.one_hot(torch.zeros(x.shape[0], dtype=torch.long, device=x.device), TONES + 1).float()
        for _ in range(wait):
            state = self.step(silence, state)
        memory = state[1]
        keys = (F.normalize(self.read0, dim=0), F.normalize(self.read1, dim=0))
        reads = [torch.einsum("bkv,k->bv", memory, key) for key in keys]
        predictions = [self.reader(read) for read in reads]
        return predictions, memory


def losses(predictions, labels):
    total = 0
    for i, pred in enumerate(predictions):
        tone = labels[:, 2 * i] - 1
        duration = labels[:, 2 * i + 1] - 3
        total = total + F.cross_entropy(pred[:, :TONES], tone)
        total = total + F.cross_entropy(pred[:, TONES:], duration)
    return total / 4


@torch.no_grad()
def evaluate(model, device, seed=7117, count=600):
    model.eval()
    rng = random.Random(seed)
    result = {}
    for wait in (0, 16, 64):
        tone_ok = duration_ok = joint_ok = same_joint = same_count = swap_ok = 0
        for _ in range(count // 50):
            x, labels = batch(rng, 50, device)
            preds, _ = model(x, wait)
            tone = torch.stack([p[:, :TONES].argmax(-1) + 1 for p in preds], 1)
            duration = torch.stack([p[:, TONES:].argmax(-1) + 3 for p in preds], 1)
            target_tone = labels[:, [0, 2]]
            target_duration = labels[:, [1, 3]]
            tone_ok += (tone == target_tone).all(1).sum().item()
            duration_ok += (duration == target_duration).all(1).sum().item()
            correct = (tone == target_tone).all(1) & (duration == target_duration).all(1)
            joint_ok += correct.sum().item()
            same = labels[:, 0] == labels[:, 2]
            same_joint += (correct & same).sum().item()
            same_count += same.sum().item()
        # Explicit counterfactual pairs: frequency order and total voiced time are identical.
        pairs = []
        for _ in range(100):
            frames, (a, da, b, db) = make_pair(rng, same_tone=rng.random() < 0.5)
            gap = 3
            pairs.append(([a] * da + [0] * gap + [b] * db + [0] * 4, (a, da, b, db)))
            pairs.append(([a] * db + [0] * gap + [b] * da + [0] * 4, (a, db, b, da)))
        x, labels = batch(rng, len(pairs), device, fixed=pairs)
        preds, _ = model(x, wait)
        duration = torch.stack([p[:, TONES:].argmax(-1) + 3 for p in preds], 1)
        target = labels[:, [1, 3]]
        swap_ok = ((duration[0::2] == target[0::2]).all(1) &
                   (duration[1::2] == target[1::2]).all(1)).sum().item()
        result[str(wait)] = {"n": count, "tone_pair_accuracy": tone_ok / count,
                             "duration_pair_accuracy": duration_ok / count,
                             "joint_pair_accuracy": joint_ok / count,
                             "same_tone_joint_accuracy": same_joint / max(1, same_count),
                             "counterfactual_pair_both_correct": swap_ok / 100}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7117)
    random.seed(7117)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"ordered": AssociativeEventMemory(ordered=True).to(device),
              "pooled": AssociativeEventMemory(ordered=False).to(device)}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=0.001)
                  for name, model in models.items()}
    rng = random.Random(7117)
    logs = []
    steps = 2 if args.smoke else args.steps
    for step in range(1, steps + 1):
        x, labels = batch(rng, args.batch, device)
        for name, model in models.items():
            model.train()
            pred, _ = model(x)
            loss = losses(pred, labels)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            if step == 1 or step % 100 == 0 or step == steps:
                logs.append({"step": step, "model": name, "loss": float(loss.item())})
        if step % 100 == 0 or step == steps:
            with (args.out / "train_log.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("step", "model", "loss"))
                writer.writeheader()
                writer.writerows(logs)
            print(json.dumps(logs[-2:]), flush=True)
    report = {name: evaluate(model, device, count=50 if args.smoke else 600)
              for name, model in models.items()}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, model in models.items():
        torch.save(model.state_dict(), args.out / f"{name}.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
