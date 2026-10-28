"""Trial 8: variable-length event binding and sequence capacity."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TONES = 8
SCALES = 8
KEY = 48
VALUE = 96


def example(rng, length, force_gap=None):
    tones, durations, gaps = [], [], []
    for i in range(length):
        gap = rng.randint(0, 3) if force_gap is None else force_gap
        if i and gap == 0:
            choices = [k for k in range(1, TONES + 1) if k != tones[-1]]
            tone = rng.choice(choices)
        elif i and rng.random() < 0.45:
            tone = tones[-1]
        else:
            tone = rng.randint(1, TONES)
        tones.append(tone)
        durations.append(rng.randint(3, 8))
        if i:
            gaps.append(gap)
    frames = []
    for i, (tone, duration) in enumerate(zip(tones, durations)):
        if i:
            frames += [0] * gaps[i - 1]
        frames += [tone] * duration
    frames += [0] * 4
    return frames, tones, durations, gaps


def make_batch(rng, size, device, min_length, max_length, force_gap=None):
    items = [example(rng, rng.randint(min_length, max_length), force_gap)
             for _ in range(size)]
    steps = max(len(item[0]) for item in items)
    x = torch.zeros(size, steps, TONES + 1)
    x[:, :, 0] = 1
    tones = torch.zeros(size, max_length, dtype=torch.long)
    durations = torch.zeros_like(tones)
    lengths = torch.zeros(size, dtype=torch.long)
    for b, (frames, tone_list, duration_list, _) in enumerate(items):
        for t, tone in enumerate(frames):
            x[b, t, 0] = 0
            x[b, t, tone] = 1
        n = len(tone_list)
        lengths[b] = n
        tones[b, :n] = torch.tensor(tone_list)
        durations[b, :n] = torch.tensor(duration_list)
    return x.to(device), tones.to(device), durations.to(device), lengths.to(device), items


class EventMemory(nn.Module):
    def __init__(self, position_kind):
        super().__init__()
        self.position_kind = position_kind
        self.register_buffer("decay", torch.tensor([0.20, 0.45, 0.65, 0.78,
                                                     0.86, 0.91, 0.95, 0.975]))
        self.position_start = nn.Parameter(torch.randn(KEY) * 0.1)
        self.position_step = nn.Linear(KEY, KEY, bias=False)
        phases = torch.linspace(0.12, 1.57, KEY // 2)
        self.register_buffer("rotation_cos", phases.cos())
        self.register_buffer("rotation_sin", phases.sin())
        self.key_map = nn.Linear(KEY, KEY)
        self.value_map = nn.Linear(TONES * SCALES, VALUE)
        self.reader = nn.Sequential(nn.Linear(VALUE, 128), nn.Tanh(),
                                    nn.Linear(128, TONES + 6))

    def advance(self, position):
        if self.position_kind == "learned":
            return F.normalize(position + torch.tanh(self.position_step(position)), dim=-1)
        pairs = position.reshape(position.shape[0], KEY // 2, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack((self.rotation_cos * a - self.rotation_sin * b,
                               self.rotation_sin * a + self.rotation_cos * b), -1)
        return F.normalize(rotated.flatten(1), dim=-1)

    def initial_position(self, size):
        return F.normalize(self.position_start, dim=0).unsqueeze(0).expand(size, -1)

    def forward(self, x, wait=0, slots=8):
        b = x.shape[0]
        position = self.initial_position(b)
        memory = x.new_zeros(b, KEY, VALUE)
        duration = x.new_zeros(b, SCALES)
        previous = x.new_zeros(b, TONES + 1)
        for t in range(x.shape[1]):
            frame = x[:, t]
            energy = frame[:, 1:].sum(-1, keepdim=True)
            old_energy = previous[:, 1:].sum(-1, keepdim=True)
            same = (frame[:, 1:] * previous[:, 1:]).sum(-1, keepdim=True)
            change = old_energy * (1 - same)
            joint = (previous[:, 1:].unsqueeze(-1) * duration.unsqueeze(1)).flatten(1)
            value = torch.tanh(self.value_map(joint))
            key = F.normalize(self.key_map(position), dim=-1)
            memory = 0.999 * memory + change.unsqueeze(-1) * key.unsqueeze(-1) * value.unsqueeze(1)
            position = F.normalize(position + change * (self.advance(position) - position), dim=-1)
            duration = energy * (same * (self.decay * duration + 1 - self.decay) +
                                 (1 - same) * (1 - self.decay))
            previous = frame
        memory = memory * (0.999 ** wait)
        query = self.initial_position(b)
        outputs = []
        for _ in range(slots):
            key = F.normalize(self.key_map(query), dim=-1)
            read = torch.einsum("bkv,bk->bv", memory, key)
            outputs.append(self.reader(read))
            query = self.advance(query)
        return torch.stack(outputs, 1), memory


def loss_fn(outputs, tones, durations, lengths):
    mask = torch.arange(outputs.shape[1], device=outputs.device)[None, :] < lengths[:, None]
    tone_loss = F.cross_entropy(outputs[..., :TONES][mask], tones[mask] - 1)
    duration_loss = F.cross_entropy(outputs[..., TONES:][mask], durations[mask] - 3)
    return (tone_loss + duration_loss) / 2


@torch.no_grad()
def evaluate(model, device, seed):
    model.eval()
    rng = random.Random(seed)
    report = {}
    for n in (2, 4, 6, 7, 8):
        for gap_name, force_gap in (("mixed", None), ("direct", 0)):
            x, tones, durations, lengths, items = make_batch(
                rng, 200, device, n, n, force_gap)
            for wait in (0, 64):
                outputs, _ = model(x, wait=wait, slots=n)
                pred_tone = outputs[..., :TONES].argmax(-1) + 1
                pred_duration = outputs[..., TONES:].argmax(-1) + 3
                tone_ok = pred_tone == tones
                duration_ok = pred_duration == durations
                joint = tone_ok & duration_ok
                repeated = torch.tensor([len(set(item[1])) < n for item in items],
                                        device=device)
                report[f"n{n}_{gap_name}_wait{wait}"] = {
                    "n": 200,
                    "tone_per_position": tone_ok.float().mean(0).tolist(),
                    "duration_per_position": duration_ok.float().mean(0).tolist(),
                    "joint_per_position": joint.float().mean(0).tolist(),
                    "joint_all": joint.all(1).float().mean().item(),
                    "repeated_joint_all": joint.all(1)[repeated].float().mean().item()
                    if repeated.any() else None,
                }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(8181)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {kind: EventMemory(kind).to(device) for kind in ("learned", "rotation")}
    optimizers = {kind: torch.optim.AdamW(model.parameters(), lr=0.0007)
                  for kind, model in models.items()}
    rng = random.Random(8181)
    logs = []
    steps = 2 if args.smoke else args.steps
    for step in range(1, steps + 1):
        x, tones, durations, lengths, _ = make_batch(rng, args.batch, device, 2, 6)
        for kind, model in models.items():
            model.train()
            outputs, _ = model(x, slots=6)
            loss = loss_fn(outputs, tones, durations, lengths)
            optimizers[kind].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizers[kind].step()
            if step == 1 or step % 100 == 0 or step == steps:
                logs.append({"step": step, "model": kind, "loss": loss.item()})
        if step % 100 == 0 or step == steps:
            with (args.out / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=("step", "model", "loss"))
                writer.writeheader()
                writer.writerows(logs)
            print(json.dumps(logs[-2:]), flush=True)
    report = {kind: evaluate(model, device, 8818) for kind, model in models.items()}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for kind, model in models.items():
        torch.save(model.state_dict(), args.out / f"{kind}.pt")
    print(json.dumps({kind: {k: v["joint_all"] for k, v in result.items()}
                      for kind, result in report.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
