"""Trial 10: recurrent neural activity and local short-term synaptic states.

No addressable event memory, position keys, or matrix query is used. The only
runtime states are neuron activity, adaptation, presynaptic traces, and local
synaptic facilitation. A supervised diagnostic head is used only for testing
whether information remains in the neural population after input.
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
MAX_SEGMENTS = 6
DURATION_CLASSES = 6  # 3..8 frames


def sample(rng, length, swap=False):
    tones = [rng.randint(1, TONES)]
    durations = [rng.randint(3, 8)]
    gaps = []
    for _ in range(1, length):
        gap = rng.randint(0, 3)
        choices = list(range(1, TONES + 1)) if gap else [k for k in range(1, TONES + 1) if k != tones[-1]]
        tone = rng.choice(choices)
        tones.append(tone)
        durations.append(rng.randint(3, 8))
        gaps.append(gap)
    if swap and length >= 2:
        durations[0], durations[1] = durations[1], durations[0]
    frames = []
    for i in range(length):
        if i:
            frames.extend([0] * gaps[i - 1])
        frames.extend([tones[i]] * durations[i])
    frames.extend([0] * 5)
    return frames, tones, durations, gaps


def make_batch(rng, count, lo, hi, device, force_pair=False):
    items = [sample(rng, 2 if force_pair else rng.randint(lo, hi)) for _ in range(count)]
    if force_pair:
        paired = []
        for frames, tones, durations, gaps in items:
            a, b = tones
            da, db = durations
            gap = gaps[0]
            if da == db:
                db = 3 if da != 3 else 8
            f1 = [a] * da + [0] * gap + [b] * db + [0] * 5
            f2 = [a] * db + [0] * gap + [b] * da + [0] * 5
            paired.extend(((f1, tones, [da, db], gaps),
                           (f2, tones, [db, da], gaps)))
        items = paired
    b = len(items)
    steps = max(len(item[0]) for item in items)
    x = torch.zeros(b, steps, TONES + 1)
    x[:, :, 0] = 1
    t_targets = torch.zeros(b, MAX_SEGMENTS, dtype=torch.long)
    d_targets = torch.zeros_like(t_targets)
    lengths = torch.zeros(b, dtype=torch.long)
    for row, (frames, tones, durations, _) in enumerate(items):
        for t, tone in enumerate(frames):
            x[row, t, 0] = 0
            x[row, t, tone] = 1
        n = len(tones)
        lengths[row] = n
        t_targets[row, :n] = torch.tensor(tones)
        d_targets[row, :n] = torch.tensor(durations)
    return (x.to(device), t_targets.to(device), d_targets.to(device),
            lengths.to(device), items)


class NeuralDynamics(nn.Module):
    def __init__(self, width=96, plastic=True, seed=1010):
        super().__init__()
        self.width = width
        self.plastic = plastic
        self.sensory = nn.Linear(TONES + 1, width, bias=False)
        self.raw_recurrent = nn.Parameter(torch.randn(width, width) * 0.12)
        self.bias = nn.Parameter(torch.zeros(width))
        self.time_logit = nn.Parameter(torch.linspace(-1.6, 1.6, width))
        self.adapt_gain = nn.Parameter(torch.full((width,), -1.5))
        self.plastic_rate = nn.Parameter(torch.full((width,), -3.5))
        self.readout = nn.Linear(2 * width, MAX_SEGMENTS * (TONES + DURATION_CLASSES))
        generator = torch.Generator().manual_seed(seed)
        mask = (torch.rand(width, width, generator=generator) < 0.15).float()
        mask.fill_diagonal_(1)
        signs = torch.ones(width)
        signs[int(width * 0.8):] = -1
        self.register_buffer("connectivity", mask)
        self.register_buffer("source_sign", signs)

    def initial_state(self, batch, device):
        h = torch.zeros(batch, self.width, device=device)
        adaptation = torch.zeros_like(h)
        pretrace = torch.zeros_like(h)
        facilitation = torch.zeros(batch, self.width, self.width, device=device)
        return h, adaptation, pretrace, facilitation

    def step(self, frame, state):
        h, adaptation, pretrace, facilitation = state
        # Dale-like fixed sign for each source neuron. Facilitation only changes
        # an existing local synapse's strength, never its sign or address.
        weights = F.softplus(self.raw_recurrent) * self.connectivity
        weights = weights * self.source_sign[None, :] / 5.0
        effective = weights[None, :, :] * (1 + facilitation)
        current = torch.bmm(effective, h.unsqueeze(-1)).squeeze(-1)
        drive = self.sensory(frame) + current + self.bias - F.softplus(self.adapt_gain) * adaptation
        rate = torch.sigmoid(self.time_logit)[None, :]
        new_h = (1 - rate) * h + rate * torch.tanh(drive)
        new_adaptation = 0.94 * adaptation + 0.06 * new_h.abs()
        new_pretrace = 0.70 * pretrace + 0.30 * h
        if self.plastic:
            local_coactivity = new_h.unsqueeze(-1) * new_pretrace.unsqueeze(1)
            local_rate = F.softplus(self.plastic_rate)[None, :, None]
            # Every entry is a local pre/post synaptic state. No key-value read.
            facilitation = torch.clamp(0.985 * facilitation +
                                        local_rate * local_coactivity * self.connectivity,
                                        min=-0.5, max=0.5)
        return new_h, new_adaptation, new_pretrace, facilitation

    def forward(self, x, wait=0):
        state = self.initial_state(x.shape[0], x.device)
        for t in range(x.shape[1]):
            state = self.step(x[:, t], state)
        if wait:
            silence = F.one_hot(torch.zeros(x.shape[0], dtype=torch.long, device=x.device), TONES + 1).float()
            for _ in range(wait):
                state = self.step(silence, state)
        h, adaptation, _, facilitation = state
        diagnostic = self.readout(torch.cat((h, adaptation), -1))
        diagnostic = diagnostic.reshape(x.shape[0], MAX_SEGMENTS, TONES + DURATION_CLASSES)
        return diagnostic, h, facilitation


def diagnostic_loss(pred, tones, durations, lengths):
    mask = torch.arange(MAX_SEGMENTS, device=pred.device)[None, :] < lengths[:, None]
    tone = F.cross_entropy(pred[..., :TONES][mask], tones[mask] - 1)
    duration = F.cross_entropy(pred[..., TONES:][mask], durations[mask] - 3)
    return (tone + duration) / 2


@torch.no_grad()
def evaluate(model, device, seed):
    model.eval()
    rng = random.Random(seed)
    result = {}
    for n in (2, 4, 6):
        x, tones, durations, lengths, items = make_batch(rng, 160, n, n, device)
        for wait in (0, 16, 64):
            pred, h, plastic_state = model(x, wait)
            pt = pred[:, :n, :TONES].argmax(-1) + 1
            pd = pred[:, :n, TONES:].argmax(-1) + 3
            tone_ok = pt == tones[:, :n]
            duration_ok = pd == durations[:, :n]
            joint = tone_ok & duration_ok
            repeated = torch.tensor([len(set(item[1])) < n for item in items], device=device)
            result[f"n{n}_wait{wait}"] = {
                "n": len(items),
                "tone_all": tone_ok.all(1).float().mean().item(),
                "duration_all": duration_ok.all(1).float().mean().item(),
                "joint_all": joint.all(1).float().mean().item(),
                "joint_per_position": joint.float().mean(0).tolist(),
                "repeated_joint_all": joint.all(1)[repeated].float().mean().item() if repeated.any() else None,
                "mean_abs_synaptic_change": plastic_state.abs().mean().item(),
            }
    x, tones, durations, lengths, _ = make_batch(rng, 100, 2, 2, device, force_pair=True)
    pred, _, _ = model(x, wait=16)
    pt = pred[:, :2, :TONES].argmax(-1) + 1
    pd = pred[:, :2, TONES:].argmax(-1) + 3
    correct = (pt == tones[:, :2]).all(1) & (pd == durations[:, :2]).all(1)
    result["duration_swap_pairs"] = {
        "n_pairs": 100,
        "both_correct": (correct[0::2] & correct[1::2]).float().mean().item(),
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2200)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1010)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"plastic": NeuralDynamics(plastic=True).to(device),
              "fixed_synapses": NeuralDynamics(plastic=False).to(device)}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=0.0008)
                  for name, model in models.items()}
    rng = random.Random(1010)
    logs = []
    steps = 2 if args.smoke else args.steps
    for step in range(1, steps + 1):
        high = 2 if step <= steps // 4 else 4 if step <= steps // 2 else 6
        x, tones, durations, lengths, _ = make_batch(rng, args.batch, 2, high, device)
        for name, model in models.items():
            model.train()
            pred, _, _ = model(x)
            loss = diagnostic_loss(pred, tones, durations, lengths)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers[name].step()
            if step == 1 or step % 100 == 0 or step == steps:
                logs.append({"step": step, "model": name, "max_segments": high, "loss": loss.item()})
        if step % 100 == 0 or step == steps:
            with (args.out / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=("step", "model", "max_segments", "loss"))
                writer.writeheader()
                writer.writerows(logs)
            print(json.dumps(logs[-2:]), flush=True)
    report = {name: evaluate(model, device, 11010) for name, model in models.items()}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, model in models.items():
        torch.save(model.state_dict(), args.out / f"{name}.pt")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
