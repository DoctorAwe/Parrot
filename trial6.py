"""Trial 6: duration-sensitive populations around a fixed-state stream memory.

The memory is a learned distributed recurrent state. No raw input frames or
decoded symbols are stored for model readback. This remains a synthetic task.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from trial1 import N_TONES
from trial2 import write_wave
from trial3_align_eval import aligned_score


def variable_example(rng, min_words, max_words):
    """Frequency-frame stream with independently varying tone durations."""
    words = [rng.randrange(1, N_TONES + 1) for _ in range(rng.randint(min_words, max_words))]
    utterance = []
    for i, word in enumerate(words):
        utterance.extend([word] * rng.randint(3, 8))
        if i + 1 < len(words):
            utterance.extend([0] * rng.randint(1, 3))
    source = utterance + [0] * rng.randint(5, 10)
    target = [0] * len(source) + utterance + [0] * 8
    source.extend([0] * (len(target) - len(source)))
    return source, target, words


def variable_batch(rng, count, lo, hi, device):
    items = [variable_example(rng, lo, hi) for _ in range(count)]
    length = max(len(item[1]) for item in items)
    x = torch.zeros(count, length, N_TONES + 1, device=device)
    x[:, :, 0] = 1.0
    for b, (source, _, _) in enumerate(items):
        index = torch.arange(len(source), device=device)
        x[b, index, 0] = 0.0
        x[b, index, torch.tensor(source, device=device)] = 1.0
    return x, items


class DurationPopulationMemory(nn.Module):
    """Causal frequency traces, distributed memory, and adapting output activity."""

    def __init__(self, width=96, memory_pairs=192, probe_lags=32):
        super().__init__()
        self.width = width
        self.memory_pairs = memory_pairs
        self.memory_size = memory_pairs * 2
        self.probe_lags = probe_lags
        self.scales = 4
        self.population_size = N_TONES * self.scales * 2
        self.register_buffer("trace_keep", torch.tensor([0.25, 0.65, 0.85, 0.95]))
        self.register_buffer("onset_keep", torch.tensor([0.15, 0.40, 0.70, 0.90]))
        self.input_map = nn.Linear(N_TONES + 1 + self.population_size, width)
        self.controller = nn.ModuleList([
            nn.Linear(width * 3 + self.memory_size, width) for _ in range(3)
        ])
        self.controller_keep = nn.Parameter(torch.tensor([0.0, 2.0, 4.0]))
        self.write_gate = nn.Linear(width * 3 + N_TONES + 1 + self.population_size, self.memory_size)
        self.memory_input = nn.Linear(N_TONES + 1 + self.population_size, self.memory_size, bias=False)
        self.memory_query = nn.Linear(width * 3, self.memory_size)
        self.motor_drive = nn.Linear(width * 3 + self.memory_size, width)
        self.motor_rec = nn.Linear(width, width, bias=False)
        self.motor_adapt = nn.Parameter(torch.tensor(0.5))
        self.output = nn.Linear(width, N_TONES + 1)
        self.duration_probe = nn.Linear(self.population_size, 9)
        self.history_probe = nn.Linear(self.memory_size, probe_lags * (N_TONES + 1))
        # Fixed rotation frequencies span several temporal resolutions.
        angles = torch.linspace(0.03, math.pi * 0.92, memory_pairs)
        self.register_buffer("rotation_cos", angles.cos())
        self.register_buffer("rotation_sin", angles.sin())
        retention = torch.linspace(0.965, 0.998, memory_pairs)
        self.memory_keep_logit = nn.Parameter(torch.logit(retention))

    def initial_state(self, batch_size, device, dtype, delay):
        if delay < 1:
            raise ValueError("delay must be positive")
        controller = [torch.zeros(batch_size, self.width, device=device, dtype=dtype)
                      for _ in range(3)]
        memory = torch.zeros(batch_size, self.memory_size, device=device, dtype=dtype)
        traces = torch.zeros(batch_size, N_TONES, self.scales, device=device, dtype=dtype)
        onsets = torch.zeros_like(traces)
        motor = torch.zeros(batch_size, self.width, device=device, dtype=dtype)
        adaptation = torch.zeros_like(motor)
        # This short queue belongs to the simulated speaker-to-mic environment.
        acoustic_queue = [torch.zeros(batch_size, N_TONES + 1, device=device, dtype=dtype)
                          for _ in range(delay)]
        return controller, memory, traces, onsets, motor, adaptation, acoustic_queue

    def step(self, external_frame, state, feedback_gain=0.0):
        controller, memory, traces, onsets, motor, adaptation, acoustic_queue = state
        heard = external_frame + feedback_gain * acoustic_queue[0]
        excitation = heard[:, 1:].clamp(0, 1).unsqueeze(-1)
        new_onset = F.relu(excitation - traces)
        traces = self.trace_keep * traces + (1 - self.trace_keep) * excitation
        onsets = self.onset_keep * onsets + (1 - self.onset_keep) * new_onset
        population = torch.cat((traces.flatten(1), onsets.flatten(1)), dim=-1)
        sensed = torch.cat((heard, population), dim=-1)
        previous = torch.cat(controller, dim=-1)
        controller_drive = self.input_map(sensed)
        old_memory = memory
        old_pair = old_memory.reshape(old_memory.shape[0], self.memory_pairs, 2)
        a, b = old_pair[..., 0], old_pair[..., 1]
        retention = torch.sigmoid(self.memory_keep_logit)
        rotated = torch.stack((self.rotation_cos * a - self.rotation_sin * b,
                               self.rotation_sin * a + self.rotation_cos * b), dim=-1)
        rotated = (rotated * retention[None, :, None]).reshape_as(memory)
        acoustic_energy = heard[:, 1:].sum(-1, keepdim=True).clamp(0, 1)
        write = torch.sigmoid(self.write_gate(torch.cat((previous, sensed), dim=-1)))
        memory = rotated + acoustic_energy * write * self.memory_input(sensed)
        # Soft boundedness keeps the recurrent state numerically stable.
        memory = 3.0 * torch.tanh(memory / 3.0)
        controller_input = torch.cat((previous, memory), dim=-1)
        keep = torch.sigmoid(self.controller_keep)
        controller = [
            keep[i] * controller[i] + (1 - keep[i]) *
            torch.tanh(controller_drive + self.controller[i](controller_input))
            for i in range(3)
        ]
        current = torch.cat(controller, dim=-1)
        memory_read = memory * torch.sigmoid(self.memory_query(current))
        drive = self.motor_drive(torch.cat((current, memory_read), dim=-1))
        motor = 0.75 * motor + 0.25 * torch.tanh(
            drive + 0.25 * self.motor_rec(motor) - F.softplus(self.motor_adapt) * adaptation)
        adaptation = 0.92 * adaptation + 0.08 * motor.abs()
        logits = self.output(motor)
        acoustic = F.softmax(logits, dim=-1)
        acoustic = torch.cat((acoustic[:, :1] * 0, acoustic[:, 1:]), dim=-1)
        state = (controller, memory, traces, onsets, motor, adaptation,
                 acoustic_queue[1:] + [acoustic])
        return logits, state, memory, population

    def forward(self, external, feedback_gain, delay, return_memory=False):
        state = self.initial_state(external.shape[0], external.device, external.dtype, delay)
        outputs, memories, populations = [], [], []
        for t in range(external.shape[1]):
            logits, state, memory, population = self.step(external[:, t], state, feedback_gain)
            outputs.append(logits)
            if return_memory:
                memories.append(memory)
                populations.append(population)
        if return_memory:
            return torch.stack(outputs, 1), torch.stack(memories, 1), torch.stack(populations, 1)
        return torch.stack(outputs, 1)


def tone_runs_with_duration(frames):
    result = []
    current = 0
    length = 0
    for token in frames:
        token = int(token)
        if token == current and token != 0:
            length += 1
        else:
            if current != 0:
                result.append((current, length))
            current = token
            length = 1 if token != 0 else 0
    if current != 0:
        result.append((current, length))
    return result


def duration_aware_similarity(original, predicted):
    """Monotone edit score with graded duration cost for matching tones."""
    a = tone_runs_with_duration(original)
    b = tone_runs_with_duration(predicted)
    if not a and not b:
        return 1.0
    previous = [float(j) for j in range(len(b) + 1)]
    for i, (tone_a, length_a) in enumerate(a, 1):
        row = [float(i)]
        for j, (tone_b, length_b) in enumerate(b, 1):
            if tone_a == tone_b:
                substitute = min(1.0, 0.35 * abs(math.log(length_a / length_b)))
            else:
                substitute = 1.0
            row.append(min(row[-1] + 1.0, previous[j] + 1.0,
                           previous[j - 1] + substitute))
        previous = row
    return max(0.0, 1.0 - previous[-1] / max(len(a), len(b), 1))


def duration_tolerant_loss(logits, items):
    """CTC marginalizes over output start and per-tone duration; no cue to model."""
    log_prob = F.log_softmax(logits, dim=-1)
    segments = []
    labels = []
    input_lengths = []
    label_lengths = []
    blank_costs = []
    length_costs = []
    for b, (source, target, words) in enumerate(items):
        last_input = max(i for i, value in enumerate(source) if value != 0)
        end = len(target)
        segment = log_prob[b, last_input + 1:end]
        segments.append(segment)
        input_lengths.append(len(segment))
        labels.extend(words)
        label_lengths.append(len(words))
        blank_costs.append(-log_prob[b, :last_input + 1, 0].mean())
        expected_duration = sum(value != 0 for value in source[:last_input + 1])
        predicted_duration = (1.0 - segment[:, 0].exp()).sum()
        length_costs.append(F.smooth_l1_loss(
            predicted_duration / expected_duration, predicted_duration.new_tensor(1.0)))
    padded = pad_sequence(segments, batch_first=False)
    ctc = F.ctc_loss(padded, torch.tensor(labels, dtype=torch.long, device=logits.device),
                     torch.tensor(input_lengths, dtype=torch.long),
                     torch.tensor(label_lengths, dtype=torch.long),
                     blank=0, reduction="mean", zero_infinity=True)
    return ctc + 0.5 * torch.stack(blank_costs).mean() + 0.2 * torch.stack(length_costs).mean()


def history_loss(model, memories, external, items):
    """Training-only probe: recover recent past frames from the distributed state."""
    b, steps, _ = external.shape
    logits = model.history_probe(memories).reshape(
        b, steps, model.probe_lags, N_TONES + 1)
    labels = external.argmax(-1)
    total = logits.new_zeros(())
    count = logits.new_zeros(())
    valid = torch.tensor([min(len(item[1]), steps) for item in items],
                         device=external.device)
    for lag in range(1, model.probe_lags + 1):
        if lag >= steps:
            break
        t = torch.arange(lag, steps, device=external.device)
        usable = t.unsqueeze(0) < valid.unsqueeze(1)
        target = labels[:, :steps - lag]
        term = F.cross_entropy(logits[:, lag:, lag - 1].transpose(1, 2), target,
                               reduction="none")
        total = total + (term * usable).sum()
        count = count + usable.sum()
    return total / count.clamp_min(1)


def duration_readout_loss(model, populations, external):
    """Auxiliary training probe; duration labels never enter the streaming model."""
    logits, elapsed, active = duration_readout(model, populations, external)
    if not active.any():
        return populations.sum() * 0
    return F.cross_entropy(logits[active], elapsed[active].clamp_max(8))


def duration_readout(model, populations, external):
    tokens = external.argmax(-1)
    elapsed = torch.zeros_like(tokens)
    for t in range(tokens.shape[1]):
        previous = tokens[:, t - 1] if t else torch.zeros_like(tokens[:, t])
        elapsed[:, t] = torch.where(tokens[:, t] != 0,
                                    torch.where(tokens[:, t] == previous,
                                                elapsed[:, t - 1] + 1 if t else 1, 1), 0)
    logits = model.duration_probe(populations)
    return logits, elapsed, tokens != 0


@torch.no_grad()
def evaluate(model, device, seed, count, gain):
    model.eval()
    rng = random.Random(seed)
    groups = {}
    examples = []
    for index in range(count):
        x, items = variable_batch(rng, 1, 3, 8, device)
        source, target, words = items[0]
        logits, _, populations = model(x, gain, rng.randint(2, 5), return_memory=True)
        probe_logits, elapsed, active = duration_readout(model, populations, x)
        probe_correct = ((probe_logits.argmax(-1) == elapsed.clamp_max(8)) & active).sum().item()
        probe_count = active.sum().item()
        predicted = logits.argmax(-1)[0, :len(target)].tolist()
        decoded = [tone for tone, _ in tone_runs_with_duration(predicted)]
        alignment = aligned_score(source, target, predicted)
        similarity = duration_aware_similarity(source, predicted)
        repeated = len(set(words)) < len(words)
        for key in ("overall", f"length_{len(words)}", "repeated" if repeated else "distinct"):
            if key not in groups:
                groups[key] = {"n": 0, "similarity": 0.0, "no_interruption": 0,
                               "sequence_exact": 0, "acoustic_proxy": 0.0,
                               "duration_probe_correct": 0, "duration_probe_count": 0}
            g = groups[key]
            g["n"] += 1
            g["similarity"] += similarity if alignment["no_interruption"] else 0.0
            g["no_interruption"] += int(alignment["no_interruption"])
            g["sequence_exact"] += int(alignment["no_interruption"] and decoded == words)
            g["acoustic_proxy"] += alignment["input_active_frame_accuracy"]
            g["duration_probe_correct"] += probe_correct
            g["duration_probe_count"] += probe_count
        if index < 5:
            examples.append({"source_words": words, "predicted_words": decoded,
                             "source_runs": tone_runs_with_duration(source),
                             "predicted_runs": tone_runs_with_duration(predicted),
                             "duration_aware_similarity": similarity,
                             "input_frames": source, "predicted_frames": predicted,
                             **alignment})
    metrics = {key: {"n": value["n"],
                     "duration_aware_similarity": value["similarity"] / value["n"],
                     "no_interruption_rate": value["no_interruption"] / value["n"],
                     "sequence_exact_diagnostic": value["sequence_exact"] / value["n"],
                     "aligned_active_frame_accuracy": value["acoustic_proxy"] / value["n"],
                     "duration_probe_accuracy": value["duration_probe_correct"] /
                     max(1, value["duration_probe_count"])}
               for key, value in groups.items()}
    return metrics, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--eval", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(5050)
    torch.manual_seed(5050)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DurationPopulationMemory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)
    rng = random.Random(5050)
    steps = 2 if args.smoke else args.steps
    history = []
    print(json.dumps({"device": str(device), "steps": steps}), flush=True)
    for step in range(1, steps + 1):
        model.train()
        if step <= steps // 3:
            lo, hi = 2, 3
        elif step <= 2 * steps // 3:
            lo, hi = 2, 5
        else:
            lo, hi = 3, 8
        x, items = variable_batch(rng, args.batch, lo, hi, device)
        gain = rng.choice((0.0, 0.15, 0.35))
        logits, memories, populations = model(x, gain, rng.randint(2, 5), return_memory=True)
        copy_loss = duration_tolerant_loss(logits, items)
        probe_loss = history_loss(model, memories, x, items)
        duration_probe_loss = duration_readout_loss(model, populations, x)
        loss = copy_loss + 0.15 * probe_loss + 0.05 * duration_probe_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            row = {"step": step, "loss": float(loss.item()),
                   "copy_loss": float(copy_loss.item()),
                   "probe_loss": float(probe_loss.item()),
                   "duration_probe_loss": float(duration_probe_loss.item()),
                   "min_words": lo, "max_words": hi}
            if step % 500 == 0 or step == steps:
                validation, _ = evaluate(model, device, 6161, 5 if args.smoke else 80, 0.15)
                row["validation_similarity"] = validation["overall"]["duration_aware_similarity"]
                row["validation_repeated_similarity"] = validation.get("repeated", {}).get("duration_aware_similarity")
                row["validation_length_6_similarity"] = validation.get("length_6", {}).get("duration_aware_similarity")
            history.append(row)
            with (args.out / "train_log.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("step", "loss", "copy_loss", "probe_loss",
                                                         "duration_probe_loss",
                                                         "min_words", "max_words", "validation_similarity",
                                                         "validation_repeated_similarity",
                                                         "validation_length_6_similarity"))
                writer.writeheader()
                writer.writerows(history)
            print(json.dumps(row), flush=True)
    report = {"device": str(device), "steps": steps,
              "architecture": "duration_population_memory", "conditions": {}}
    for gain in (0.0, 0.15, 0.35):
        metrics, examples = evaluate(model, device, 7171, 5 if args.smoke else args.eval, gain)
        report["conditions"][str(gain)] = metrics
        if gain == 0.15:
            (args.out / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
            first = examples[0]
            write_wave(args.out / "input.wav", first["input_frames"])
            write_wave(args.out / "model.wav", first["predicted_frames"])
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.out / "model.pt")
    print(json.dumps(report["conditions"]["0.15"], indent=2), flush=True)


if __name__ == "__main__":
    main()
