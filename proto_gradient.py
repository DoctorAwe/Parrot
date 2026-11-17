"""Truncated backpropagation through the continuous 100 Hz microbe world.

The controller only sees two concentration sensors. Resource coordinates are
used by the external training loss and are never passed to neural dynamics.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np


def train_gradient(net, out=None, epochs=16, worlds=16, ticks_per_episode=600,
                   segment=100, learning_rate=0.015, device="cuda"):
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU runtime in Colab")
    cfg = net.cfg
    if ticks_per_episode % segment:
        raise ValueError("ticks_per_episode must be divisible by segment")
    torch.manual_seed(73)
    generator = np.random.default_rng(73)
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    pre = torch.as_tensor(net.pre.astype(np.int64), device=device)
    post = torch.as_tensor(net.post.astype(np.int64), device=device)
    start_weights = net.reflex_genome()
    start_weights += generator.normal(0, 0.025, net.synapses).astype(np.float32)
    weights = torch.nn.Parameter(t(start_weights))
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    # Direct sensory-to-motor synapses are trained first. Other synapses are
    # admitted later, so a useful short path exists before long loops adapt.
    direct = ((net.pre < 2) & (net.post >= net.n - 4))
    direct_mask = t(direct.astype(np.float32))
    history = []

    def make_world(epoch):
        resources, positions, headings = [], [], []
        for seed in range(10000 + epoch * worlds, 10000 + (epoch + 1) * worlds):
            rng = np.random.default_rng(seed)
            resources.append(rng.uniform(1, cfg.world_size - 1,
                                         (cfg.resource_count, 2)))
            positions.append(rng.uniform(0.45 * cfg.world_size,
                                         0.55 * cfg.world_size, 2))
            headings.append(rng.uniform(-np.pi, np.pi))
        state = {
            "position": t(np.asarray(positions)),
            "heading": t(np.asarray(headings)),
            "phase": t([0.0, 1.6]).expand(worlds, 2).clone(),
            "speed": torch.zeros(worlds, device=device),
            "remaining": torch.ones((worlds, cfg.resource_count),
                                    dtype=torch.bool, device=device),
            "voltage": torch.zeros((worlds, net.n), device=device),
            "output": torch.zeros((worlds, net.n), device=device),
            "adaptation": torch.zeros((worlds, net.n), device=device),
            "eaten": torch.zeros(worlds, dtype=torch.int32, device=device),
        }
        return t(np.asarray(resources)), state

    def tick(resources, state):
        position, heading = state["position"], state["heading"]
        phase, speed = state["phase"], state["speed"]
        remaining = state["remaining"]
        forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        side = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
        center = position + cfg.sensor_forward * forward
        sites = torch.stack((center + cfg.sensor_offset * side,
                             center - cfg.sensor_offset * side), dim=1)
        distances = torch.linalg.vector_norm(
            sites[:, :, None, :] - resources[:, None, :, :], dim=-1)
        contribution = torch.exp(-distances / cfg.diffusion_length)
        sensed = (torch.where(contribution >= cfg.cutoff, contribution, 0.0)
                  * remaining[:, None, :]).sum(dim=2)
        current = torch.zeros_like(state["voltage"])
        current.index_add_(1, post, weights[None, :] *
                           state["output"].index_select(1, pre))
        current[:, :2] += 1.8 * torch.log1p(sensed)
        voltage = 0.88 * state["voltage"] + 0.12 * (
            current - 0.25 * state["adaptation"])
        output = 0.82 * state["output"] + 0.18 * torch.tanh(voltage)
        adaptation = 0.985 * state["adaptation"] + 0.015 * torch.abs(output)
        motor = output[:, -4:]
        difference = torch.stack((motor[:, 0] - motor[:, 1],
                                  motor[:, 2] - motor[:, 3]), dim=1)
        rates = 2.2 + 1.5 * torch.tanh(difference)
        phase = torch.remainder(phase + rates * cfg.dt, 2.0 * np.pi)
        drive = torch.sigmoid(3.0 * (difference - 0.25))
        thrust = drive * (0.6 + 0.4 * torch.cos(phase).square())
        speed = cfg.speed_retention * speed + (
            1 - cfg.speed_retention) * cfg.max_speed * thrust.mean(dim=1)
        heading = heading + cfg.turn_gain * (thrust[:, 1] - thrust[:, 0]) * cfg.dt
        direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        position = position + (speed * cfg.dt)[:, None] * direction
        distance = torch.linalg.vector_norm(resources - position[:, None, :], dim=2)
        captured = remaining & (distance <= cfg.capture_radius)
        return {
            "position": position, "heading": heading, "phase": phase,
            "speed": speed, "remaining": remaining & ~captured,
            "voltage": voltage, "output": output, "adaptation": adaptation,
            "eaten": state["eaten"] + captured.sum(dim=1).to(torch.int32),
        }

    def distance_loss(resources, state):
        distance = torch.linalg.vector_norm(
            resources - state["position"][:, None, :], dim=2)
        # Smooth nearest *available* resource. The mask is external to the
        # controller; it only determines what the trainer rewards next.
        distance = torch.where(state["remaining"], distance, 1e3)
        return (-2.0 * torch.logsumexp(-distance / 2.0, dim=1)).mean()

    for epoch in range(epochs):
        started = perf_counter()
        resources, state = make_world(epoch)
        epoch_losses = []
        for _ in range(ticks_per_episode // segment):
            for tick_index in range(segment):
                state = tick(resources, state)
            loss = distance_loss(resources, state) + 1e-5 * weights.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch < epochs // 4:
                weights.grad.mul_(direct_mask)
            torch.nn.utils.clip_grad_norm_([weights], 1.0)
            optimizer.step()
            with torch.no_grad():
                weights.clamp_(-4.0, 4.0)
            epoch_losses.append(loss.detach())
            state = {name: value.detach() for name, value in state.items()}
        if device == "cuda":
            torch.cuda.synchronize()
        row = {"epoch": epoch + 1,
               "mean_distance_loss": float(torch.stack(epoch_losses).mean()),
               "mean_eaten_during_training": float(state["eaten"].float().mean()),
               "elapsed_s": round(perf_counter() - started, 2)}
        history.append(row)
        print(row, flush=True)
        if out is not None:
            out_path = Path(out)
            out_path.mkdir(parents=True, exist_ok=True)
            np.save(out_path / "best_genome.npy", weights.detach().cpu().numpy())
            (out_path / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
    return weights.detach().cpu().numpy(), history
