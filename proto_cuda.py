"""Batched CUDA simulation of the sparse bilateral nerve net and nutrient world.

The tick loop is causal. Within a tick, synapses, neurons, candidate genomes,
and independent worlds are evaluated in parallel. This is an experiment
simulator; no trajectory or resource coordinates enter the neural controller.
"""

import numpy as np
from time import perf_counter

from proto_microbe import save_checkpoint


def _torch():
    import torch
    return torch


def cuda_available():
    return _torch().cuda.is_available()


def rollout_batch(net, genomes, seeds, device="cuda", steps=None):
    """Return [genome, seed] eaten counts, with all episodes run together."""
    torch = _torch()
    cfg = net.cfg
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; enable a GPU runtime in Colab")
    genomes = np.asarray(genomes, dtype=np.float32)
    if genomes.ndim == 1:
        genomes = genomes[None, :]
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one world seed is needed")
    if genomes.shape[1] != net.synapses:
        raise ValueError("Genome length does not match the network synapses")
    g, s = len(genomes), len(seeds)
    b, n, r = g * s, net.n, cfg.resource_count
    # Match the NumPy world's initial conditions, then broadcast each seed
    # across genomes. Initial conditions are external environment state.
    resources_np, position_np, heading_np = [], [], []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        resources_np.append(rng.uniform(1.0, cfg.world_size - 1.0, (r, 2)))
        position_np.append(rng.uniform(3, cfg.world_size - 3, 2))
        heading_np.append(rng.uniform(-np.pi, np.pi))
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    resources = t(np.asarray(resources_np)).repeat(g, 1, 1)
    position = t(np.asarray(position_np)).repeat(g, 1)
    heading = t(np.asarray(heading_np)).repeat(g)
    phase = t([0.0, 1.6]).expand(b, 2).clone()
    remaining = torch.ones((b, r), dtype=torch.bool, device=device)
    eaten = torch.zeros(b, dtype=torch.int32, device=device)
    voltage = torch.zeros((b, n), device=device)
    output = torch.zeros_like(voltage)
    adaptation = torch.zeros_like(voltage)
    weights = t(genomes).repeat_interleave(s, dim=0)
    pre = torch.as_tensor(net.pre.astype(np.int64), device=device)
    post = torch.as_tensor(net.post.astype(np.int64), device=device)
    two_pi = 2.0 * np.pi
    for _ in range(cfg.steps if steps is None else int(steps)):
        forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        side = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
        center = position + cfg.sensor_forward * forward
        sites = torch.stack((center + cfg.sensor_offset * side,
                             center - cfg.sensor_offset * side), dim=1)
        distances = torch.linalg.vector_norm(
            sites[:, :, None, :] - resources[:, None, :, :], dim=-1)
        contribution = torch.exp(-distances / cfg.diffusion_length)
        sensed = torch.where(contribution >= cfg.cutoff, contribution, 0.0)
        sensed = (sensed * remaining[:, None, :]).sum(dim=2)
        # Each real synapse transmits one weighted presynaptic output. The
        # reduction accumulates into postsynaptic cells in parallel on CUDA.
        current = torch.zeros_like(voltage)
        current.index_add_(1, post, weights * output.index_select(1, pre))
        current[:, :2] += 1.8 * torch.log1p(sensed)
        voltage = 0.88 * voltage + 0.12 * (current - 0.25 * adaptation)
        output = 0.82 * output + 0.18 * torch.tanh(voltage)
        adaptation = 0.985 * adaptation + 0.015 * torch.abs(output)
        motor = output[:, -4:]
        rates = 2.2 + 1.5 * torch.tanh(torch.stack((
            motor[:, 0] - motor[:, 1], motor[:, 2] - motor[:, 3]), dim=1))
        phase = torch.remainder(phase + rates * cfg.dt, two_pi)
        thrust = 0.60 + 0.42 * rates * (0.5 + 0.5 * torch.cos(phase).square())
        speed = 0.31 * thrust.sum(dim=1)
        heading = heading + 0.80 * (thrust[:, 1] - thrust[:, 0]) * cfg.dt
        direction = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        position = position + (speed * cfg.dt)[:, None] * direction
        low, high = 0.25, cfg.world_size - 0.25
        hit_x = (position[:, 0] < low) | (position[:, 0] > high)
        heading = torch.where(hit_x, np.pi - heading, heading)
        hit_y = (position[:, 1] < low) | (position[:, 1] > high)
        heading = torch.where(hit_y, -heading, heading)
        position = position.clamp(low, high)
        distance = torch.linalg.vector_norm(resources - position[:, None, :], dim=2)
        captured = remaining & (distance <= cfg.capture_radius)
        remaining = remaining & ~captured
        eaten += captured.sum(dim=1).to(torch.int32)
    return eaten.reshape(g, s).cpu().numpy()


def evaluate_batch(net, genomes, seeds, device="cuda"):
    scores = rollout_batch(net, genomes, seeds, device=device)
    return scores.mean(axis=1), scores


def evolve_cuda(net, generations=12, population=16,
                train_seeds=range(1000, 1006),
                validation_seeds=range(2000, 2006), out=None):
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; enable a GPU runtime in Colab")
    rng = np.random.default_rng(420)
    incumbent = net.reflex_genome()
    history = []
    with torch.inference_mode():
        incumbent_validation = float(evaluate_batch(
            net, incumbent, validation_seeds)[0][0])
        for generation in range(generations):
            started = perf_counter()
            sigma = 0.30 * 0.96 ** generation + 0.06
            candidates = np.stack([incumbent] + [np.clip(
                incumbent + rng.normal(0, sigma, net.synapses), -4.0, 4.0
            ).astype(np.float32) for _ in range(population - 1)])
            training_scores, _ = evaluate_batch(net, candidates, train_seeds)
            ranked = np.argsort(training_scores)[-min(4, population):]
            validation_scores, _ = evaluate_batch(
                net, candidates[ranked], validation_seeds)
            winner_local = int(np.argmax(validation_scores))
            validation = float(validation_scores[winner_local])
            if validation > incumbent_validation:
                incumbent = candidates[ranked[winner_local]].copy()
                incumbent_validation = validation
            torch.cuda.synchronize()
            row = {"generation": generation + 1,
                   "best_training_eaten": float(training_scores.max()),
                   "incumbent_validation_eaten": incumbent_validation,
                   "sigma": float(sigma),
                   "elapsed_s": round(perf_counter() - started, 2)}
            history.append(row)
            if (generation + 1) % 3 == 0 or generation == 0:
                print(row, flush=True)
            if out is not None:
                save_checkpoint(out, net, incumbent, history)
    return incumbent, history


def evaluate_cuda(net, genome, seeds):
    with _torch().inference_mode():
        values = rollout_batch(net, genome, seeds)[0].tolist()
    return {"episodes": len(values), "mean_eaten": float(np.mean(values)),
            "median_eaten": float(np.median(values)),
            "eaten_per_world": values}
