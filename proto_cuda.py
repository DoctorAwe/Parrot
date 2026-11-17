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


def rollout_batch(net, genomes, seeds, device="cuda", steps=None, use_graph=True):
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
        position_np.append(rng.uniform(0.45 * cfg.world_size,
                                       0.55 * cfg.world_size, 2))
        heading_np.append(rng.uniform(-np.pi, np.pi))
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    resources = t(np.asarray(resources_np)).repeat(g, 1, 1)
    position_initial = t(np.asarray(position_np)).repeat(g, 1)
    heading_initial = t(np.asarray(heading_np)).repeat(g)
    phase_initial = t([0.0, 1.6]).expand(b, 2).clone()
    state = {
        "position": position_initial.clone(), "heading": heading_initial.clone(),
        "phase": phase_initial.clone(),
        "speed": torch.zeros(b, device=device),
        "remaining": torch.ones((b, r), dtype=torch.bool, device=device),
        "eaten": torch.zeros(b, dtype=torch.int32, device=device),
        "voltage": torch.zeros((b, n), device=device),
        "output": torch.zeros((b, n), device=device),
        "adaptation": torch.zeros((b, n), device=device),
    }
    weights = t(genomes).repeat_interleave(s, dim=0)
    pre = torch.as_tensor(net.pre.astype(np.int64), device=device)
    post = torch.as_tensor(net.post.astype(np.int64), device=device)
    two_pi = 2.0 * np.pi
    def reset_state():
        state["position"].copy_(position_initial)
        state["heading"].copy_(heading_initial)
        state["phase"].copy_(phase_initial)
        state["speed"].zero_()
        state["remaining"].fill_(True)
        state["eaten"].zero_()
        state["voltage"].zero_()
        state["output"].zero_()
        state["adaptation"].zero_()

    def tick():
        position, heading = state["position"], state["heading"]
        phase, remaining = state["phase"], state["remaining"]
        speed = state["speed"]
        voltage, output = state["voltage"], state["output"]
        adaptation = state["adaptation"]
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
        next_voltage = 0.88 * voltage + 0.12 * (current - 0.25 * adaptation)
        next_output = 0.82 * output + 0.18 * torch.tanh(next_voltage)
        next_adaptation = 0.985 * adaptation + 0.015 * torch.abs(next_output)
        motor = next_output[:, -4:]
        rates = 2.2 + 1.5 * torch.tanh(torch.stack((
            motor[:, 0] - motor[:, 1], motor[:, 2] - motor[:, 3]), dim=1))
        next_phase = torch.remainder(phase + rates * cfg.dt, two_pi)
        drive = torch.sigmoid(3.0 * (torch.stack((
            motor[:, 0] - motor[:, 1], motor[:, 2] - motor[:, 3]), dim=1) - 0.25))
        thrust = drive * (0.6 + 0.4 * torch.cos(next_phase).square())
        target_speed = cfg.max_speed * thrust.mean(dim=1)
        next_speed = (cfg.speed_retention * speed +
                      (1 - cfg.speed_retention) * target_speed)
        next_heading = heading + cfg.turn_gain * (thrust[:, 1] - thrust[:, 0]) * cfg.dt
        direction = torch.stack((torch.cos(next_heading), torch.sin(next_heading)), dim=1)
        next_position = position + (next_speed * cfg.dt)[:, None] * direction
        distance = torch.linalg.vector_norm(resources - next_position[:, None, :], dim=2)
        captured = remaining & (distance <= cfg.capture_radius)
        state["voltage"].copy_(next_voltage)
        state["output"].copy_(next_output)
        state["adaptation"].copy_(next_adaptation)
        state["phase"].copy_(next_phase)
        state["speed"].copy_(next_speed)
        state["heading"].copy_(next_heading)
        state["position"].copy_(next_position)
        state["remaining"].copy_(remaining & ~captured)
        state["eaten"].add_(captured.sum(dim=1).to(torch.int32))

    total_steps = cfg.steps if steps is None else int(steps)
    if use_graph and device == "cuda":
        # Warmup on a side stream, then reset the static state before capture.
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                tick()
        torch.cuda.current_stream().wait_stream(warmup)
        reset_state()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tick()
        reset_state()
        for _ in range(total_steps):
            graph.replay()
        torch.cuda.synchronize()
    else:
        for _ in range(total_steps):
            tick()
    return state["eaten"].reshape(g, s).cpu().numpy()


def evaluate_batch(net, genomes, seeds, device="cuda", use_graph=True):
    scores = rollout_batch(net, genomes, seeds, device=device, use_graph=use_graph)
    return scores.mean(axis=1), scores


def evolve_cuda(net, generations=12, population=16,
                train_seeds=range(1000, 1006),
                validation_seeds=range(2000, 2006), out=None, use_graph=True):
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; enable a GPU runtime in Colab")
    rng = np.random.default_rng(420)
    incumbent = net.reflex_genome()
    history = []
    with torch.inference_mode():
        incumbent_validation = float(evaluate_batch(
            net, incumbent, validation_seeds, use_graph=use_graph)[0][0])
        for generation in range(generations):
            started = perf_counter()
            sigma = 0.30 * 0.96 ** generation + 0.06
            candidates = np.stack([incumbent] + [np.clip(
                incumbent + rng.normal(0, sigma, net.synapses), -4.0, 4.0
            ).astype(np.float32) for _ in range(population - 1)])
            training_scores, _ = evaluate_batch(
                net, candidates, train_seeds, use_graph=use_graph)
            ranked = np.argsort(training_scores)[-min(4, population):]
            validation_scores, _ = evaluate_batch(
                net, candidates[ranked], validation_seeds, use_graph=use_graph)
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


def evaluate_cuda(net, genome, seeds, use_graph=True):
    with _torch().inference_mode():
        values = rollout_batch(net, genome, seeds, use_graph=use_graph)[0].tolist()
    return {"episodes": len(values), "mean_eaten": float(np.mean(values)),
            "median_eaten": float(np.median(values)),
            "eaten_per_world": values}
