"""Pre-laminar bilateral nerve net for the nutrient world, at 100 ticks/s.

The arrays below represent membrane/output levels and individual synapses.
There is no addressable memory, key, query, event slot, or stored trajectory in
the controller. A trajectory may be recorded by the external experimenter.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from microbe_world import NutrientWorld


@dataclass
class ProtoConfig:
    world_size: float = 20.0
    resource_count: int = 18
    diffusion_length: float = 1.5
    cutoff: float = 0.025
    capture_radius: float = 0.48
    sensor_offset: float = 0.55
    sensor_forward: float = 0.45
    steps: int = 3000
    dt: float = 0.01
    neuron_count: int = 128
    fan_in: int = 12
    flagellum_span: float = 0.72


class BilateralNerveNet:
    def __init__(self, cfg=None, seed=404):
        self.cfg = cfg or ProtoConfig()
        self.n = self.cfg.neuron_count
        if self.n < 16 or self.cfg.fan_in < 4:
            raise ValueError("Need at least 16 neurons and 4 incoming synapses")
        rng = np.random.default_rng(seed)
        # Anatomical coordinates: sensory at the head, motors at the tail,
        # interneurons in two unlayered bilateral ganglia.
        coordinates = rng.uniform((-0.8, -0.7), (0.8, 0.7), (self.n, 2))
        coordinates[:2] = ((0.95, 0.4), (0.95, -0.4))
        coordinates[-4:] = ((-0.95, 0.45), (-0.95, 0.25),
                            (-0.95, -0.25), (-0.95, -0.45))
        self.coordinates = coordinates
        pre, post = [], []
        for target in range(2, self.n):
            distance = np.linalg.norm(coordinates - coordinates[target], axis=1)
            preference = np.exp(-distance / 0.42) + 0.025
            preference[target] += 0.18
            preference /= preference.sum()
            sources = rng.choice(self.n, size=min(self.cfg.fan_in, self.n),
                                 replace=False, p=preference)
            for source in sources:
                pre.append(int(source))
                post.append(target)
        # Ascending sensory-to-motor projections allow a minimal reflex; they
        # are ordinary axons and may be weakened by evolution.
        for target in range(self.n - 4, self.n):
            for source in (0, 1):
                if not any(p == target and q == source for p, q in zip(post, pre)):
                    pre.append(source)
                    post.append(target)
        self.pre = np.asarray(pre, dtype=np.int32)
        self.post = np.asarray(post, dtype=np.int32)
        self.synapses = len(pre)

    def reflex_genome(self):
        weights = np.zeros(self.synapses, dtype=np.float32)
        for k, (post, pre) in enumerate(zip(self.post, self.pre)):
            if post == self.n - 4:  # left motor: less left, more right
                weights[k] = -2.4 if pre == 0 else 2.4 if pre == 1 else 0
            elif post == self.n - 2:  # right motor: more left, less right
                weights[k] = 2.4 if pre == 0 else -2.4 if pre == 1 else 0
        return weights

    def random_genome(self, rng):
        return rng.normal(0, 0.35, self.synapses).astype(np.float32)

    def initial_state(self):
        return (np.zeros(self.n, dtype=np.float32),
                np.zeros(self.n, dtype=np.float32),
                np.zeros(self.n, dtype=np.float32))

    def step(self, sensed, state, weights):
        voltage, output, adaptation = state
        # Sparse synaptic transmission: one multiply per real edge, then
        # addition at each receiving neuron. No dense neuron-pair matrix.
        synaptic_input = np.bincount(
            self.post, weights=weights * output[self.pre], minlength=self.n)
        synaptic_input[:2] += 1.8 * np.log1p(sensed)
        voltage = 0.88 * voltage + 0.12 * (
            synaptic_input - 0.25 * adaptation)
        output = 0.82 * output + 0.18 * np.tanh(voltage)
        adaptation = 0.985 * adaptation + 0.015 * np.abs(output)
        return (voltage, output, adaptation), output[-4:]


def rollout(net, genome, seed, cfg=None, record=False):
    cfg = cfg or net.cfg
    world = NutrientWorld(seed, cfg)
    state = net.initial_state()
    frames = []
    initial_resources = world.resources.copy()
    for tick in range(cfg.steps):
        sensed, sensor_sites = world.sensors()
        state, motor = net.step(sensed, state, genome)
        swing, _ = world.step(motor)
        if record:
            frames.append({"tick": tick, "position": world.position.tolist(),
                           "heading": float(world.heading),
                           "sensor_sites": sensor_sites.tolist(),
                           "sensed": sensed.tolist(), "motor": motor.tolist(),
                           "swing": swing.tolist(),
                           "remaining": world.remaining.astype(int).tolist(),
                           "eaten": world.eaten})
    score = float(world.eaten)
    if not record:
        return score
    return score, {"config": vars(cfg), "resources": initial_resources.tolist(),
                   "frames": frames, "eaten": world.eaten}


def evaluate(net, genome, seeds):
    scores = [rollout(net, genome, int(s)) for s in seeds]
    return {"episodes": len(scores), "mean_eaten": float(np.mean(scores)),
            "median_eaten": float(np.median(scores)), "eaten_per_world": scores}


def evolve(net, generations=12, population=16, train_seeds=range(1000, 1006),
           validation_seeds=range(2000, 2006), out=None):
    rng = np.random.default_rng(420)
    incumbent = net.reflex_genome()
    incumbent_validation = evaluate(net, incumbent, validation_seeds)["mean_eaten"]
    history = []
    for generation in range(generations):
        sigma = 0.30 * 0.96 ** generation + 0.06
        candidates = [incumbent] + [np.clip(incumbent +
                      rng.normal(0, sigma, net.synapses), -4.0, 4.0).astype(np.float32)
                      for _ in range(population - 1)]
        training_scores = [evaluate(net, g, train_seeds)["mean_eaten"] for g in candidates]
        ranked = np.argsort(training_scores)[-min(4, population):]
        checked = [(evaluate(net, candidates[int(i)], validation_seeds)["mean_eaten"], int(i))
                   for i in ranked]
        checked.append((incumbent_validation, 0))
        validation, winner = max(checked)
        if validation > incumbent_validation:
            incumbent = candidates[winner].copy()
            incumbent_validation = validation
        row = {"generation": generation + 1,
               "best_training_eaten": float(max(training_scores)),
               "incumbent_validation_eaten": float(incumbent_validation),
               "sigma": float(sigma)}
        history.append(row)
        if (generation + 1) % 3 == 0 or generation == 0:
            print(row, flush=True)
        if out is not None:
            save_checkpoint(out, net, incumbent, history)
    return incumbent, history


def save_checkpoint(out, net, genome, history):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "best_genome.npy", genome)
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(vars(net.cfg), indent=2), encoding="utf-8")


def replay_figure(replay, every=30):
    """Browser-native Plotly animation; no matplotlib or notebook widget needed."""
    import plotly.graph_objects as go

    cfg = replay["config"]
    resources = np.asarray(replay["resources"])
    frames = replay["frames"][::max(1, int(every))]
    axis = np.linspace(0, cfg["world_size"], 75)
    xx, yy = np.meshgrid(axis, axis)
    distance = np.linalg.norm(
        np.stack((xx, yy), axis=-1)[..., None, :] - resources, axis=-1)
    field = np.exp(-distance / cfg["diffusion_length"])
    field[field < cfg["cutoff"]] = 0
    field = field.sum(axis=-1)

    def dynamic_traces(index):
        item = frames[index]
        p = np.asarray(item["position"])
        a = item["heading"]
        forward = np.array((np.cos(a), np.sin(a)))
        side = np.array((-forward[1], forward[0]))
        trail = np.asarray([f["position"] for f in frames[:index + 1]])
        seen = np.asarray(item["remaining"], dtype=bool)
        data = [
            go.Scatter(x=resources[:, 0], y=resources[:, 1], mode="markers",
                       marker=dict(color=np.where(seen, "darkgreen", "lightgray"), size=10),
                       name="Resources"),
            go.Scatter(x=trail[:, 0], y=trail[:, 1], mode="lines",
                       line=dict(color="black", width=2), name="Path"),
            go.Scatter(x=[p[0]], y=[p[1]], mode="markers",
                       marker=dict(color="black", size=15), name="Body"),
            go.Scatter(x=np.asarray(item["sensor_sites"])[:, 0],
                       y=np.asarray(item["sensor_sites"])[:, 1], mode="markers",
                       marker=dict(color="cyan", size=7), name="Sensors"),
        ]
        for k, color in enumerate(("blue", "red")):
            root = p - 0.25 * forward + (0.24 if k == 0 else -0.24) * side
            tip = root - 0.8 * forward + item["swing"][k] * side
            data.append(go.Scatter(x=[root[0], tip[0]], y=[root[1], tip[1]],
                                   mode="lines", line=dict(color=color, width=4),
                                   name="Left flagellum" if k == 0 else "Right flagellum"))
        return data

    frame_objects = [go.Frame(data=dynamic_traces(i), traces=list(range(1, 7)),
                              name=str(f["tick"])) for i, f in enumerate(frames)]
    fig = go.Figure(data=[go.Heatmap(x=axis, y=axis, z=field,
                                     colorscale="YlGn", opacity=0.65,
                                     showscale=False)] + dynamic_traces(0),
                    frames=frame_objects)
    fig.update_layout(
        title=f"100 ticks/s · eaten {replay['eaten']} / {len(resources)}",
        xaxis=dict(range=[0, cfg["world_size"]], scaleanchor="y"),
        yaxis=dict(range=[0, cfg["world_size"]]),
        height=650,
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 65,
                 "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="Pause", method="animate", args=[[None], {"mode": "immediate",
                 "frame": {"duration": 0, "redraw": False}}])])],
        sliders=[dict(currentvalue={"prefix": "Tick "}, steps=[dict(
            label=str(f["tick"]), method="animate",
            args=[[str(f["tick"])], {"mode": "immediate", "frame": {"duration": 0,
                                      "redraw": True}}]) for f in frames])],
    )
    return fig
