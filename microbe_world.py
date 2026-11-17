"""Small 2D nutrient world controlled by a sparse neural circuit.

Requires numpy; plotting and widgets are imported only by show_replay().
The controller has exactly two sensory neurons and four motor neurons. Its
state consists of activity and adaptation, with no addressable memory.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class Config:
    world_size: float = 20.0
    resource_count: int = 18
    diffusion_length: float = 1.5
    cutoff: float = 0.025
    capture_radius: float = 0.48
    sensor_offset: float = 0.55
    sensor_forward: float = 0.45
    steps: int = 320
    dt: float = 0.13
    neuron_count: int = 32  # 2 sensory, 26 internal, 4 motor
    connection_density: float = 0.16
    flagellum_span: float = 0.72


class NutrientWorld:
    def __init__(self, seed, config=None):
        self.cfg = config or Config()
        self.rng = np.random.default_rng(seed)
        lo, hi = 1.0, self.cfg.world_size - 1.0
        self.resources = self.rng.uniform(lo, hi, (self.cfg.resource_count, 2))
        self.remaining = np.ones(self.cfg.resource_count, dtype=bool)
        self.position = self.rng.uniform(3, self.cfg.world_size - 3, 2)
        self.heading = self.rng.uniform(-np.pi, np.pi)
        self.phases = np.array([0.0, 1.6])
        self.eaten = 0
        self.path_length = 0.0

    def concentration(self, points):
        points = np.asarray(points, dtype=float)
        delta = points[..., None, :] - self.resources
        distance = np.linalg.norm(delta, axis=-1)
        contribution = np.exp(-distance / self.cfg.diffusion_length)
        contribution[contribution < self.cfg.cutoff] = 0.0
        return np.sum(contribution * self.remaining, axis=-1)

    def sensors(self):
        forward = np.array([np.cos(self.heading), np.sin(self.heading)])
        left = np.array([-forward[1], forward[0]])
        center = self.position + self.cfg.sensor_forward * forward
        sites = np.stack((center + self.cfg.sensor_offset * left,
                          center - self.cfg.sensor_offset * left))
        return self.concentration(sites), sites

    def step(self, motor):
        # Opposing neural signals determine each flagellum's beating rate.
        motor = np.asarray(motor, dtype=float)
        rates = 2.2 + 1.5 * np.tanh(np.array([motor[0] - motor[1],
                                              motor[2] - motor[3]]))
        self.phases = (self.phases + rates * self.cfg.dt) % (2 * np.pi)
        swing = self.cfg.flagellum_span * np.sin(self.phases)
        thrust = 0.60 + 0.42 * rates * (0.5 + 0.5 * np.cos(self.phases) ** 2)
        speed = 0.31 * (thrust[0] + thrust[1])
        turn_rate = 0.80 * (thrust[1] - thrust[0])
        self.heading += turn_rate * self.cfg.dt
        delta = speed * self.cfg.dt * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position = self.position + delta
        self.path_length += np.linalg.norm(delta)
        margin = 0.25
        low, high = margin, self.cfg.world_size - margin
        for axis in range(2):
            if self.position[axis] < low or self.position[axis] > high:
                self.position[axis] = np.clip(self.position[axis], low, high)
                self.heading = (np.pi - self.heading) if axis == 0 else -self.heading
        distance = np.linalg.norm(self.resources - self.position, axis=1)
        newly_eaten = self.remaining & (distance <= self.cfg.capture_radius)
        self.remaining[newly_eaten] = False
        self.eaten += int(newly_eaten.sum())
        return swing, int(newly_eaten.sum())


class SparseCircuit:
    def __init__(self, config=None, seed=42):
        self.cfg = config or Config()
        self.n = self.cfg.neuron_count
        assert self.n >= 10
        rng = np.random.default_rng(seed)
        # First two neurons receive the sensory channels. Last four innervate
        # the two flagella. Edges are local synaptic connections, not slots.
        edges = []
        for post in range(2, self.n):
            for pre in range(self.n):
                if rng.random() < self.cfg.connection_density or post == pre:
                    edges.append((post, pre))
        for post in range(self.n - 4, self.n):
            for pre in (0, 1):
                if (post, pre) not in edges:
                    edges.append((post, pre))
        self.post, self.pre = np.asarray(edges, dtype=int).T
        self.mask = np.zeros((self.n, self.n), dtype=float)
        self.mask[self.post, self.pre] = 1.0
        self.edge_count = len(edges)
        self.genome_size = self.edge_count + 2 * (self.n - 2) + 2

    def random_genome(self, rng):
        weights = rng.normal(0, 0.7, self.edge_count)
        biases = rng.normal(0, 0.2, self.n - 2)
        time_logits = rng.normal(-0.5, 0.45, self.n - 2)
        sensory_gain = np.array([1.3, 1.3])
        return np.concatenate((weights, biases, time_logits, sensory_gain))

    def reflex_genome(self):
        """Simple bilateral chemotaxis starting circuit, then evolved freely."""
        g = np.zeros(self.genome_size)
        for index, (post, pre) in enumerate(zip(self.post, self.pre)):
            if post == self.n - 4:  # left positive motor
                g[index] = -2.2 if pre == 0 else 2.2 if pre == 1 else 0
            elif post == self.n - 2:  # right positive motor
                g[index] = 2.2 if pre == 0 else -2.2 if pre == 1 else 0
        start = self.edge_count + self.n - 2
        g[start:start + self.n - 2] = 1.0
        g[-2:] = 1.5
        return g

    def unpack(self, genome):
        g = np.asarray(genome)
        p = self.edge_count
        w = np.zeros((self.n, self.n))
        w[self.post, self.pre] = g[:p]
        bias = np.zeros(self.n)
        bias[2:] = g[p:p + self.n - 2]
        time = 1 / (1 + np.exp(-np.clip(g[p + self.n - 2:p + 2 * (self.n - 2)], -6, 6)))
        gain = np.clip(g[-2:], 0.15, 5.0)
        return w, bias, time, gain

    def initial_state(self):
        return np.zeros(self.n), np.zeros(self.n)

    def step(self, sensors, state, params):
        h, adaptation = state
        w, bias, time, gain = params
        current = w @ h + bias
        updated = h.copy()
        updated[:2] = 0.75 * h[:2] + 0.25 * np.tanh(gain * sensors)
        updated[2:] = (1 - time) * h[2:] + time * np.tanh(current[2:] - 0.35 * adaptation[2:])
        adaptation = 0.92 * adaptation + 0.08 * np.abs(updated)
        return (updated, adaptation), updated[-4:]


def rollout(circuit, genome, seed, config=None, record=False):
    cfg = config or circuit.cfg
    world = NutrientWorld(seed, cfg)
    state = circuit.initial_state()
    params = circuit.unpack(genome)
    trajectory = []
    initial_resources = world.resources.copy()
    initial_concentration = world.concentration(world.position)
    for frame in range(cfg.steps):
        sensed, sites = world.sensors()
        state, motor = circuit.step(np.log1p(sensed), state, params)
        swing, eaten = world.step(motor)
        if record:
            trajectory.append({"t": frame, "p": world.position.tolist(),
                               "angle": float(world.heading), "sensors": sites.tolist(),
                               "signal": sensed.tolist(), "motor": motor.tolist(),
                               "swing": swing.tolist(), "remaining": world.remaining.astype(int).tolist(),
                               "eaten": world.eaten})
    # Collection dominates; concentration gain only guides early evolution.
    reward = (10.0 * world.eaten +
              0.30 * (world.concentration(world.position) - initial_concentration) +
              0.005 * world.path_length)
    if record:
        return reward, {"config": vars(cfg), "resources": initial_resources.tolist(),
                        "trajectory": trajectory, "reward": float(reward), "eaten": world.eaten}
    return float(reward), world.eaten


def train(generations=35, population=32, seeds=range(1000, 1012), config=None,
          out=None, verbose=True):
    cfg = config or Config()
    circuit = SparseCircuit(cfg)
    rng = np.random.default_rng(2026)
    best = circuit.reflex_genome()
    best_score = -np.inf
    history = []
    for generation in range(generations):
        sigma = 0.18 * (0.97 ** generation) + 0.025
        candidates = [best] + [np.clip(best + rng.normal(0, sigma, best.shape), -4, 4)
                               for _ in range(population - 1)]
        scores = []
        eaten_counts = []
        for genome in candidates:
            results = [rollout(circuit, genome, seed, cfg) for seed in seeds]
            scores.append(float(np.mean([r[0] for r in results])))
            eaten_counts.append(float(np.mean([r[1] for r in results])))
        winner = int(np.argmax(scores))
        if scores[winner] >= best_score:
            best = candidates[winner].copy()
            best_score = scores[winner]
        row = {"generation": generation + 1, "best_reward": best_score,
               "generation_reward": scores[winner], "generation_eaten": eaten_counts[winner],
               "sigma": sigma}
        history.append(row)
        if verbose and (generation == 0 or (generation + 1) % 5 == 0):
            print(row, flush=True)
        if out is not None:
            save_checkpoint(out, best, history, cfg)
    return circuit, best, history


def save_checkpoint(out, genome, history, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "best_genome.npy", genome)
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(vars(config), indent=2), encoding="utf-8")


def evaluate(circuit, genome, seeds=range(3000, 3020)):
    results = [rollout(circuit, genome, seed) for seed in seeds]
    return {"episodes": len(results), "mean_eaten": float(np.mean([r[1] for r in results])),
            "median_eaten": float(np.median([r[1] for r in results])),
            "mean_reward": float(np.mean([r[0] for r in results]))}


def show_replay(replay):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from ipywidgets import IntSlider, Play, HBox, VBox, jslink, HTML, Output, Dropdown
    from IPython.display import display

    cfg = replay["config"]
    trajectory = replay["trajectory"]
    resources = np.asarray(replay["resources"])
    grid = np.linspace(0, cfg["world_size"], 120)
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack((xx, yy), -1)
    dist = np.linalg.norm(points[..., None, :] - resources, axis=-1)
    field = np.exp(-dist / cfg["diffusion_length"])
    field[field < cfg["cutoff"]] = 0
    field = field.sum(-1)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(field, origin="lower", extent=[0, cfg["world_size"]] * 2,
              cmap="YlGn", alpha=0.7)
    dots = ax.scatter(resources[:, 0], resources[:, 1], c="darkgreen", s=36)
    body = Circle((0, 0), 0.32, fc="black", ec="white", zorder=5)
    ax.add_patch(body)
    nose, = ax.plot([], [], color="white", lw=2, zorder=6)
    sensors, = ax.plot([], [], "o", color="cyan", ms=5, zorder=6)
    left_flag, = ax.plot([], [], color="blue", lw=2, zorder=5)
    right_flag, = ax.plot([], [], color="red", lw=2, zorder=5)
    trail, = ax.plot([], [], color="black", lw=1, alpha=0.5)
    ax.set(xlim=(0, cfg["world_size"]), ylim=(0, cfg["world_size"]),
           aspect="equal", title="Microbe nutrient world")
    status = HTML()
    slider = IntSlider(min=0, max=len(trajectory) - 1, step=1, value=0,
                       description="Frame", continuous_update=True,
                       layout={"width": "520px"})
    play = Play(min=0, max=len(trajectory) - 1, step=1, interval=45)
    jslink((play, "value"), (slider, "value"))
    speed = Dropdown(options=[("Slow", 180), ("Normal", 90), ("Fast", 35)],
                     value=90, description="Speed")
    speed.observe(lambda change: setattr(play, "interval", change["new"]), names="value")
    canvas = Output()

    def update(change):
        i = change["new"]
        f = trajectory[i]
        p = np.asarray(f["p"])
        angle = f["angle"]
        forward = np.array([np.cos(angle), np.sin(angle)])
        side = np.array([-forward[1], forward[0]])
        body.center = p
        nose.set_data([p[0], p[0] + 0.5 * forward[0]],
                      [p[1], p[1] + 0.5 * forward[1]])
        s = np.asarray(f["sensors"])
        sensors.set_data(s[:, 0], s[:, 1])
        for j, artist in enumerate((left_flag, right_flag)):
            root = p - 0.25 * forward + (0.24 if j == 0 else -0.24) * side
            tip = root - 0.8 * forward + f["swing"][j] * side
            artist.set_data([root[0], tip[0]], [root[1], tip[1]])
        history = np.asarray([v["p"] for v in trajectory[:i + 1]])
        trail.set_data(history[:, 0], history[:, 1])
        colors = ["darkgreen" if present else "lightgray" for present in f["remaining"]]
        dots.set_color(colors)
        status.value = (f"<b>Frame {i + 1}/{len(trajectory)}</b> · "
                        f"Eaten {f['eaten']} · Left sensor {f['signal'][0]:.2f} · "
                        f"Right sensor {f['signal'][1]:.2f}")
        with canvas:
            canvas.clear_output(wait=True)
            display(fig)

    slider.observe(update, names="value")
    update({"new": 0})
    display(VBox([HBox([play, slider, speed]), status, canvas]))
    return fig
