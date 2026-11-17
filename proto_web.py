"""Gradio front end for the 100 Hz bilateral nerve-net experiment."""

from dataclasses import replace

import gradio as gr

from proto_microbe import ProtoConfig, rollout, replay_figure


def create_demo(net, learned_genome=None):
    def simulate(seed, seconds, resources, mode):
        cfg = replace(net.cfg, steps=int(seconds * 100), resource_count=int(resources))
        genome = (learned_genome if mode == "Trained" and learned_genome is not None
                  else net.reflex_genome())
        eaten, replay = rollout(net, genome, int(seed), cfg=cfg, record=True)
        plot = replay_figure(replay, every=max(1, cfg.steps // 120))
        return plot, (f"{eaten:.0f}/{cfg.resource_count} resources eaten · "
                      f"{seconds:.0f} s · 100 neural/world ticks per second · "
                      f"{net.n} neurons · {net.synapses} synapses")

    with gr.Blocks(title="Bilateral microbe · 100 Hz") as demo:
        gr.Markdown("# Two flagella, two sensors, sparse bilateral nerve net\n"
                    "Each run simulates the neural circuit and world sequentially at 100 ticks/s. "
                    "Play and pause the replay, or drag the tick slider.")
        with gr.Row():
            seed = gr.Slider(0, 9999, value=3000, step=1, label="World seed")
            seconds = gr.Slider(5, 60, value=30, step=1, label="Simulation seconds")
            resources = gr.Slider(30, 240, value=net.cfg.resource_count,
                                  step=1, label="Resources")
            modes = ["Reflex"] + (["Trained"] if learned_genome is not None else [])
            mode = gr.Radio(modes, value=modes[-1], label="Controller")
        run = gr.Button("Run simulation", variant="primary")
        stats = gr.Textbox(label="Result")
        plot = gr.Plot(label="Browser replay")
        run.click(simulate, inputs=[seed, seconds, resources, mode],
                  outputs=[plot, stats])
    return demo
