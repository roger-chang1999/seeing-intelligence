"""版本 A：純 Gradio。每動一次滑桿，伺服器用 matplotlib 重畫一張圖。"""

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import core

plt.rcParams["font.family"] = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CURVE, RESID, BEND, POINT = "#2a78d6", "#e34948", "#eb6834", "#7a7975"


def draw(p, n, show_resid):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    pred = core.predict(core.X, p, n)
    if show_resid:
        ax.vlines(core.X, core.Y, pred, color=RESID, alpha=0.45, lw=1.5, zorder=1)
    ax.scatter(core.X, core.Y, s=34, color=POINT, edgecolor="#fcfcfb", lw=1.5, zorder=3)
    xs = np.linspace(core.X_MIN, core.X_MAX, 400)
    ax.plot(xs, core.predict(xs, p, n), color=CURVE, lw=2.5, zorder=2)
    for c in p["c"][:n]:
        ax.scatter([c], core.predict([c], p, n), s=90, color=BEND, edgecolor="white", lw=2, zorder=4)
    ax.set_xlim(core.X_MIN - 0.3, core.X_MAX + 0.3)
    ax.set_ylim(core.Y_MIN, core.Y_MAX)
    ax.set_xticks(range(8, 23, 2), [f"{h}:00" for h in range(8, 23, 2)])
    ax.set_xlabel("時間", color=INK2)
    ax.set_ylabel("來客數（人）", color=INK2)
    ax.grid(axis="y", color=GRID, lw=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_title(f"平均每個時段猜錯約 {core.rmse(p, n):.1f} 位客人", loc="left", color=INK, fontsize=14, pad=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


def collect(b, a, n, show_resid, *neuron_vals):
    p = core.default_params()
    p["b"], p["a"] = b, a
    p["c"] = list(neuron_vals[0::2])
    p["s"] = list(neuron_vals[1::2])
    return p, int(n), show_resid


def on_change(*args):
    return draw(*collect(*args))


def on_count(n):
    return [gr.update(visible=i < n) for i in range(core.MAX_NEURONS)]


def on_learn(*args):
    p, n, show_resid = collect(*args)
    for f in core.train(p, n):
        q = f["params"]
        neuron_updates = []
        for c, s in zip(q["c"], q["s"]):
            neuron_updates += [round(c, 2), round(s, 2)]
        yield [draw(q, n, show_resid), round(q["b"], 2), round(q["a"], 2), *neuron_updates]


def build():
    p0 = core.default_params()
    with gr.Blocks(title="彎折工具・版本 A") as demo:
        gr.Markdown("## 彎折工具・版本 A（純 Gradio）\n一家店每個時段的來客數。試著調整參數，讓藍線貼近灰色的點。")
        with gr.Row():
            with gr.Column(scale=3):
                plot = gr.Plot(draw(p0, 0, True), show_label=False)
            with gr.Column(scale=2):
                b = gr.Slider(0, 100, value=p0["b"], step=0.5, label="早上 8 點的基本人數")
                a = gr.Slider(-20, 20, value=p0["a"], step=0.1, label="每過一小時，人數變化")
                n = gr.Slider(0, core.MAX_NEURONS, value=0, step=1, label="神經元（彎折）數量")
                resid = gr.Checkbox(True, label="顯示每個點猜錯多少（紅線）")
                groups, neuron_inputs = [], []
                for i in range(core.MAX_NEURONS):
                    with gr.Group(visible=False) as g:
                        c = gr.Slider(core.X_MIN, core.X_MAX, value=p0["c"][i], step=0.1, label=f"神經元 {i + 1}：從幾點開始彎")
                        s = gr.Slider(-60, 60, value=0, step=0.5, label=f"神經元 {i + 1}：彎多大力")
                    groups.append(g)
                    neuron_inputs += [c, s]
                learn = gr.Button("讓機器自己學", variant="primary")

        inputs = [b, a, n, resid, *neuron_inputs]
        for comp in inputs:
            comp.input(on_change, inputs, plot, trigger_mode="always_last", show_progress="hidden")
        n.input(on_count, n, groups)
        learn.click(on_learn, inputs, [plot, b, a, *neuron_inputs], show_progress="hidden")
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
