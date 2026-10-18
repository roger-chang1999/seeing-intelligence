"""版本 B：Gradio 當外殼，畫面與滑桿在瀏覽器端用 JS 即時重畫；訓練仍由伺服器的 Python 執行。"""

from pathlib import Path

import gradio as gr

import core

HERE = Path(__file__).parent

HTML = """
<div class="bend">
  <div class="plot">
    <div class="headline"></div>
    <div class="canvas-wrap"><canvas></canvas><div class="tip"></div></div>
  </div>
  <div class="side">
    <div class="controls"></div>
    <div class="count-row">
      神經元（彎折）數量 <b class="count">0</b>
      <button class="remove">－ 拿掉一個</button><button class="add">＋ 加一個</button>
    </div>
    <label class="check"><input type="checkbox" class="resid" checked> 顯示每個點猜錯多少（紅線）</label>
    <button class="learn">讓機器自己學</button>
  </div>
</div>
"""

CSS = """
.bend { display:flex; gap:24px; flex-wrap:wrap; font-family:'Noto Sans TC','Noto Sans CJK TC','Microsoft JhengHei','PingFang TC',sans-serif; color:#0b0b0b; }
.bend .plot { flex:3 1 520px; min-width:0; }
.bend .side { flex:2 1 300px; display:flex; flex-direction:column; gap:12px; }
.bend .headline { font-size:18px; font-weight:600; margin:4px 0 8px; font-variant-numeric:tabular-nums; }
.bend .canvas-wrap { position:relative; border-radius:8px; overflow:hidden; }
.bend canvas { display:block; touch-action:none; }
.bend .tip { position:absolute; display:none; pointer-events:none; background:#0b0b0b; color:#fff;
  font-size:13px; padding:6px 10px; border-radius:6px; white-space:nowrap; }
.bend .row { display:grid; grid-template-columns:1fr auto; gap:2px 8px; font-size:14px; color:#52514e; }
.bend .row input { grid-column:1 / -1; width:100%; accent-color:#2a78d6; }
.bend .val { font-variant-numeric:tabular-nums; color:#0b0b0b; }
.bend .neuron { border-left:3px solid #eb6834; padding:6px 0 6px 12px; margin-top:10px; display:flex; flex-direction:column; gap:8px; }
.bend .ntitle { font-weight:600; font-size:14px; }
.bend .neuron .row input { accent-color:#eb6834; }
.bend .count-row { font-size:14px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.bend button { font:inherit; border:1px solid #d5d4cf; background:#fff; border-radius:8px; padding:6px 12px; cursor:pointer; }
.bend button:disabled { opacity:.4; cursor:default; }
.bend .learn { background:#2a78d6; color:#fff; border:none; font-size:16px; font-weight:600; padding:12px; }
.bend .check { font-size:14px; color:#52514e; }
"""


def train(payload):
    # Gradio 的 server function 只收一個參數，所以從 JS 傳一個物件過來
    return core.train(payload["params"], int(payload["n"]))


def build():
    data = {
        "x": core.X.tolist(), "y": core.Y.tolist(),
        "xmin": core.X_MIN, "xmax": core.X_MAX, "ymin": core.Y_MIN, "ymax": core.Y_MAX,
        "maxN": core.MAX_NEURONS, "params": core.default_params(),
    }
    with gr.Blocks(title="彎折工具・版本 B") as demo:
        gr.Markdown("## 彎折工具・版本 B（Gradio + 瀏覽器端 JS）\n一家店每個時段的來客數。試著調整參數，讓藍線貼近灰色的點。橘色彎折點可以直接用滑鼠拖。")
        gr.HTML(
            html_template=HTML,
            css_template=CSS,
            js_on_load=(HERE / "bend_widget.js").read_text(encoding="utf-8"),
            server_functions=[train],
            data=data,
        )
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
