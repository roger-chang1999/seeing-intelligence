"""版本 B：Gradio 當外殼，畫面與滑桿在瀏覽器端用 JS 即時重畫；訓練仍由伺服器的 Python 執行。"""

from pathlib import Path

import gradio as gr

import core

HERE = Path(__file__).parent

HTML = """
<div class="bend">
  <div class="plot">
    <div class="topbar">
      <div class="scores">
        <div class="score-train"><i class="sw train"></i>練習：平均猜錯 <b>0</b> 人</div>
        <div class="score-test"><i class="sw test"></i>實戰：平均猜錯 <b>0</b> 人</div>
      </div>
      <div class="phase"><button class="practice on">練習</button><button class="exam">進入實戰</button></div>
    </div>
    <div class="canvas-wrap">
      <canvas class="main"></canvas>
      <div class="card-head">
        <span class="card-title">成績單</span>
        <span class="legend"><i class="sw train"></i>練習</span>
        <span class="legend legend-test"><i class="sw test"></i>實戰</span>
        <span class="card-hint">每按一次「讓機器自己學」就記一筆</span>
      </div>
      <canvas class="card"></canvas>
      <div class="tip"></div>
    </div>
  </div>
  <div class="side">
    <div class="controls"></div>
    <div class="count-row">
      神經元（彎折）數量 <b class="count">0</b>
      <button class="remove">－ 拿掉一個</button><button class="add">＋ 加一個</button>
    </div>
    <label class="check"><input type="checkbox" class="resid" checked> <span class="resid-label"></span></label>
    <button class="learn">讓機器自己學</button>
    <button class="sweep">講師用：一鍵掃描 0–12 個神經元</button>
  </div>
</div>
"""

CSS = """
.bend { display:flex; gap:24px; flex-wrap:wrap; font-family:'Noto Sans TC','Noto Sans CJK TC','Microsoft JhengHei','PingFang TC',sans-serif; color:#0b0b0b; }
.bend .plot { flex:3 1 520px; min-width:0; }
.bend .side { flex:2 1 300px; display:flex; flex-direction:column; gap:12px; }
.bend .topbar { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin:4px 0 10px; }
.bend .scores { display:flex; gap:20px; flex-wrap:wrap; font-size:17px; font-weight:600; font-variant-numeric:tabular-nums; }
.bend .sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
.bend .sw.train { background:#7a7975; }
.bend .sw.test { background:#1baf7a; }
.bend .phase { display:flex; border:1px solid #d5d4cf; border-radius:8px; overflow:hidden; }
.bend .phase button { border:none; border-radius:0; padding:8px 16px; background:#fff; color:#52514e; }
.bend .phase button.on { background:#0b0b0b; color:#fff; }
.bend .phase .exam.on { background:#1baf7a; }
.bend .canvas-wrap { position:relative; }
.bend canvas { display:block; touch-action:none; border-radius:8px; }
.bend .card-head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin:18px 0 4px; font-size:14px; color:#52514e; }
.bend .card-title { font-size:16px; font-weight:600; color:#0b0b0b; }
.bend .card-hint { margin-left:auto; font-size:12px; }
.bend .tip { position:absolute; display:none; pointer-events:none; background:#0b0b0b; color:#fff;
  font-size:13px; padding:6px 10px; border-radius:6px; white-space:nowrap; z-index:2; }
.bend .row { display:grid; grid-template-columns:1fr auto; gap:2px 8px; font-size:14px; color:#52514e; }
.bend .row input { grid-column:1 / -1; width:100%; accent-color:#2a78d6; }
.bend .val { font-variant-numeric:tabular-nums; color:#0b0b0b; }
.bend .neuron { border-left:3px solid #eb6834; padding:6px 0 6px 12px; margin-top:10px; display:flex; flex-direction:column; gap:8px; }
.bend .ntitle { font-weight:600; font-size:14px; }
.bend .neuron .row input { accent-color:#eb6834; }
.bend .auto-note { margin-top:10px; padding:8px 12px; border-left:3px solid #eb6834; background:#fdf1ec; font-size:14px; }
.bend .count-row { font-size:14px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.bend button { font:inherit; border:1px solid #d5d4cf; background:#fff; border-radius:8px; padding:6px 12px; cursor:pointer; }
.bend button:disabled { opacity:.4; cursor:default; }
.bend .learn { background:#2a78d6; color:#fff; border:none; font-size:16px; font-weight:600; padding:12px; }
.bend .sweep { font-size:13px; color:#52514e; }
.bend .check { font-size:14px; color:#52514e; }
"""


def train(payload):
    # Gradio 的 server function 只收一個參數，所以從 JS 傳一個物件過來
    return core.train(payload["params"], int(payload["n"]))


def sweep(_payload):
    return core.sweep()


def build():
    data = {
        "x": core.X.tolist(), "y": core.Y.tolist(), "xt": core.XT.tolist(), "yt": core.YT.tolist(),
        "xmin": core.X_MIN, "xmax": core.X_MAX, "ymin": core.Y_MIN, "ymax": core.Y_MAX,
        "maxN": core.MAX_NEURONS, "manualN": core.MANUAL_NEURONS, "params": core.default_params(),
    }
    with gr.Blocks(title="彎折工具・版本 B") as demo:
        gr.Markdown("## 彎折工具・版本 B（Gradio + 瀏覽器端 JS）\n一家店每個時段的來客數。先用 **練習資料**（灰點）讓藍線貼近；再按 **進入實戰**，看模型遇到沒看過的資料（綠點）表現如何。橘色彎折點可以直接用滑鼠拖。")
        gr.HTML(
            html_template=HTML,
            css_template=CSS,
            js_on_load=(HERE / "bend_widget.js").read_text(encoding="utf-8"),
            server_functions=[train, sweep],
            data=data,
        )
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
