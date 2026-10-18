// 版本 B 的瀏覽器端：畫面與滑桿都在學員電腦上即時重畫，訓練交給伺服器的 Python。
// Gradio 的 js_on_load 會提供 element、props、server 三個變數。

const D = props.data;
const p = JSON.parse(JSON.stringify(D.params));
let n = 0;
let showResid = true;
let busy = false;

const $ = (sel) => element.querySelector(sel);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const tip = $(".tip");
const H = 420;
const PAD = { l: 52, r: 18, t: 16, b: 40 };
const COLOR = {
  surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", grid: "#e6e5e0",
  curve: "#2a78d6", resid: "rgba(227,73,72,0.45)", bend: "#eb6834", point: "#7a7975",
};

// ---------- 模型（和 core.py 同一條公式） ----------
function predict(x) {
  let y = p.b + p.a * (x - D.xmin);
  for (let i = 0; i < n; i++) y += p.s[i] * Math.max(x - p.c[i], 0);
  return y;
}
function rmse() {
  let s = 0;
  D.x.forEach((x, i) => { const r = predict(x) - D.y[i]; s += r * r; });
  return Math.sqrt(s / D.x.length);
}

// ---------- 座標 ----------
let W = 600;
const sx = (x) => PAD.l + ((x - D.xmin + 0.3) / (D.xmax - D.xmin + 0.6)) * (W - PAD.l - PAD.r);
const sy = (y) => PAD.t + (1 - (y - D.ymin) / (D.ymax - D.ymin)) * (H - PAD.t - PAD.b);
const invx = (px) => D.xmin - 0.3 + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (D.xmax - D.xmin + 0.6);

function resize() {
  W = canvas.parentElement.clientWidth;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

// ---------- 繪圖 ----------
function draw() {
  ctx.fillStyle = COLOR.surface; ctx.fillRect(0, 0, W, H);
  ctx.font = "12px 'Noto Sans TC', 'Noto Sans CJK TC', 'Microsoft JhengHei', 'PingFang TC', sans-serif";
  ctx.fillStyle = COLOR.ink2; ctx.strokeStyle = COLOR.grid; ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let y = 0; y <= D.ymax; y += 20) {
    ctx.beginPath(); ctx.moveTo(PAD.l, sy(y)); ctx.lineTo(W - PAD.r, sy(y)); ctx.stroke();
    ctx.fillText(y, PAD.l - 8, sy(y));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let h = 8; h <= 22; h += 2) ctx.fillText(h + ":00", sx(h), H - PAD.b + 10);

  if (showResid) {
    ctx.strokeStyle = COLOR.resid; ctx.lineWidth = 1.5;
    D.x.forEach((x, i) => {
      ctx.beginPath(); ctx.moveTo(sx(x), sy(D.y[i])); ctx.lineTo(sx(x), sy(predict(x))); ctx.stroke();
    });
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(PAD.l, PAD.t, W - PAD.l - PAD.r, H - PAD.t - PAD.b); ctx.clip();
  ctx.strokeStyle = COLOR.curve; ctx.lineWidth = 2.5; ctx.lineJoin = "round";
  ctx.beginPath();
  for (let k = 0; k <= 300; k++) {
    const x = D.xmin + (k / 300) * (D.xmax - D.xmin);
    k ? ctx.lineTo(sx(x), sy(predict(x))) : ctx.moveTo(sx(x), sy(predict(x)));
  }
  ctx.stroke();
  ctx.restore();

  D.x.forEach((x, i) => dot(sx(x), sy(D.y[i]), 4.5, COLOR.point, 1.5));
  for (let i = 0; i < n; i++) dot(sx(p.c[i]), sy(predict(p.c[i])), 7, COLOR.bend, 2.5);

  $(".headline").textContent = `平均每個時段猜錯約 ${rmse().toFixed(1)} 位客人`;
}
function dot(x, y, r, fill, ring) {
  ctx.beginPath(); ctx.arc(x, y, r + ring, 0, 2 * Math.PI); ctx.fillStyle = COLOR.surface; ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fillStyle = fill; ctx.fill();
}

// ---------- 控制面板 ----------
const panel = $(".controls");
function slider(label, min, max, step, get, set) {
  const row = document.createElement("label");
  row.className = "row";
  row.innerHTML = `<span class="lab">${label}</span><span class="val"></span><input type="range" min="${min}" max="${max}" step="${step}">`;
  const input = row.querySelector("input"), val = row.querySelector(".val");
  const sync = () => { input.value = get(); val.textContent = (+get()).toFixed(1); };
  input.addEventListener("input", () => { if (busy) return; set(+input.value); val.textContent = (+input.value).toFixed(1); draw(); });
  sync();
  return { row, sync };
}
const rows = [
  slider("早上 8 點的基本人數", 0, 100, 0.5, () => p.b, (v) => (p.b = v)),
  slider("每過一小時，人數變化", -20, 20, 0.1, () => p.a, (v) => (p.a = v)),
];
rows.forEach((r) => panel.appendChild(r.row));
const neuronBox = document.createElement("div");
panel.appendChild(neuronBox);
const neuronRows = [];
for (let i = 0; i < D.maxN; i++) {
  const g = document.createElement("div");
  g.className = "neuron";
  g.innerHTML = `<div class="ntitle"><i></i>神經元 ${i + 1}</div>`;
  const rc = slider("從幾點開始彎", D.xmin, D.xmax, 0.1, () => p.c[i], (v) => (p.c[i] = v));
  const rs = slider("彎多大力", -60, 60, 0.5, () => p.s[i], (v) => (p.s[i] = v));
  g.append(rc.row, rs.row);
  neuronBox.appendChild(g);
  neuronRows.push({ g, rc, rs });
}
const syncAll = () => {
  rows.forEach((r) => r.sync());
  neuronRows.forEach((r, i) => { r.rc.sync(); r.rs.sync(); r.g.style.display = i < n ? "" : "none"; });
  $(".count").textContent = n;
  $(".add").disabled = busy || n >= D.maxN;
  $(".remove").disabled = busy || n <= 0;
  $(".learn").disabled = busy;
};

$(".add").addEventListener("click", () => { n++; syncAll(); draw(); });
$(".remove").addEventListener("click", () => { n--; p.s[n] = 0; syncAll(); draw(); });
$(".resid").addEventListener("change", (e) => { showResid = e.target.checked; draw(); });

// ---------- 讓機器自己學：伺服器的 Python 訓練，瀏覽器在影格之間平滑內插 ----------
$(".learn").addEventListener("click", async () => {
  busy = true; syncAll();
  $(".learn").textContent = "機器學習中…";
  const frames = await server.train({ params: p, n });
  const total = 2500, t0 = performance.now();
  const lerp = (u, v, t) => u + (v - u) * t;
  await new Promise((done) => {
    function tick(now) {
      const f = Math.min((now - t0) / total, 1) * (frames.length - 1);
      const i = Math.min(Math.floor(f), frames.length - 2), t = f - i;
      const A = frames[i].params, B = frames[i + 1].params;
      p.b = lerp(A.b, B.b, t); p.a = lerp(A.a, B.a, t);
      for (let k = 0; k < D.maxN; k++) { p.c[k] = lerp(A.c[k], B.c[k], t); p.s[k] = lerp(A.s[k], B.s[k], t); }
      draw(); syncAll();
      f < frames.length - 1 ? requestAnimationFrame(tick) : done();
    }
    requestAnimationFrame(tick);
  });
  busy = false; syncAll();
  $(".learn").textContent = "讓機器自己學";
});

// ---------- 直接拖動彎折點、滑過資料點看提示 ----------
let dragging = -1;
const hitBend = (mx, my) => {
  for (let i = 0; i < n; i++) if (Math.hypot(mx - sx(p.c[i]), my - sy(predict(p.c[i]))) < 14) return i;
  return -1;
};
const fmtTime = (x) => `${Math.floor(x)}:${String(Math.round((x % 1) * 60)).padStart(2, "0")}`;
canvas.addEventListener("pointerdown", (e) => {
  if (busy) return;
  dragging = hitBend(e.offsetX, e.offsetY);
  if (dragging >= 0) canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  const mx = e.offsetX, my = e.offsetY;
  if (dragging >= 0) {
    p.c[dragging] = Math.min(D.xmax, Math.max(D.xmin, invx(mx)));
    neuronRows[dragging].rc.sync(); draw();
  }
  const b = dragging >= 0 ? dragging : hitBend(mx, my);
  canvas.style.cursor = b >= 0 ? "ew-resize" : "default";
  let text = "";
  if (b >= 0) text = `神經元 ${b + 1}：從 ${fmtTime(p.c[b])} 開始彎（左右拖動）`;
  else {
    let best = -1, bd = 12;
    D.x.forEach((x, i) => { const d = Math.hypot(mx - sx(x), my - sy(D.y[i])); if (d < bd) { bd = d; best = i; } });
    if (best >= 0) {
      const x = D.x[best];
      text = `${fmtTime(x)}　實際 ${D.y[best].toFixed(0)} 人　預測 ${predict(x).toFixed(0)} 人`;
    }
  }
  tip.style.display = text ? "block" : "none";
  tip.textContent = text;
  tip.style.left = Math.min(mx + 14, W - 230) + "px";
  tip.style.top = my + 14 + "px";
});
canvas.addEventListener("pointerup", () => { dragging = -1; });
canvas.addEventListener("pointerleave", () => { if (dragging < 0) tip.style.display = "none"; });

new ResizeObserver(resize).observe(canvas.parentElement);
syncAll();
resize();
