// 版本 B 的瀏覽器端：畫面與滑桿都在學員電腦上即時重畫，訓練交給伺服器的 Python。
// Gradio 的 js_on_load 會提供 element、props、server 三個變數。

const D = props.data;
const p = JSON.parse(JSON.stringify(D.params));
let n = 0;
let showResid = true;
let exam = false;      // false = 練習，true = 實戰（揭曉實戰資料）
let busy = false;
const records = {};    // 成績單：records[神經元數] = { train, test, params }

const $ = (sel) => element.querySelector(sel);
const canvas = $(".main");
const ctx = canvas.getContext("2d");
const card = $(".card");
const cctx = card.getContext("2d");
const tip = $(".tip");
const H = 400, CH = 210;
const PAD = { l: 52, r: 18, t: 16, b: 40 };
const CPAD = { l: 52, r: 18, t: 28, b: 40 };
const FONT = "'Noto Sans TC', 'Noto Sans CJK TC', 'Microsoft JhengHei', 'PingFang TC', sans-serif";
const COLOR = {
  surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", grid: "#e6e5e0", band: "rgba(42,120,214,0.07)",
  curve: "#2a78d6", bend: "#eb6834",
  train: "#7a7975", trainResid: "rgba(227,73,72,0.45)",
  test: "#1baf7a", testResid: "rgba(27,175,122,0.45)",
};

// ---------- 模型（和 core.py 同一條公式） ----------
function predict(x, q = p, k = n) {
  let y = q.b + q.a * (x - D.xmin);
  for (let i = 0; i < k; i++) y += q.s[i] * Math.max(x - q.c[i], 0);
  return y;
}
function rmse(xs, ys) {
  let s = 0;
  xs.forEach((x, i) => { const r = predict(x) - ys[i]; s += r * r; });
  return Math.sqrt(s / xs.length);
}

// ---------- 主圖座標 ----------
let W = 600, CW = 600;
const sx = (x) => PAD.l + ((x - D.xmin + 0.3) / (D.xmax - D.xmin + 0.6)) * (W - PAD.l - PAD.r);
const sy = (y) => PAD.t + (1 - (y - D.ymin) / (D.ymax - D.ymin)) * (H - PAD.t - PAD.b);
const invx = (px) => D.xmin - 0.3 + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (D.xmax - D.xmin + 0.6);

function setup(cv, c2, w, h) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + "px"; cv.style.height = h + "px";
  c2.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function resize() {
  W = canvas.parentElement.clientWidth; CW = card.parentElement.clientWidth;
  setup(canvas, ctx, W, H); setup(card, cctx, CW, CH);
  draw();
}

function dot(c, x, y, r, fill, ring) {
  c.beginPath(); c.arc(x, y, r + ring, 0, 2 * Math.PI); c.fillStyle = COLOR.surface; c.fill();
  c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fillStyle = fill; c.fill();
}
function resid(xs, ys, color) {
  ctx.strokeStyle = color; ctx.lineWidth = 1.5;
  xs.forEach((x, i) => { ctx.beginPath(); ctx.moveTo(sx(x), sy(ys[i])); ctx.lineTo(sx(x), sy(predict(x))); ctx.stroke(); });
}

// ---------- 主圖 ----------
function drawMain() {
  ctx.fillStyle = COLOR.surface; ctx.fillRect(0, 0, W, H);
  ctx.font = `12px ${FONT}`;
  ctx.fillStyle = COLOR.ink2; ctx.strokeStyle = COLOR.grid; ctx.lineWidth = 1;
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let y = 0; y <= D.ymax; y += 20) {
    ctx.beginPath(); ctx.moveTo(PAD.l, sy(y)); ctx.lineTo(W - PAD.r, sy(y)); ctx.stroke();
    ctx.fillText(y, PAD.l - 8, sy(y));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let h = 8; h <= 22; h += 2) ctx.fillText(h + ":00", sx(h), H - PAD.b + 10);

  if (showResid) exam ? resid(D.xt, D.yt, COLOR.testResid) : resid(D.x, D.y, COLOR.trainResid);

  ctx.save();
  ctx.beginPath(); ctx.rect(PAD.l, PAD.t, W - PAD.l - PAD.r, H - PAD.t - PAD.b); ctx.clip();
  ctx.strokeStyle = COLOR.curve; ctx.lineWidth = 2.5; ctx.lineJoin = "round";
  ctx.beginPath();
  for (let k = 0; k <= 400; k++) {
    const x = D.xmin + (k / 400) * (D.xmax - D.xmin);
    k ? ctx.lineTo(sx(x), sy(predict(x))) : ctx.moveTo(sx(x), sy(predict(x)));
  }
  ctx.stroke();
  ctx.restore();

  if (exam) D.xt.forEach((x, i) => dot(ctx, sx(x), sy(D.yt[i]), 4, COLOR.test, 1.5));
  D.x.forEach((x, i) => dot(ctx, sx(x), sy(D.y[i]), 5.5, COLOR.train, 2));
  for (let i = 0; i < n; i++) dot(ctx, sx(p.c[i]), sy(predict(p.c[i])), 6, COLOR.bend, 2.5);
}

// ---------- 成績單：每個神經元數量的練習 / 實戰誤差 ----------
let cardBars = [];
function drawCard() {
  cctx.fillStyle = COLOR.surface; cctx.fillRect(0, 0, CW, CH);
  const ns = [...Array(D.maxN + 1).keys()];
  const ymax = 30;
  const ch = CH - CPAD.t - CPAD.b, cw = CW - CPAD.l - CPAD.r;
  const gy = (v) => CPAD.t + (1 - Math.min(v, ymax) / ymax) * ch;
  const slot = cw / ns.length, bw = Math.min(14, (slot - 8) / 2);
  cctx.font = `12px ${FONT}`;
  cctx.strokeStyle = COLOR.grid; cctx.lineWidth = 1; cctx.fillStyle = COLOR.ink2;
  cctx.textAlign = "right"; cctx.textBaseline = "middle";
  for (let v = 0; v <= ymax; v += 10) {
    cctx.beginPath(); cctx.moveTo(CPAD.l, gy(v)); cctx.lineTo(CW - CPAD.r, gy(v)); cctx.stroke();
    cctx.fillText(v, CPAD.l - 8, gy(v));
  }
  // 目前神經元數量的欄位底色
  cctx.fillStyle = COLOR.band;
  cctx.fillRect(CPAD.l + n * slot, CPAD.t, slot, ch);

  // 實戰最佳
  let best = -1;
  if (exam) ns.forEach((k) => { if (records[k] && (best < 0 || records[k].test < records[best].test)) best = k; });

  cardBars = [];
  ns.forEach((k) => {
    const cx = CPAD.l + (k + 0.5) * slot;
    cctx.fillStyle = k === n ? COLOR.ink : COLOR.ink2;
    cctx.textAlign = "center"; cctx.textBaseline = "top";
    cctx.fillText(k, cx, CH - CPAD.b + 8);
    const r = records[k];
    if (!r) return;
    const bars = [[r.train, COLOR.train, cx - bw - 1]];
    if (exam) bars.push([r.test, COLOR.test, cx + 1]);
    else bars[0][2] = cx - bw / 2;
    bars.forEach(([v, color, x]) => {
      const top = gy(v), h = CPAD.t + ch - top;
      cctx.fillStyle = color;
      cctx.beginPath(); cctx.roundRect(x, top, bw, Math.max(h, 1), [4, 4, 0, 0]); cctx.fill();
    });
    cardBars.push({ k, x0: cx - slot / 2, x1: cx + slot / 2 });
    if (k === best) {
      cctx.fillStyle = COLOR.test; cctx.textAlign = "center"; cctx.textBaseline = "bottom";
      cctx.font = `600 12px ${FONT}`;
      cctx.fillText("★ 實戰最佳", cx, gy(Math.max(r.train, r.test)) - 4);
      cctx.font = `12px ${FONT}`;
    }
  });
  cctx.fillStyle = COLOR.ink2; cctx.textAlign = "center"; cctx.textBaseline = "top";
  cctx.fillText("神經元（彎折）數量", CPAD.l + cw / 2, CH - 16);
}

function draw() {
  drawMain();
  drawCard();
  $(".score-train b").textContent = rmse(D.x, D.y).toFixed(1);
  $(".score-test b").textContent = rmse(D.xt, D.yt).toFixed(1);
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
const neuronRows = [];
for (let i = 0; i < D.manualN; i++) {
  const g = document.createElement("div");
  g.className = "neuron";
  g.innerHTML = `<div class="ntitle">神經元 ${i + 1}</div>`;
  const rc = slider("從幾點開始彎", D.xmin, D.xmax, 0.1, () => p.c[i], (v) => (p.c[i] = v));
  const rs = slider("彎多大力", -80, 80, 0.5, () => p.s[i], (v) => (p.s[i] = v));
  g.append(rc.row, rs.row);
  panel.appendChild(g);
  neuronRows.push({ g, rc, rs });
}
const autoNote = document.createElement("div");
autoNote.className = "auto-note";
panel.appendChild(autoNote);

const syncAll = () => {
  rows.forEach((r) => r.sync());
  neuronRows.forEach((r, i) => { r.rc.sync(); r.rs.sync(); r.g.style.display = i < n ? "" : "none"; });
  const extra = n - D.manualN;
  autoNote.style.display = extra > 0 ? "" : "none";
  autoNote.textContent = `另外 ${extra} 個神經元太多了，只能交給機器學`;
  $(".count").textContent = n;
  $(".add").disabled = busy || n >= D.maxN;
  $(".remove").disabled = busy || n <= 0;
  element.querySelectorAll(".learn, .sweep, .phase button").forEach((b) => (b.disabled = busy));
  $(".phase .practice").classList.toggle("on", !exam);
  $(".phase .exam").classList.toggle("on", exam);
  $(".score-test").style.display = exam ? "" : "none";
  $(".legend-test").style.display = exam ? "" : "none";
  $(".resid-label").textContent = exam ? "顯示每個實戰點猜錯多少（綠線）" : "顯示每個練習點猜錯多少（紅線）";
};

$(".add").addEventListener("click", () => { n++; syncAll(); draw(); });
$(".remove").addEventListener("click", () => { n--; p.s[n] = 0; p.c[n] = D.params.c[n]; syncAll(); draw(); });
$(".resid").addEventListener("change", (e) => { showResid = e.target.checked; draw(); });
$(".phase .practice").addEventListener("click", () => { exam = false; syncAll(); draw(); });
$(".phase .exam").addEventListener("click", () => { exam = true; syncAll(); draw(); });

// ---------- 讓機器自己學：伺服器的 Python 訓練，瀏覽器在影格之間平滑內插 ----------
function setParams(q) {
  p.b = q.b; p.a = q.a;
  for (let k = 0; k < D.maxN; k++) { p.c[k] = q.c[k]; p.s[k] = q.s[k]; }
}
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
  const last = frames[frames.length - 1];
  records[n] = { train: last.train, test: last.test, params: JSON.parse(JSON.stringify(last.params)) };
  busy = false; syncAll(); draw();
  $(".learn").textContent = "讓機器自己學";
});

// ---------- 講師用：一鍵掃描 0–12 個神經元，逐欄長出成績單 ----------
$(".sweep").addEventListener("click", async () => {
  busy = true; syncAll();
  $(".sweep").textContent = "掃描中…";
  const results = await server.sweep({});
  for (const r of results) {
    records[r.n] = { train: r.train, test: r.test, params: r.params };
    drawCard();
    await new Promise((ok) => setTimeout(ok, 120));
  }
  busy = false; syncAll(); draw();
  $(".sweep").textContent = "講師用：一鍵掃描 0–12 個神經元";
});

// ---------- 主圖：直接拖動彎折點、滑過資料點看提示 ----------
let dragging = -1;
const hitBend = (mx, my) => {
  for (let i = 0; i < n; i++) if (Math.hypot(mx - sx(p.c[i]), my - sy(predict(p.c[i]))) < 14) return i;
  return -1;
};
const fmtTime = (x) => { const m = Math.round(x * 60); return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`; };
function showTip(text, x, y, host) {
  tip.style.display = text ? "block" : "none";
  tip.textContent = text;
  const hostRect = host.getBoundingClientRect(), wrap = tip.parentElement.getBoundingClientRect();
  tip.style.left = Math.min(x + hostRect.left - wrap.left + 14, wrap.width - 260) + "px";
  tip.style.top = y + hostRect.top - wrap.top + 14 + "px";
}
function nearest(xs, ys, mx, my) {
  let best = -1, bd = 12;
  xs.forEach((x, i) => { const d = Math.hypot(mx - sx(x), my - sy(ys[i])); if (d < bd) { bd = d; best = i; } });
  return best;
}
canvas.addEventListener("pointerdown", (e) => {
  if (busy) return;
  dragging = hitBend(e.offsetX, e.offsetY);
  if (dragging >= 0) canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  const mx = e.offsetX, my = e.offsetY;
  if (dragging >= 0) {
    p.c[dragging] = Math.min(D.xmax, Math.max(D.xmin, invx(mx)));
    if (dragging < D.manualN) neuronRows[dragging].rc.sync();
    draw();
  }
  const b = dragging >= 0 ? dragging : hitBend(mx, my);
  canvas.style.cursor = b >= 0 ? "ew-resize" : "default";
  let text = "";
  if (b >= 0) text = `神經元 ${b + 1}：從 ${fmtTime(p.c[b])} 開始彎（左右拖動）`;
  else {
    const i = nearest(D.x, D.y, mx, my);
    const j = exam ? nearest(D.xt, D.yt, mx, my) : -1;
    if (i >= 0) text = `練習｜${fmtTime(D.x[i])}　實際 ${D.y[i].toFixed(0)} 人　預測 ${predict(D.x[i]).toFixed(0)} 人`;
    else if (j >= 0) text = `實戰｜${fmtTime(D.xt[j])}　實際 ${D.yt[j].toFixed(0)} 人　預測 ${predict(D.xt[j]).toFixed(0)} 人`;
  }
  showTip(text, mx, my, canvas);
});
canvas.addEventListener("pointerup", () => { dragging = -1; });
canvas.addEventListener("pointerleave", () => { if (dragging < 0) tip.style.display = "none"; });

// ---------- 成績單：滑過看數字、點一下載入那個模型 ----------
const barAt = (mx) => cardBars.find((b) => mx >= b.x0 && mx < b.x1);
card.addEventListener("pointermove", (e) => {
  const b = barAt(e.offsetX);
  card.style.cursor = b && !busy ? "pointer" : "default";
  if (!b) return showTip("", 0, 0, card);
  const r = records[b.k];
  let text = `${b.k} 個神經元　練習猜錯 ${r.train.toFixed(1)} 人`;
  if (exam) text += `　實戰猜錯 ${r.test.toFixed(1)} 人`;
  showTip(text + "（點一下看這個模型）", e.offsetX, e.offsetY, card);
});
card.addEventListener("pointerleave", () => { tip.style.display = "none"; });
card.addEventListener("click", (e) => {
  const b = barAt(e.offsetX);
  if (!b || busy) return;
  n = b.k; setParams(records[b.k].params);
  syncAll(); draw();
});

new ResizeObserver(resize).observe(element.querySelector(".plot"));
syncAll();
resize();
