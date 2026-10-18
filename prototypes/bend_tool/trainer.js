// 訓練：core.py 的 train / learn / sweep 移植到瀏覽器，同一套演算法（Adam 梯度下降）。
// 需要全域的 DATA（由 build.py 從 core.py 產生）。

const STEPS = 4000;

function predictWith(x, q, k) {
  let y = q.b + q.a * (x - DATA.xmin);
  for (let i = 0; i < k; i++) y += q.s[i] * Math.max(x - q.c[i], 0);
  return y;
}

function rmseWith(xs, ys, q, k) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) { const r = predictWith(xs[i], q, k) - ys[i]; s += r * r; }
  return Math.sqrt(s / xs.length);
}

const clone = (q) => ({ b: q.b, a: q.a, c: q.c.slice(), s: q.s.slice() });

function snapshot(q, k) {
  return { params: clone(q), train: rmseWith(DATA.x, DATA.y, q, k), test: rmseWith(DATA.xt, DATA.yt, q, k) };
}

// 從 p 出發訓練前 n 個神經元，回傳動畫用的參數快照
function train(p, n, steps = STEPS, nFrames = 60) {
  const q = clone(p);
  const X = DATA.x, Y = DATA.y, N = X.length;
  const lr = { b: 1.0, a: 0.3, c: 0.05, s: 0.4 };
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const m = { b: 0, a: 0, c: new Array(n).fill(0), s: new Array(n).fill(0) };
  const v = { b: 0, a: 0, c: new Array(n).fill(0), s: new Array(n).fill(0) };

  // 前段變化最劇烈，快照取樣用平方分布讓動畫前段細、後段疏
  const keep = new Set();
  for (let i = 0; i < nFrames; i++) keep.add(Math.round((i / (nFrames - 1)) ** 2 * steps));
  const frames = keep.has(0) ? [snapshot(q, n)] : [];

  const gc = new Array(n), gs = new Array(n), r = new Array(N);
  for (let t = 1; t <= steps; t++) {
    let gb = 0, ga = 0;
    gc.fill(0); gs.fill(0);
    for (let j = 0; j < N; j++) {
      r[j] = predictWith(X[j], q, n) - Y[j];
      gb += 2 * r[j] / N;
      ga += 2 * r[j] * (X[j] - DATA.xmin) / N;
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < N; j++) {
        const h = X[j] - q.c[i];
        if (h > 0) { gs[i] += 2 * r[j] * h / N; gc[i] -= 2 * r[j] * q.s[i] / N; }
      }
    }
    const c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
    const step = (g, key, idx) => {
      if (idx === undefined) {
        m[key] = b1 * m[key] + (1 - b1) * g; v[key] = b2 * v[key] + (1 - b2) * g * g;
        return lr[key] * (m[key] / c1) / (Math.sqrt(v[key] / c2) + eps);
      }
      m[key][idx] = b1 * m[key][idx] + (1 - b1) * g; v[key][idx] = b2 * v[key][idx] + (1 - b2) * g * g;
      return lr[key] * (m[key][idx] / c1) / (Math.sqrt(v[key][idx] / c2) + eps);
    };
    q.b -= step(gb, "b"); q.a -= step(ga, "a");
    for (let i = 0; i < n; i++) {
      q.s[i] -= step(gs[i], "s", i);
      q.c[i] = Math.min(DATA.xmax, Math.max(DATA.xmin, q.c[i] - step(gc[i], "c", i)));
    }
    if (keep.has(t)) frames.push(snapshot(q, n));
  }
  return frames;
}

// 「讓機器自己學」：從學員當下的狀態學，但避免卡在局部最小值（與 core.learn 相同）
function learn(p, n) {
  const q = clone(p), dflt = clone(DATA.params);
  const xmax = Math.max(...DATA.x);
  for (let i = 0; i < n; i++) if (q.c[i] > xmax) { q.c[i] = dflt.c[i]; q.s[i] = 0; }
  const mine = train(q, n), fresh = train(dflt, n);
  if (fresh[fresh.length - 1].train < mine[mine.length - 1].train - 0.5) {
    return { frames: [snapshot(p, n), ...fresh], restarted: true };
  }
  return { frames: mine, restarted: false };
}

// 講師用：0 到 maxN 個神經元，各從預設起點訓練一次
function sweep() {
  const out = [];
  for (let n = 0; n <= DATA.maxN; n++) out.push({ n, ...train(DATA.params, n, STEPS, 2).pop() });
  return out;
}
