"""「彎折」工具共用核心：資料、模型、訓練。

故事情境：一家店從早上 8 點到晚上 10 點，每個時段的來客數。
午餐、晚餐各有一個尖峰 —— 一條直線怎麼調都貼不上。

模型：
    預測人數 = 基本人數 b + 每小時變化 a × (時間 − 8)
             + Σ 彎折力度 s_i × ReLU(時間 − 折點 c_i)

每個神經元 = 一個彎折：從時間 c_i 開始，斜率多改變 s_i。

資料分兩份：
    練習資料：10 筆，模型只看得到這些
    實戰資料：40 筆，按下「進入實戰」才揭曉
固定用 seed 5 —— 從 60 組裡驗證過，從預設起點訓練時：
0–2 個神經元兩邊都爛（欠擬合）、3 個剛好、4 個以上練習 0 分錯但實戰沒變好（背答案）。
"""

import numpy as np

X_MIN, X_MAX = 8.0, 22.0
Y_MIN, Y_MAX = 0.0, 110.0
MAX_NEURONS = 12
MANUAL_NEURONS = 6  # 超過這個數量的神經元只能交給機器學，不顯示滑桿
DATA_SEED = 5
STEPS = 4000


def true_f(x):
    lunch = 55 * np.exp(-(((x - 12.2) / 1.3) ** 2))
    dinner = 70 * np.exp(-(((x - 18.8) / 1.6) ** 2))
    return 15 + lunch + dinner


def make_data(seed=DATA_SEED, n_train=10, n_test=40, noise=12):
    rng = np.random.default_rng(seed)
    x = np.sort(np.linspace(8.4, 21.6, n_train) + rng.uniform(-0.4, 0.4, n_train))
    y = np.clip(true_f(x) + rng.normal(0, noise, n_train), 0, None)
    xt = np.linspace(8.2, 21.8, n_test)
    yt = np.clip(true_f(xt) + rng.normal(0, 3, n_test), 0, None)
    return x, y, xt, yt


X, Y, XT, YT = make_data()


def default_bends():
    """前三個折點分散在全天；之後的放在練習點之間，讓多神經元時背得起來。"""
    mids = list((X[1:] + X[:-1]) / 2)
    first = [11.0, 15.0, 19.0]
    out = first + [m for m in mids if all(abs(m - f) > 0.5 for f in first)]
    while len(out) < MAX_NEURONS:
        out.append(mids[len(out) % len(mids)])
    return [round(float(v), 2) for v in out[:MAX_NEURONS]]


def default_params():
    """一開始只有一條直線，所有神經元的力度都是 0（加進來不改變任何東西）。"""
    return {"b": 30.0, "a": 2.0, "c": default_bends(), "s": [0.0] * MAX_NEURONS}


def predict(x, p, n):
    x = np.asarray(x, dtype=float)
    y = p["b"] + p["a"] * (x - X_MIN)
    for c, s in zip(p["c"][:n], p["s"][:n]):
        y = y + s * np.maximum(x - c, 0)
    return y


def rmse(p, n, test=False):
    x, y = (XT, YT) if test else (X, Y)
    return float(np.sqrt(np.mean((predict(x, p, n) - y) ** 2)))


def train(p, n, steps=STEPS, n_frames=60):
    """從學員目前的參數出發做梯度下降（Adam），回傳動畫用的參數快照。"""
    b, a = float(p["b"]), float(p["a"])
    c = np.array(p["c"][:n], dtype=float)
    s = np.array(p["s"][:n], dtype=float)
    lr = {"b": 1.0, "a": 0.3, "c": 0.05, "s": 0.4}
    state = {k: [0.0, 0.0] for k in ("b", "a")}
    mc, vc, ms, vs = (np.zeros(n) for _ in range(4))
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    xs = X - X_MIN

    def adam(g, m, v, t):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh, vh = m / (1 - beta1**t), v / (1 - beta2**t)
        return mh / (np.sqrt(vh) + eps), m, v

    def snapshot():
        full_c = list(c) + list(p["c"][n:])
        full_s = list(s) + list(p["s"][n:])
        q = {"b": b, "a": a, "c": [float(v) for v in full_c], "s": [float(v) for v in full_s]}
        return {"params": q, "train": rmse(q, n), "test": rmse(q, n, test=True)}

    # 前段變化最劇烈，快照取樣用平方分布讓動畫前段細、後段疏
    keep = sorted({int(round((i / (n_frames - 1)) ** 2 * steps)) for i in range(n_frames)})
    frames = [snapshot()] if 0 in keep else []
    for t in range(1, steps + 1):
        hinge = np.maximum(X[:, None] - c[None, :], 0)  # (N, n)
        r = b + a * xs + hinge @ s - Y
        gb = 2 * r.mean()
        ga = 2 * (r * xs).mean()
        gs = 2 * (r[:, None] * hinge).mean(0)
        gc = -2 * (r[:, None] * (hinge > 0) * s[None, :]).mean(0)

        step, *state["b"] = adam(gb, *state["b"], t); b -= lr["b"] * step
        step, *state["a"] = adam(ga, *state["a"], t); a -= lr["a"] * step
        step, ms, vs = adam(gs, ms, vs, t); s -= lr["s"] * step
        step, mc, vc = adam(gc, mc, vc, t); c -= lr["c"] * step
        c = np.clip(c, X_MIN, X_MAX)
        if t in keep:
            frames.append(snapshot())
    return frames


def learn(p, n):
    """「讓機器自己學」：從學員當下的狀態學，但避免卡在局部最小值。

    1. 折點跑到所有練習點右邊的神經元梯度永遠是 0（死掉了），先重設
    2. 同時從預設起點再學一次，取練習誤差較低的那次
    回傳 (frames, restarted)；restarted 表示採用了預設起點那次。
    """
    q = {"b": p["b"], "a": p["a"], "c": list(p["c"]), "s": list(p["s"])}
    dflt = default_params()
    for i in range(n):
        if q["c"][i] > X.max():
            q["c"][i], q["s"][i] = dflt["c"][i], 0.0
    mine, fresh = train(q, n), train(dflt, n)
    if fresh[-1]["train"] < mine[-1]["train"] - 0.5:
        start = {"params": p, "train": rmse(p, n), "test": rmse(p, n, test=True)}
        return [start] + fresh, True
    return mine, False


def sweep():
    """講師用：0 到 MAX_NEURONS 個神經元，各從預設起點訓練一次，回傳整張成績單。"""
    return [{"n": n, **train(default_params(), n, n_frames=2)[-1]} for n in range(MAX_NEURONS + 1)]
