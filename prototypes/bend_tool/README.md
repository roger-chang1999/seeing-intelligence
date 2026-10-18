# 彎折工具原型（第 5–7 步）

同一個工具的兩種做法，用來決定整門課互動工具的技術路線。

| 版本 | 檔案 | 埠 | 做法 |
|---|---|---|---|
| A | `app_gradio_pure.py` | 7860 | 純 Gradio：每次動滑桿，伺服器用 matplotlib 重畫 |
| B | `app_gradio_js.py` + `bend_widget.js` | 7861 | Gradio 外殼；畫面在瀏覽器端用 JS 即時重畫，訓練仍由伺服器的 Python（`core.train`）執行 |

共用核心在 `core.py`：資料（店家各時段來客數）、模型（直線 + ReLU 彎折）、梯度下降。

## 在本機執行（Windows / macOS）
需要 Python 3.10 以上。解壓縮後，在這個資料夾裡打開終端機：

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux：source .venv/bin/activate
pip install -r requirements.txt

python app_gradio_pure.py        # 版本 A → 自動打開 http://127.0.0.1:7860
python app_gradio_js.py          # 版本 B → 自動打開 http://127.0.0.1:7861
```
兩個版本可以開兩個終端機同時跑，方便並排比較。按 `Ctrl+C` 結束。

## 試玩重點
1. 拖動「基本人數」滑桿：線有沒有跟著手動？
2. 按「＋ 加一個」，加 3 個神經元，自己試著把線拼到點上
3. 按「讓機器自己學」，看它幾秒內拼好
4. 版本 B 才有：直接拖橘色彎折點；滑鼠移到灰點上看實際與預測人數
