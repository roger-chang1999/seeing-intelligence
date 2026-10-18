"""把 core.py 的資料、trainer.js、bend_widget.js 組成一個單一的 HTML 檔。

    python build.py   →   ../../docs/bend_tool.html

產出的檔案不需要伺服器：雙擊即可開啟，也可以直接放上 GitHub Pages。
"""

import json
from pathlib import Path

import core

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "docs" / "bend_tool.html"


def main():
    data = {
        "x": core.X.tolist(), "y": core.Y.tolist(), "xt": core.XT.tolist(), "yt": core.YT.tolist(),
        "xmin": core.X_MIN, "xmax": core.X_MAX, "ymin": core.Y_MIN, "ymax": core.Y_MAX,
        "maxN": core.MAX_NEURONS, "manualN": core.MANUAL_NEURONS, "params": core.default_params(),
    }
    read = lambda name: (HERE / name).read_text(encoding="utf-8")
    page = (
        read("template.html")
        .replace("/*__DATA__*/", json.dumps(data))
        .replace("/*__TRAINER__*/", read("trainer.js"))
        .replace("/*__WIDGET__*/", read("bend_widget.js"))
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
