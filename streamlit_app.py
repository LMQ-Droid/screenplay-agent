# -*- coding: utf-8 -*-
"""Streamlit Community Cloud 入口。

Streamlit Cloud 只认仓库根目录的入口脚本。这个文件负责
把项目根加入 sys.path，让 app.py 里的 `from screenplay import ...`
能找到模块，然后以 __main__ 语义执行它。

本地同样可用：
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for p in (ROOT,):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
