"""轻量 Streamlit 冒烟页：不加载 LLM，只验证电池数据处理与页面渲染。

供 Playwright E2E 使用。运行：
    streamlit run tests/e2e/smoke_app.py --server.headless true
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from src.battery_processing import build_windows, reconstruct_capacity

st.set_page_config(page_title="电池数据处理冒烟测试", layout="centered")
st.title("电池数据处理 · 冒烟测试页")
st.caption("不加载 LLM，仅用于 Playwright 自动化验证核心数据处理与渲染")

with st.sidebar:
    st.header("导航")
    st.write("冒烟测试环境")

tab1, tab2 = st.tabs(["容量重构", "窗口样本"])

with tab1:
    st.subheader("电流积分容量重构")
    if st.button("计算容量", key="cap_btn"):
        t = np.array([0, 900, 1800, 2700, 3600])
        current = np.full_like(t, -2.0)
        cap = reconstruct_capacity(t, current)
        st.metric("放电容量(Ah)", f"{cap:.2f}")
        st.success(f"恒流 2A 放电 1 小时，容量 = {cap:.2f} Ah")

with tab2:
    st.subheader("滑动窗口样本数")
    n = st.slider("序列长度", 6, 30, 12)
    step = st.number_input("窗口长度 time_step", min_value=1, max_value=10, value=5, step=1)
    if st.button("生成窗口", key="win_btn"):
        series = np.arange(n, dtype=float)
        x, _ = build_windows(series, int(step))
        st.write(f"样本数：{x.shape[0]}（应为 {n - int(step)}）")
        st.dataframe(x)
