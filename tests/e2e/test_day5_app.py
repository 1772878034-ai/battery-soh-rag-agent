"""Playwright E2E：测试完整 day5_web_app.py（RAG 问答 + LSTM 预测）。

运行（battery 环境）：
    python -m pytest tests/e2e/test_day5_app.py -v
产物：tests/e2e/screenshots、tests/e2e/logs（失败额外 __FAILED.png）。
依赖：data/hf_models（Qwen/reranker）、data/model（MiniLM）、data/faiss_db、
      data/B0005.mat、data/lstm_battery_model.h5。
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DAY5_E2E") != "1",
    reason="day5 E2E 依赖本地未入库的模型/数据，设置 RUN_DAY5_E2E=1 后运行",
)

ROOT = Path(__file__).resolve().parents[2]
E2E_DIR = Path(__file__).resolve().parent
SHOT_DIR = E2E_DIR / "screenshots"
LOG_DIR = E2E_DIR / "logs"
APP_PATH = ROOT / "src" / "day5_web_app.py"

TAB_RAG = "🔍 RAG智能问答"
TAB_LSTM = "📈 LSTM电池SOH预测"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def day5_server():
    port = _free_port()
    base_url = f"http://localhost:{port}"

    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_CACHE"] = str(ROOT / "data" / "model")  # MiniLM 缓存
    env["HF_HUB_OFFLINE"] = "1"  # 模型均已在本地，强制离线避免在线校验挂起

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP_PATH),
        "--server.headless", "true",
        "--server.address", "localhost",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    for _ in range(60):
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"day5 启动失败：\n{output[-4000:]}")
        try:
            with urllib.request.urlopen(base_url, timeout=2) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(1)

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True)
        else:
            proc.kill()
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = proc.stdout.read() if proc.stdout else ""
        (LOG_DIR / "day5_server.log").write_text(out, encoding="utf-8")
    except Exception:
        pass


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture()
def page(request, day5_server):
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg, context, records = None, None, None

        # day5 首次要加载 Qwen + build_retriever（含 reranker），必须等首屏
        # 脚本完整跑完、“回答”按钮出现才算就绪，否则提前关页会中断加载、缓存建不起来
        for wait_s in (420, 150, 150):
            context = browser.new_context(viewport={"width": 1400, "height": 1000})
            candidate = context.new_page()
            logs = []
            candidate.on("console", lambda m: logs.append(f"[console.{m.type}] {m.text}"))
            candidate.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            candidate.goto(day5_server, wait_until="domcontentloaded")
            try:
                candidate.get_by_role("button", name="回答").wait_for(
                    state="visible", timeout=wait_s * 1000)
                pg, records = candidate, logs
                break
            except Exception:
                context.close()

        if pg is None:
            browser.close()
            raise RuntimeError("day5 多次会话后仍未就绪（见 logs/day5_server.log）")

        yield pg

        failed = bool(getattr(request.node, "rep_call", None)
                      and request.node.rep_call.failed)
        name = request.node.name
        if failed:
            records.append("---- 失败时页面文本 ----")
            records.append(pg.inner_text("body"))
            pg.screenshot(path=str(SHOT_DIR / f"{name}__FAILED.png"), full_page=True)
        pg.screenshot(path=str(SHOT_DIR / f"{name}.png"), full_page=True)
        (LOG_DIR / f"{name}.log").write_text("\n".join(records), encoding="utf-8")

        context.close()
        browser.close()


def test_day5_tabs_present(page):
    expect_title = page.get_by_text("锂电池SOH预测 | RAG智能问答 Demo").first
    expect_title.wait_for(state="visible", timeout=30000)
    page.get_by_role("tab", name=TAB_RAG).wait_for(state="visible")
    page.get_by_role("tab", name=TAB_LSTM).wait_for(state="visible")


def test_day5_rag_answers(page):
    # 默认首标签即 RAG，输入框已有默认问题，直接点“回答”
    page.get_by_role("button", name="回答").click()
    page.get_by_role("heading", name="回答").wait_for(
        state="visible", timeout=240000)
    page.get_by_text("检索到的参考文档片段").first.wait_for(
        state="visible", timeout=30000)


def test_day5_lstm_predicts(page):
    page.get_by_role("tab", name=TAB_LSTM).click()
    page.get_by_role("button", name="加载数据 & 开始预测").click()
    page.get_by_text("预测完成").first.wait_for(
        state="visible", timeout=180000)
