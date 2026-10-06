"""Playwright E2E：自动启动本地 Streamlit 并测试，输出截图与失败日志。

运行（在 battery 环境）：
    python -m pytest tests/e2e -s
产物：
    tests/e2e/screenshots/*.png   每个用例一张，失败额外生成 __FAILED.png
    tests/e2e/logs/*.log          浏览器控制台 / 页面错误 / 失败堆栈
可用环境变量 STREAMLIT_APP 指定被测页面（默认轻量冒烟页）。
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect

ROOT = Path(__file__).resolve().parents[2]
SHOT_DIR = Path(__file__).resolve().parent / "screenshots"
LOG_DIR = Path(__file__).resolve().parent / "logs"
DEFAULT_APP = Path(__file__).resolve().parent / "smoke_app.py"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def streamlit_server():
    app_path = Path(os.environ.get("STREAMLIT_APP", str(DEFAULT_APP)))
    port = int(os.environ.get("STREAMLIT_PORT", _free_port()))
    base_url = f"http://localhost:{port}"

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.headless", "true",
        "--server.address", "localhost",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    ready = False
    for _ in range(90):
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"Streamlit 启动失败：\n{output}")
        try:
            with urllib.request.urlopen(base_url, timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(1)
    if not ready:
        proc.kill()
        raise RuntimeError("等待 Streamlit 就绪超时")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True)
        else:
            proc.kill()


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture()
def page(request, streamlit_server):
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    ready_tab = os.environ.get("STREAMLIT_READY_TAB", "容量重构")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg, context, records = None, None, None

        # 首个会话可能卡在 "Running..."，换新页面（新会话）重试直到就绪
        for _ in range(6):
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            candidate = context.new_page()
            logs = []
            candidate.on("console", lambda m: logs.append(f"[console.{m.type}] {m.text}"))
            candidate.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
            candidate.goto(streamlit_server, wait_until="domcontentloaded")
            try:
                candidate.get_by_role("tab", name=ready_tab).wait_for(
                    state="visible", timeout=25000)
                pg, records = candidate, logs
                break
            except Exception:
                context.close()

        if pg is None:
            browser.close()
            raise RuntimeError("多次新建会话后页面仍未就绪")

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


def test_home_renders(page):
    expect(page.get_by_text("电池数据处理 · 冒烟测试页").first).to_be_visible()
    expect(page.get_by_role("tab", name="容量重构")).to_be_visible()
    expect(page.get_by_role("tab", name="窗口样本")).to_be_visible()
    expect(page.get_by_text("冒烟测试环境").first).to_be_visible()


def test_capacity_reconstruction(page):
    page.get_by_role("button", name="计算容量").click()
    expect(page.get_by_text("2.00").first).to_be_visible(timeout=30000)
    expect(page.get_by_text("容量 = 2.00 Ah").first).to_be_visible()


def test_window_samples(page):
    page.get_by_role("tab", name="窗口样本").click()
    page.get_by_role("button", name="生成窗口").click()
    expect(page.get_by_text("样本数：7").first).to_be_visible(timeout=30000)
