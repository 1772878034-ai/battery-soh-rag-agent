# 测试与 CI 使用说明

本项目测试运行在 **battery conda 环境（Python 3.10）**：`D:\software\anaconda3\envs\battery`。
下文命令均在项目根目录 `D:\02_Workproject\battery_project` 下执行。

## 1. 一次性安装

```powershell
# 用 battery 环境的 python（示例用完整路径，激活环境后可直接用 python）
D:\software\anaconda3\envs\battery\python.exe -m pip install -r requirements-dev.txt

# 安装 Playwright 浏览器（国内网络建议走镜像，否则 cdn 可能卡住）
$env:PLAYWRIGHT_DOWNLOAD_HOST="https://cdn.npmmirror.com/binaries/playwright"
D:\software\anaconda3\envs\battery\python.exe -m playwright install chromium
```

`requirements-dev.txt` 只含轻量测试工具链（numpy/scipy/streamlit/pytest/pytest-html/playwright），不含 tensorflow、torch。

## 2. 运行单元测试（Pytest）

```powershell
# 默认只跑 tests/unit（pytest.ini 中 testpaths 已配置）
D:\software\anaconda3\envs\battery\python.exe -m pytest

# 生成 HTML + JUnit 报告
D:\software\anaconda3\envs\battery\python.exe -m pytest tests/unit `
  --html=reports/report.html --self-contained-html --junitxml=reports/junit.xml
```

共 10 个测试函数（参数化后展开为 20 个用例），覆盖：
- **缺失值**：内部/端点线性插值、均值/置零、全缺报错、非法方式；
- **容量重构**：恒流/非均匀采样积分、长度不一致、点数不足、时间倒序、充电电流；
- **窗口长度**：样本数与形状、N=step+1/step/step−1 边界、非法 step、含 NaN；
- **异常输入**：None、字符串、二维、空序列；另有真实 B0005 数据集成用例（无文件自动 skip）。

## 3. 运行 Playwright（Streamlit E2E）

```powershell
D:\software\anaconda3\envs\battery\python.exe -m pytest tests/e2e -v
```

- 测试会**自动启动并关闭** Streamlit（无需手动 `streamlit run`），默认被测页面是轻量冒烟页 `tests/e2e/smoke_app.py`（不加载 LLM，保证快速稳定）。
- 产物：
  - 截图：`tests/e2e/screenshots/<用例>.png`，失败额外生成 `__FAILED.png`；
  - 日志：`tests/e2e/logs/<用例>.log`（记录浏览器 console、pageerror；失败时附带页面正文）。
- 说明：首个浏览器会话可能卡在 Streamlit 的 "Running..."（首次连接竞态），fixture 会自动新建会话重试直到就绪。

**改测完整页面 day5（可选，较重）**：day5 启动会加载 Qwen2.5-1.5B、bge-reranker 等模型，需提前备好模型与依赖。

```powershell
$env:STREAMLIT_APP="src/day5_web_app.py"
$env:STREAMLIT_READY_TAB="🔍 RAG智能问答"
D:\software\anaconda3\envs\battery\python.exe -m pytest tests/e2e -v
```

## 4. GitHub Actions

工作流文件：`.github/workflows/test.yml`，两个 Job：
1. **unit-tests**：安装依赖（requirements-dev.txt）→ 运行 Pytest 生成 HTML/JUnit → 发布结果摘要 → 上传 `unit-test-report` 工件；
2. **e2e-streamlit**：安装依赖与 Chromium → 运行 Playwright → 上传 `e2e-screenshots-logs` 工件（截图/日志）。

触发时机：push 到 main、Pull Request、手动 workflow_dispatch。

**前置：外层目录当前还不是 git 仓库**，需先初始化并推送到 GitHub（可用 GitHub Desktop，或本机 git）：

```powershell
git init
git add .
git commit -m "chore: add tests and ci"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

推送后在仓库 **Actions** 页可看到运行记录，运行结束在对应 run 页面下载 Artifacts（报告 / 截图日志）。

## 5. 目录结构（新增部分）

```
battery_project/
├─ src/
│  ├─ battery_processing.py        # 纯函数：容量重构/缺失值/窗口/归一化
│  └─ ...
├─ tests/
│  ├─ unit/test_battery_processing.py
│  └─ e2e/
│     ├─ smoke_app.py              # 轻量 Streamlit 冒烟页
│     ├─ test_streamlit_app.py     # Playwright E2E
│     ├─ screenshots/
│     └─ logs/
├─ docs/
│  ├─ 软件测试复习.md
│  ├─ 车机软件模块概览.md
│  └─ 测试与CI使用说明.md
├─ .github/workflows/test.yml
├─ requirements-dev.txt
├─ pytest.ini
└─ conftest.py
```
