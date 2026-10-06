"""电池数据处理单元测试：缺失值 / 容量重构 / 窗口长度 / 异常输入。"""
import numpy as np
import pytest

from src.battery_processing import (
    as_float_array,
    build_windows,
    fill_missing,
    reconstruct_capacity,
)


# ---------- 容量重构 ----------

def test_reconstruct_capacity_values():
    # 恒流 -2A 放电 1 小时 -> 2 Ah
    t = [0, 1800, 3600]
    i = [-2, -2, -2]
    assert reconstruct_capacity(t, i) == pytest.approx(2.0)

    # 半小时 -> 1 Ah
    assert reconstruct_capacity([0, 1800], [-2, -2]) == pytest.approx(1.0)

    # 非均匀采样：-3A，dt 分别为 600、3000 -> 3 Ah
    assert reconstruct_capacity([0, 600, 3600], [-3, -3, -3]) == pytest.approx(3.0)


@pytest.mark.parametrize("t, i", [
    ([0, 1, 2], [-2, -2]),            # 时间、电流长度不一致
    ([0], [-2]),                      # 采样点不足
    ([0, 2, 1], [-2, -2, -2]),        # 时间非单调
    ([0, 1, np.nan], [-2, -2, -2]),   # 含 NaN
    ([0, 1800, 3600], [2, 2, 2]),     # 充电电流（正）导致容量为负
])
def test_reconstruct_invalid_inputs(t, i):
    with pytest.raises(ValueError):
        reconstruct_capacity(t, i)


# ---------- 缺失值 ----------

def test_fill_linear_interior_and_edges():
    out = fill_missing([1, np.nan, 3, np.nan, 5])
    assert out == pytest.approx([1, 2, 3, 4, 5])

    # 端点缺失用最近的有效值补齐
    out = fill_missing([np.nan, 2, 3])
    assert out == pytest.approx([2, 2, 3])


def test_fill_methods_and_failures():
    assert fill_missing([1, np.nan, 3], method="mean") == pytest.approx([1, 2, 3])
    assert fill_missing([1, np.nan, 3], method="zero") == pytest.approx([1, 0, 3])

    with pytest.raises(ValueError):
        fill_missing([1, 2, 3], method="median")      # 不支持的方式
    with pytest.raises(ValueError):
        fill_missing([np.nan, np.nan])                # 整段全缺


# ---------- 窗口长度 ----------

def test_build_windows_count_shape_content():
    series = np.arange(10, dtype=float)
    x, y = build_windows(series, time_step=5)

    assert x.shape == (5, 5)
    assert y.shape == (5,)
    assert x[0] == pytest.approx(series[:5])
    assert y[0] == pytest.approx(series[5])


@pytest.mark.parametrize("length, expect_samples", [
    (6, 1),   # 长度 = 窗口 + 1，恰好 1 个样本（边界）
])
def test_build_windows_boundary(length, expect_samples):
    series = np.arange(length, dtype=float)
    x, y = build_windows(series, time_step=5)
    assert x.shape[0] == expect_samples
    assert y.shape[0] == expect_samples

    # 长度等于 / 小于窗口，无法构造样本
    for too_short in (5, 4):
        with pytest.raises(ValueError):
            build_windows(np.arange(too_short, dtype=float), time_step=5)


@pytest.mark.parametrize("step", [0, -1, 2.5, True])
def test_build_windows_invalid_step(step):
    with pytest.raises((TypeError, ValueError)):
        build_windows(np.arange(10, dtype=float), time_step=step)


def test_build_windows_rejects_nan():
    series = np.array([1, 2, np.nan, 4, 5, 6], dtype=float)
    with pytest.raises(ValueError):
        build_windows(series, time_step=2)


# ---------- 异常输入 ----------

@pytest.mark.parametrize("values", [
    None,                       # None
    ["a", "b", "c"],            # 非数值字符串
    [[1, 2], [3, 4]],           # 二维
    [],                         # 空序列
])
def test_non_numeric_and_shape_inputs(values):
    with pytest.raises((TypeError, ValueError)):
        as_float_array(values)


# ---------- 真实数据集成（无文件时跳过） ----------

@pytest.mark.integration
def test_b0005_discharge_capacities():
    from pathlib import Path
    import scipy.io

    mat_path = Path(__file__).resolve().parents[2] / "data" / "B0005.mat"
    if not mat_path.exists():
        pytest.skip(f"未找到数据文件 {mat_path}")

    cycles = scipy.io.loadmat(mat_path)["B0005"][0][0]["cycle"][0]
    caps = []
    for cyc in cycles:
        if cyc["type"][0] != "discharge":
            continue
        d = cyc["data"][0][0]
        caps.append(reconstruct_capacity(d["Time"][0], d["Current_measured"][0]))
    caps = np.asarray(caps)

    assert 100 < caps.size <= 200          # NASA B0005 约 168 次放电
    assert np.all(np.isfinite(caps))
    assert 1.7 < caps[0] < 2.0             # 额定约 1.86 Ah
    assert caps[-1] < caps[0]              # 容量整体衰减
