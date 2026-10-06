"""电池数据处理：容量重构、缺失值填补、滑动窗口。

只依赖 numpy，可被 day1/day2/day5 与测试共用。
约定：放电电流为负（NASA 数据），容量单位 Ah。
"""
import numpy as np


def as_float_array(values):
    """把序列转成一维 float ndarray，非法类型 / 形状直接报错。"""
    if values is None:
        raise TypeError("输入不能为 None")
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"无法转换为数值数组：{exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"只接受一维序列，实际为 {arr.ndim} 维")
    if arr.size == 0:
        raise ValueError("序列为空")
    return arr


def reconstruct_capacity(time, current):
    """对放电电流做时间积分求容量（安时 Ah）：sum(-I * dt) / 3600。"""
    t = as_float_array(time)
    i = as_float_array(current)
    if t.size != i.size:
        raise ValueError(f"时间与电流长度不一致：{t.size} vs {i.size}")
    if t.size < 2:
        raise ValueError("至少需要 2 个采样点才能积分")

    dt = np.diff(t)
    if np.any(dt < 0):
        raise ValueError("时间序列必须单调不减")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(i)):
        raise ValueError("时间和电流不能含 NaN / Inf")

    capacity = float(np.sum(-i[1:] * dt) / 3600.0)
    if capacity < 0:
        raise ValueError("积分容量为负，请确认传入的是放电电流（约定为负）")
    return capacity


def fill_missing(values, method="linear"):
    """填补一维序列中的 NaN。

    method: "linear" 线性插值（端点用最近的有效值补齐）、"mean" 均值、"zero" 置 0。
    整段全为 NaN 时无法推断，抛 ValueError。
    """
    arr = as_float_array(values)
    if method not in ("linear", "mean", "zero"):
        raise ValueError(f"未知的填补方式：{method}")

    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    if nan_mask.all():
        raise ValueError("整段序列全为缺失值，无法填补")

    if method == "linear":
        good = np.flatnonzero(~nan_mask)
        filled = arr.copy()
        filled[nan_mask] = np.interp(nan_mask.nonzero()[0], good, arr[good])
        return filled

    fill_value = float(np.nanmean(arr)) if method == "mean" else 0.0
    filled = arr.copy()
    filled[nan_mask] = fill_value
    return filled


def build_windows(series, time_step):
    """用长度 time_step 的滑窗构造监督样本，返回 X(n, time_step)、y(n,)。"""
    if not isinstance(time_step, (int, np.integer)) or isinstance(time_step, bool):
        raise TypeError("time_step 必须是整数")
    if time_step <= 0:
        raise ValueError("time_step 必须为正整数")

    arr = as_float_array(series)
    if np.any(np.isnan(arr)):
        raise ValueError("序列含缺失值，请先调用 fill_missing 填补")
    if arr.shape[0] <= time_step:
        raise ValueError(f"序列长度 {arr.shape[0]} 必须大于窗口长度 {time_step}")

    x, y = [], []
    for i in range(time_step, arr.shape[0]):
        x.append(arr[i - time_step:i])
        y.append(arr[i])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def minmax_scale(values):
    """归一化到 [0, 1]；常量序列统一返回 0。"""
    arr = as_float_array(values)
    if np.any(np.isnan(arr)):
        raise ValueError("归一化前不能含缺失值")
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
