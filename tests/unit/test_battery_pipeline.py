"""流水线三个类的单元测试：SOH 换算、时序划分、固定量程、窗口归属与模型/指标。"""

import numpy as np
import pytest

from src.battery_pipeline import DataLoader, FeatureBuilder, ModelRunner


# ---------- DataLoader：SOH 与划分 ----------

def test_capacity_to_soh():
    loader = DataLoader("unused.mat", nominal_capacity=2.0)
    soh = loader.capacity_to_soh([2.0, 1.8, 1.0])
    assert soh == pytest.approx([100.0, 90.0, 50.0])


def test_split_points_chronological():
    loader = DataLoader("unused.mat", train_ratio=0.7, val_ratio=0.15)
    train_end, val_end = loader.split_points(100)
    assert (train_end, val_end) == (70, 85)
    assert train_end < val_end < 100


def test_fixed_bounds_scale_roundtrip():
    loader = DataLoader("unused.mat", scale_bounds=(50.0, 105.0))
    loader.fit_scaler(np.array([60.0, 100.0]))
    scaled = loader.scale([50.0, 77.5, 105.0])
    assert scaled == pytest.approx([0.0, 0.5, 1.0])
    assert loader.inverse_scale(scaled) == pytest.approx([50.0, 77.5, 105.0])


def test_train_fit_scaler_roundtrip():
    loader = DataLoader("unused.mat")  # scale_bounds=None
    train = np.linspace(60.0, 100.0, 41)
    loader.fit_scaler(train)
    scaled = loader.scale(train)
    assert scaled[0] == pytest.approx(0.0)
    assert scaled[-1] == pytest.approx(1.0)
    assert loader.inverse_scale(scaled) == pytest.approx(train)


# ---------- FeatureBuilder：窗口形状与归属 ----------

def test_feature_builder_shapes_and_assignment():
    series = np.linspace(0.0, 1.0, 100)
    sets = FeatureBuilder(time_step=5).build(series, train_end=70, val_end=85)

    X_train, y_train, i_train = sets["train"]
    X_val, y_val, i_val = sets["val"]
    X_test, y_test, i_test = sets["test"]

    assert X_train.shape == (65, 5, 1)      # 目标点 5..69
    assert X_val.shape == (15, 5, 1)        # 目标点 70..84
    assert X_test.shape == (15, 5, 1)       # 目标点 85..99
    assert i_train[0] == 5 and i_train[-1] == 69
    assert i_val[0] == 70 and i_test[-1] == 99
    # 窗口最后一个输入点严格等于目标点的前一个值
    assert X_val[0, -1, 0] == pytest.approx(series[69])
    assert y_val[0] == pytest.approx(series[70])


# ---------- ModelRunner：搭建与指标 ----------

def test_model_runner_build():
    runner = ModelRunner(time_step=5).build()
    assert runner.model is not None
    assert runner.model.count_params() > 0


def test_evaluate_perfect_and_offset():
    y = np.array([80.0, 90.0, 100.0])

    perfect = ModelRunner.evaluate(y, y)
    assert perfect["MAE"] == pytest.approx(0.0)
    assert perfect["RMSE"] == pytest.approx(0.0)
    assert perfect["MAPE"] == pytest.approx(0.0)
    assert perfect["R2"] == pytest.approx(1.0)

    offset = ModelRunner.evaluate(y, y - 1.0)
    assert offset["MAE"] == pytest.approx(1.0)
    assert offset["RMSE"] == pytest.approx(1.0)
    assert offset["MAPE"] == pytest.approx(np.mean(1.0 / y) * 100.0)


# ---------- 真实数据集成（无文件时跳过） ----------

@pytest.mark.integration
def test_pipeline_b0005_integration():
    from pathlib import Path

    mat_path = Path(__file__).resolve().parents[2] / "data" / "B0005.mat"
    if not mat_path.exists():
        pytest.skip(f"未找到数据文件 {mat_path}")

    loader = DataLoader(str(mat_path), nominal_capacity=2.0,
                        scale_bounds=(50.0, 105.0))
    soh = loader.capacity_to_soh(loader.load_capacities())
    assert 100 < soh.size <= 200
    assert np.all(np.isfinite(soh))
    assert soh[-1] < soh[0]              # SOH 整体衰减
