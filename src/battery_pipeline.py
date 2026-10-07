"""锂电池 SOH 时序预测流水线（OOP）。

在 battery_processing 的无依赖函数之上，把脚本式流程封装成三个可复用的类：
    DataLoader     读取 .mat、容量->SOH、时序划分、归一化/反归一化
    FeatureBuilder 构造 LSTM 需要的滑动窗口三维样本
    ModelRunner    LSTM 的搭建、训练、预测、评估与持久化
"""

import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from .battery_processing import reconstruct_capacity, build_windows


class DataLoader:
    """读取 NASA 电池数据，生成 SOH 序列，并完成划分与归一化。"""

    def __init__(self, mat_path, battery_name="B0005", nominal_capacity=2.0,
                 train_ratio=0.7, val_ratio=0.15, scale_bounds=None):
        self.mat_path = mat_path
        self.battery_name = battery_name
        self.nominal_capacity = nominal_capacity
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        # None：在训练段拟合 MinMaxScaler；给定 (lo, hi)：按固定物理量程缩放。
        # SOH 预测建议用固定量程（如 50~105），否则测试段更低的 SOH 会落在训练
        # 量程之外，导致网络输出饱和、无法外推（固定量程来自先验工程边界，非泄漏）。
        self.scale_bounds = scale_bounds
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self._bounds = None

    def load_capacities(self):
        """遍历放电循环，用电流对时间的积分得到每次循环的容量(Ah)。"""
        mat = scipy.io.loadmat(self.mat_path)
        cycles = mat[self.battery_name][0][0]["cycle"][0]

        capacities = []
        for cyc in cycles:
            if cyc["type"][0] != "discharge":
                continue
            d = cyc["data"][0][0]
            capacities.append(
                reconstruct_capacity(d["Time"][0], d["Current_measured"][0])
            )
        return np.asarray(capacities, dtype=float)

    def capacity_to_soh(self, capacities):
        """SOH(%) = 当前最大可用容量 / 额定容量 * 100。"""
        return np.asarray(capacities, dtype=float) / self.nominal_capacity * 100.0

    def split_points(self, n):
        """返回 train/val、val/test 两个时间分界下标（只按时间顺序，不打乱）。"""
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        return train_end, val_end

    def fit_scaler(self, train_series):
        """确定缩放基准：固定量程直接采用；否则归一化器只在训练段拟合。"""
        if self.scale_bounds is not None:
            self._bounds = self.scale_bounds
        else:
            self.scaler.fit(np.asarray(train_series).reshape(-1, 1))
            self._bounds = None
        return self

    def scale(self, series):
        arr = np.asarray(series, dtype=float)
        if self._bounds is not None:
            lo, hi = self._bounds
            return (arr - lo) / (hi - lo)
        return self.scaler.transform(arr.reshape(-1, 1)).ravel()

    def inverse_scale(self, series):
        arr = np.asarray(series, dtype=float)
        if self._bounds is not None:
            lo, hi = self._bounds
            return arr * (hi - lo) + lo
        return self.scaler.inverse_transform(arr.reshape(-1, 1)).ravel()


class FeatureBuilder:
    """把一维退化序列切成“前 time_step 个循环预测下一个循环”的滑动窗口样本。"""

    def __init__(self, time_step=5):
        self.time_step = time_step

    def build(self, series_scaled, train_end, val_end):
        """在整条序列上开窗，再按每个样本“目标点”的位置归入对应集合。

        验证/测试段开头的窗口会引用前一段末尾的历史点，这是预测时本就已知的
        历史信息，不属于未来泄漏；窗口中任何点都严格早于其目标点。
        """
        x_2d, y = build_windows(series_scaled, self.time_step)
        target_index = np.arange(self.time_step, self.time_step + y.size)
        x = x_2d[..., np.newaxis]  # [样本数, 时间步, 特征数]

        masks = {
            "train": target_index < train_end,
            "val": (target_index >= train_end) & (target_index < val_end),
            "test": target_index >= val_end,
        }
        return {name: (x[m], y[m], target_index[m]) for name, m in masks.items()}


class ModelRunner:
    """封装 LSTM 的搭建、训练、预测、评估与读写。"""

    def __init__(self, time_step=5, n_features=1, lstm_units=50, dense_units=25):
        self.time_step = time_step
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.model = None

    def build(self):
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=False,
                 input_shape=(self.time_step, self.n_features)),
            Dense(self.dense_units, activation="relu"),
            Dense(1),
        ])
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return self

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=8, patience=15, verbose=1):
        """验证集用于 EarlyStopping，并恢复最优轮次权重，抑制过拟合。"""
        callbacks, validation_data = [], None
        if X_val is not None:
            validation_data = (X_val, y_val)
            callbacks.append(EarlyStopping(monitor="val_loss", patience=patience,
                                           restore_best_weights=True))
        return self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=verbose,
        )

    def predict(self, X):
        return self.model.predict(X, verbose=0).ravel()

    @staticmethod
    def evaluate(y_true, y_pred):
        """输入真实物理量（SOH%），返回 MAE、RMSE、MAPE、R2 四项指标。"""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)
        return self
