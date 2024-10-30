import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft, fftfreq
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class AnomalyDetector:
    def __init__(self, window_size, seasonal_period, alpha=0.5) -> None:
        self.history = []
        self.window_size = window_size
        self.seasonal_period = seasonal_period
        self.alpha = alpha
        # self.trend_moving_average_size = len(self.data) // 4
        # self.seasonal_moving_average_size = self.trend_moving_average_size // 10

    def update(self, new_point):
        self.history.append(new_point)

        if len(self.history) < self.window_size:
            return False

        window_data = pd.Series(self.history[-self.window_size:])
        model = ExponentialSmoothing(window_data, trend='add', seasonal='add', seasonal_periods=self.seasonal_period).fit()

        forecast = model.forecast(steps=1)
        print(forecast)

        residuals = window_data - model.fittedvalues
        residual_std = residuals.std()

        z_score = norm.ppf(1 - self.alpha / 2)
        lower_bound = forecast - z_score * residual_std
        upper_bound = forecast + z_score * residual_std

        is_anomaly = new_point < lower_bound or new_point > upper_bound

        return is_anomaly

    # def remove_seasonality(self, period=None, model='additive') -> np.ndarray:
    #     period = self.seasonal_period if period is None else period
    #     decomposition = seasonal_decompose(self.data, model=model, period=period)
    #     return decomposition.trend + decomposition.resid

    # @staticmethod
    # def moving_average(data, window_size) -> np.ndarray:
    #     data_length = len(data)
    #     moving_average = np.array(
    #         [np.mean(data[i - window_size:i]) for i in range(window_size, data_length)]
    #     )
    #     return moving_average

    # @property
    # def trend(self) -> np.ndarray:
    #     return self.moving_average(self.data, self.trend_moving_average_size)
    #
    # @property
    # def seasonal(self) -> np.ndarray:
    #     data = self.data[self.trend_moving_average_size - 1:-1] - self.trend
    #     return self.moving_average(data, self.seasonal_moving_average_size)

    # @property
    # def seasonal_period(self) -> float:
    #     seasonal = self.seasonal
    #     seasonal_fft = abs(fft(seasonal) / seasonal.size)
    #     frequency = fftfreq(seasonal_fft.size)
    #     return int(1 / frequency[seasonal_fft == max(seasonal_fft)][0])

    # @property
    # def random_value(self) -> np.ndarray:
    #     seasonal = self.seasonal
    #     time = np.arange(self.trend_moving_average_size + self.seasonal_moving_average_size, len(self.data))
    #     return self.seasonal - np.sin(2 * np.pi * time / self.seasonal_period)
