import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class AnomalyDetector:
    def __init__(self, window_size, seasonal_period=None, alpha=0.05) -> None:
        self.history = []
        self.window_size = window_size
        self.seasonal_period = seasonal_period
        self.alpha = alpha

    def update(self, new_point):
        self.history.append(new_point)

        if len(self.history) < self.window_size:
            return False

        if self.seasonal_period is None:
            self.seasonal_period = self.get_seasonal_period()

        window_data = pd.Series(self.history[-self.window_size:])
        model = ExponentialSmoothing(
            window_data, trend='add', seasonal='add', seasonal_periods=self.seasonal_period
            ).fit()

        forecast = model.forecast(steps=1).iloc[0]

        residuals = window_data - model.fittedvalues
        residual_std = residuals.std()

        z_score = norm.ppf(1 - self.alpha / 2)
        lower_bound = forecast - z_score * residual_std
        upper_bound = forecast + z_score * residual_std

        is_anomaly = new_point < lower_bound or new_point > upper_bound
        return is_anomaly

    def get_seasonal_period(self) -> float:
        history_fft = abs(fft(self.history) / len(self.history))
        frequency = fftfreq(history_fft.size)
        return int(1 / frequency[history_fft == max(history_fft)][0])
