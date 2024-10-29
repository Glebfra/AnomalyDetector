import numpy as np


class DataGenerator:
    def __init__(
            self,
            points_number: int = 1000,
            trend_slope: float = 0.01,
            season_amplitude: float = 10,
            season_period: float = 365,
            periodic_amplitude: float = 5,
            periodic_frequency: float = 1,
            random_noise_level: float = 1,
            random_seed: float = None
    ) -> None:
        self.points_number = points_number
        self.trend_slope = trend_slope

        self.season_amplitude = season_amplitude
        self.season_period = season_period

        self.periodic_amplitude = periodic_amplitude
        self.periodic_frequency = periodic_frequency

        self.random_noise_level = random_noise_level
        if random_seed is not None:
            np.random.seed(random_seed)

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.points_number)

    @property
    def trend(self) -> np.ndarray:
        return self.trend_slope * self.time

    @property
    def seasonality(self) -> np.ndarray:
        return self.season_amplitude * np.sin(2 * np.pi * self.time / self.season_period)

    @property
    def periodic_pattern(self) -> np.ndarray:
        return self.periodic_amplitude * np.sin(2 * np.pi * self.periodic_frequency * self.time)

    @property
    def random_noise(self) -> np.ndarray:
        return np.random.normal(0, self.random_noise_level, self.points_number)

    @property
    def data(self) -> np.ndarray:
        return self.trend + self.seasonality + self.periodic_pattern + self.random_noise
