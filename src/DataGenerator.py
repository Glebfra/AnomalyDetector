import random
from time import sleep
from typing import Generator

import numpy as np


class DataGenerator:
    time: int = 0

    def __init__(
            self,
            trend_slope: float = 0.01,
            season_amplitude: float = 10,
            season_period: float = 60,
            random_noise_level: float = 1,
            random_seed: float = None,
    ) -> None:
        self.trend_slope = trend_slope

        self.season_amplitude = season_amplitude
        self.season_period = season_period

        self.random_noise_level = random_noise_level
        if random_seed is not None:
            random.seed(random_seed)

    def generate_data(self) -> Generator:
        while True:
            yield self.trend + self.seasonality + self.random_noise_point
            self.time += 1

    @property
    def trend(self) -> float:
        return self.trend_slope * self.time

    @property
    def seasonality(self) -> float:
        return self.season_amplitude * np.sin(2 * np.pi * self.time / self.season_period)

    @property
    def random_noise_point(self) -> float:
        """Creates the random noise point between -noise_level and noise_level"""
        return ((2 * random.random()) - 1) * self.random_noise_level
