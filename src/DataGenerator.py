import random
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
    ) -> None:
        """
        This class is used to generate data for anomaly detector based on trend slope, season and random points
        :param trend_slope: The slope of the trend line
        :param season_amplitude: The amplitude of season
        :param season_period: The period of season
        :param random_noise_level: The random noise level
        """
        self.trend_slope = trend_slope

        self.season_amplitude = season_amplitude
        self.season_period = season_period

        self.random_noise_level = random_noise_level

    def generate_data(self) -> Generator:
        """Emulates the real time data"""
        while True:
            yield self.trend + self.seasonality + self.random_noise_point
            self.time += 1

    @property
    def trend(self) -> float:
        """Creates the trend points based on the trend slope and time"""
        return self.trend_slope * self.time

    @property
    def seasonality(self) -> float:
        """Creates the seasonality points between [-season_amplitude; season_amplitude]"""
        return self.season_amplitude * np.sin(2 * np.pi * self.time / self.season_period)

    @property
    def random_noise_point(self) -> float:
        """Creates the random noise point between [-noise_level; noise_level]"""
        return ((2 * random.random()) - 1) * self.random_noise_level
