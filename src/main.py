import matplotlib.pyplot as plt
import numpy as np

from AnomalyDetector import AnomalyDetector
from DataGenerator import DataGenerator

generator = DataGenerator(
    trend_slope=0.01, season_amplitude=5, season_period=60
)

detector = AnomalyDetector(
    window_size=150, seasonal_period=None, alpha=0.01
)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label="Data")
anomaly_line, = ax.plot([], [], 'ro', label="Anomaly")

ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.title('Data with anomalies')

data_history = []
data_time = []
anomaly_points = []
anomaly_time = []

for data in generator.generate_data():
    is_anomaly = detector.update(data)

    data_history.append(data)
    data_time.append(generator.time)
    line.set_data(data_time, data_history)

    if is_anomaly:
        anomaly_points.append(data)
        anomaly_time.append(generator.time)
        anomaly_line.set_data(anomaly_time, anomaly_points)

    ax.set_xlim(0, len(data_time))
    ax.set_ylim(np.min(data_history), np.max(data_history))

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.05)

plt.ioff()
plt.show()
