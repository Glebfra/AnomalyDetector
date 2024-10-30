import matplotlib.pyplot as plt
import numpy as np

from AnomalyDetector import AnomalyDetector
from DataGenerator import DataGenerator

generator = DataGenerator(
    trend_slope=0.001, random_seed=None, season_amplitude=1, season_period=60
)
detector = AnomalyDetector(window_size=120, seasonal_period=60)

plt.ion()  # Включаем интерактивный режим
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-', label="Data")  # линия для данных
anomaly_line, = ax.plot([], [], 'ro', label="Anomaly")  # точки для аномалий

ax.set_xlim(0, 50)  # первоначальные границы по оси x
ax.set_ylim(-15, 15)  # первоначальные границы по оси y
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

    fig.canvas.draw()
    fig.canvas.flush_events()

    if generator.time > 10:
        break

    plt.pause(0.1)

plt.ioff()
plt.show()
