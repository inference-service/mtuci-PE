import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from matplotlib.animation import FuncAnimation

import time

matplotlib.use('QtAgg')

import json, sys
from pathlib import Path

DEFAULT_CONFIG = {
    "dataset": {
        "n_samples": 100,
        "n_features": 2,
        "n_redundant": 0,
        "n_informative": 2,
        "n_clusters_per_class": 1,
        "class_sep": 1.0,
        "flip_y": 0.1,
        "random_state": 42
    },
    "classifier": {
        "loss": "perceptron",
        "learning_rate": "constant",
        "eta0": 0.01,
        "max_iter": 1,
        "warm_start": True,
        "tol": None,
        "random_state": 42
    },
    "training": {
        "epochs": 20,
        "window": 5,
        "delta": 1.0,
        "early_stop": False
    },
    "animation": {
        "repeat": True,
        "interval": 700
    }
}

def load_config(path="config.json"):
    """Load configuration from JSON, fallback to defaults on failure."""
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Config file '{path}' not found. Using defaults.")
        return DEFAULT_CONFIG

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            # Basic sanity check
            for section in ("dataset", "classifier", "training"):
                if section not in cfg:
                    raise KeyError(f"Missing section '{section}' in config.")
            return cfg
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[ERROR] Failed to load config: {e}")
        print("[INFO] Falling back to defaults.")
        return DEFAULT_CONFIG
    except Exception as e:
        print(f"[FATAL] Unexpected error while reading config: {e}")
        sys.exit(1)


cfg = load_config("params.json")
print(f"Loaded config: epochs={cfg['training']['epochs']}, eta0={cfg['classifier']['eta0']}")

# 1. Генерация датасета (сделаем задачу не слишком лёгкой)
X, y = make_classification(**cfg["dataset"])

# 2. Инициализация классификатора, похожего на персептрон, но с управляемой скоростью обучения
perceptron = SGDClassifier(**cfg["classifier"])

# 3. Для анимации сохраняем границы решений и ошибки
boundaries = []
errors = []
epochs = cfg["training"]["epochs"]     # максимум эпох

# Усреднение ошибок для остановки
window = cfg["training"]["window"]     # сколько эпох смотрим назад
delta = cfg["training"]["delta"]     # допустимое отклонение от среднего
early_stop = cfg["training"]["early_stop"]

start_time = time.time()
epoch_time = time.time()

for epoch in range(epochs):
    perceptron.fit(X, y)
    y_pred = perceptron.predict(X)
    errors.append(np.sum(y_pred != y))  # количество ошибок на текущей эпохе

    # Прямая разделяющая классы: w1*x1 + w2*x2 + b = 0
    w = perceptron.coef_[0]
    b = perceptron.intercept_[0]
    x_points = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    y_points = -(w[0] * x_points + b) / (w[1] + 1e-6)
    boundaries.append((x_points, y_points))

    print(f"Epoch {epoch}, error {errors[-1]}, epoch time {(time.time() - start_time)*1000} ms")
    epoch_time = time.time()

    # Ранняя остановка
    if len(errors) > window and early_stop:
        recent_mean = np.mean(errors[-window:-1])
        if abs(errors[-1] - recent_mean) <= delta:
            print(f"Остановка на эпохе {epoch + 1}: ошибка стабилизировалась")
            break

print(f"Total time elapsed: {(time.time() - start_time)*1000} ms")

n_frames = len(boundaries)

# 4. Настройка графиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Точки датасета
scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")
line, = ax1.plot([], [], "r-", linewidth=2)
ax1.set_title("Граница решений (SGDClassifier)")
ax1.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax1.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

# График ошибок
ax2.set_title("Ошибки классификации по эпохам")
ax2.set_xlim(0, epochs)
ax2.set_ylim(0, max(errors) + 2)
error_line, = ax2.plot([], [], "bo-")


# 5. Функция для обновления кадров анимации
def update(frame):
    x_points, y_points = boundaries[frame]
    line.set_data(x_points, y_points)

    error_line.set_data(range(frame + 1), errors[:frame + 1])
    return line, error_line


ani = FuncAnimation(fig, update, frames=n_frames, interval=cfg["animation"]["interval"],
                    repeat=cfg["animation"]["repeat"])
plt.show()
