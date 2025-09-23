import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from matplotlib.animation import FuncAnimation

# 1. Генерация датасета (сделаем задачу не слишком лёгкой)
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=1.0,  # чем меньше значение, тем больше пересечение классов
    flip_y=0.1,  # вероятность случайной ошибки в метке (шум)
    random_state=42
)

# 2. Инициализация классификатора, похожего на персептрон, но с управляемой скоростью обучения
perceptron = SGDClassifier(
    loss="perceptron",  # используем функцию потерь персептрона
    learning_rate="constant",
    eta0=0.01,  # скорость обучения (меньше = медленнее)
    max_iter=1,  # обучаем только 1 эпоху за вызов .fit()
    warm_start=True,  # сохраняем веса между вызовами .fit()
    tol=None,
    random_state=42
)

# 3. Для анимации сохраняем границы решений и ошибки
boundaries = []
errors = []
epochs = 20     # максимум эпох

# Усреднение ошибок для остановки
window = 5      # сколько эпох смотрим назад
delta = 1.0     # допустимое отклонение от среднего
early_stop = False

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

    # Ранняя остановка
    if len(errors) > window and early_stop:
        recent_mean = np.mean(errors[-window:-1])
        if abs(errors[-1] - recent_mean) <= delta:
            print(f"Остановка на эпохе {epoch + 1}: ошибка стабилизировалась")
            break

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


ani = FuncAnimation(fig, update, frames=n_frames, interval=700, repeat=False)
plt.show()
