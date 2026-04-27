import numpy as np
import matplotlib.pyplot as plt


def interaction_matrix(x, y, Z):
    # Матрица взаимодействия L из лекции (слайд 11)
    return np.array([
        [-1 / Z, 0, x / Z, x * y, -(1 + x ** 2), y],
        [0, -1 / Z, y / Z, 1 + y ** 2, -x * y, -x]
    ])


def simulate(v_camera, steps=100, dt=0.05):
    # Начальные координаты квадрата (4 точки)
    s = np.array([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]])
    Z = 2.0  # Начальная глубина

    history = [s.copy()]
    for _ in range(steps):
        ds_dt = []
        for point in s:
            L = interaction_matrix(point[0], point[1], Z)
            ds_dt.append(L @ v_camera)

        s += np.array(ds_dt).reshape(4, 2) * dt
        Z -= v_camera[2] * dt  # Обновление глубины: Z_dot = -v_z
        history.append(s.copy())

    return np.array(history)


def plot_trajectories(history, title, filename):
    plt.figure(figsize=(6, 6))
    for i in range(4):
        plt.plot(history[:, i, 0], history[:, i, 1], label=f'Point {i + 1}')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    print(f"График сохранен: {filename}")


# --- Запуск симуляций ---
# 1а. К изображению (v_z > 0)
hist_fwd = simulate([0, 0, 0.5, 0, 0, 0])
plot_trajectories(hist_fwd, 'Движение к объекту (v_z > 0)', 'task4_2_forward.png')

# 2а. Вращение по часовой стрелке (w_z > 0)
hist_rot = simulate([0, 0, 0, 0, 0, 0.8])
plot_trajectories(hist_rot, 'Вращение камеры (w_z > 0)', 'task4_2_rotation.png')
