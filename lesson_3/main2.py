import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Розміри площини
width = 10
height = 10

# Створення площини
x = np.linspace(0, width, 100)
y = np.linspace(0, height, 100)
x, y = np.meshgrid(x, y)

# Параметри джерела світла та спостерігача
light_position = np.array([3, 3, 10])
observer_position = np.array([2, 2, 0])

# Вектори нормалей до площини (зазвичай вони всі спрямовані вгору від площини)
normals = np.array([0, 0, 1])

# Вектори, що вказують на точки на площині
points_on_plane = np.stack([x, y, np.zeros_like(x)], axis=-1)

# Вектори, що вказують на світлове джерело від кожної точки на площині
light_directions = light_position - points_on_plane

# Вектори, що вказують на спостерігача від кожної точки на площині
observer_directions = observer_position - points_on_plane

normals = normals.astype('float64')
light_directions = light_directions.astype('float64')
observer_directions = observer_directions.astype('float64')


# Нормалізація векторів
normals /= np.linalg.norm(normals)
light_directions /= np.linalg.norm(light_directions, axis=-1, keepdims=True)
observer_directions /= np.linalg.norm(observer_directions, axis=-1, keepdims=True)

# Розрахунок освітленості (косинус кута між нормаллю, вектором до джерела та вектором до спостерігача)
brightness = np.maximum(np.sum(normals * (light_directions + observer_directions), axis=-1), 0)

# Відображення результатів в 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, brightness, cmap='viridis', rstride=5, cstride=5, alpha=0.8)

ax.scatter(light_position[0], light_position[1], light_position[2], color='red', marker='o', label='Light Source')
ax.scatter(observer_position[0], observer_position[1], observer_position[2], color='blue', marker='o', label='Observer')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Brightness')
ax.set_title('Lighting on a Plane')
ax.legend()

plt.show()
