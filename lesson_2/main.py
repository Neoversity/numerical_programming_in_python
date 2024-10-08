import numpy as np

# Приклад матриці
# matrix_a = np.array([[1, 2, 3],
#                     [4, 5, 6]])

matrix_a = np.array([[4, 0], [3, -5]])

# Обчислення транспонованої матриці
matrix_a_transposed = np.transpose(matrix_a) # або matrix_a.T

# Обчислення добутку матриці на її транспоновану
result = np.dot(matrix_a_transposed, matrix_a) # або matrix_a @ matrix_a_transposed

# Виведення результату
print("Матриця A:")
print(matrix_a)

print("\nТранспонована матриця A:")
print(matrix_a_transposed)

print("\nДобуток матриці A на її транспоновану:")
print(result)



# Обчислення власних векторів та власних значень
eigenvalues, eigenvectors = np.linalg.eig(result)

# Виведення результату
print("Власні значення:")
print(eigenvalues)

print("\nВласні вектори:")
print(eigenvectors)



# Отримання індексів, які відсортовують масив за першим стовпцем
sorted_indices = np.argsort(eigenvectors[:, 0])

V = eigenvectors[sorted_indices]

# Виведення результату
print("Вихідний масив:")
print(eigenvectors)

print("\nМатриця V:")
print(V)


AV = np.dot(matrix_a, V)
def matrix_norm(mtr):

     # Обчислення кореня квадратного з суми елементів для кожного стовпця
    sqrt_sum_columns = np.sqrt(np.sum(mtr**2, axis=0))
    res = mtr / sqrt_sum_columns
    return res

AV = matrix_norm(AV)

print(AV)
