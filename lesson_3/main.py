import numpy as np

# def normal_vector(u, v):
#     # Обчислення векторного добутку (cross product)
#     n = np.cross(u, v)

#     return n

# # Приклад використання
# u = np.array([1, 2, 3])
# v = np.array([4, 5, 6])

# normal = normal_vector(u, v)
# print("Вектор нормалі до площини:", normal)




# a = [2, -2, -3]
# b = [4, 0, 6]
# c = [-7, -7, 1]
# omega = np.linalg.det(np.dstack([a,b,c]))
# # Мішаний добуток a, b і c
# mixed_dot_product = np.linalg.det(np.dstack([a,b,c]))
# mixed_dot_product

# # Векторний добуток b і c
# cross_product = np.cross(b, c)

# # Мішаний добуток a, b і c
# mixed_dot_product = np.dot(a, cross_product)

# # Виведення результату
# print("Мішаний добуток:", mixed_dot_product)








import numpy as np

def are_vectors_linearly_independent(vectors):
    # Створення розширеної матриці з векторів
    matrix = np.array(vectors).T

    # Ранг матриці
    rank_matrix = np.linalg.matrix_rank(matrix)

    # Кількість векторів
    num_vectors = len(vectors)

    # Вектори лінійно незалежні, якщо ранг матриці рівний кількості векторів
    return rank_matrix == num_vectors

# Приклад використання
# vectors1 = np.array([1, 2, 3])
# vectors2 = np.array([-2, 1, -1])
# vectors3 = np.array([3, 2, -1])

# Приклад використання
vectors1 = np.array([1, 2, -3])
vectors2 = np.array([-1, 2, 4])
vectors3 = np.array([1, 6, -2])

# Перевірка лінійної незалежності векторів
result = are_vectors_linearly_independent([vectors1, vectors2, vectors3])

if result:
    print("Вектори лінійно незалежні.")
else:
    print("Вектори лінійно залежні.")
