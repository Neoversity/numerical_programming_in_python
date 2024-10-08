# -*- coding: utf-8 -*-
"""hw_3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VeM7q0ZFxvkN1Jv4zLnh9BfehaigPDCA

1. Визначте DataFrame з тривимірними векторами слів:

  - Завантажте модель word embeddings, використовуючи файл, який містить NLP модель.
  - Витягніть тривимірні вектори для всіх слів з цієї моделі.
  - Створіть DataFrame, в якому буде міститися інформація про слова та їхні тривимірні вектори.
"""

import pandas as pd
import pickle

# Підключення Google Drive (якщо файл на Google Drive)
from google.colab import drive
drive.mount('/content/drive')

# Шлях до файлу на Google Drive
file_path = '/content/drive/MyDrive/word_embeddings_subset.p'

# Завантаження моделі векторів слів
with open(file_path, 'rb') as file:
    word_embeddings = pickle.load(file)

# Створюємо DataFrame з даних моделі
word_embeddings_df = pd.DataFrame(word_embeddings)

# Перетворюємо колонки (слова) на рядки
words = word_embeddings_df.columns
vectors = word_embeddings_df.T  # Транспонуємо матрицю, щоб слова були рядками

# Залишаємо лише перші три виміри векторів
three_dim_vectors = vectors.iloc[:, :3]

# Створюємо новий DataFrame з тривимірними векторами та словами
three_dim_embeddings_df = pd.DataFrame(three_dim_vectors.values, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
three_dim_embeddings_df['Word'] = words

# Відображення DataFrame
three_dim_embeddings_df.head()

"""2. Визначте функції для пошуку найближчого слова:

  - Напишіть функцію, яка приймає тривимірний вектор та знаходить найближче слово в моделі word embeddings_subset.
  - Використайте цю функцію для кількох прикладів та переконайтеся, що результати є коректними.
"""

import numpy as np
import pandas as pd
import plotly.graph_objs as go

# Функція для обчислення Евклідової відстані між двома векторами
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# Функція для пошуку найближчого слова
def find_closest_word(vector, embeddings_df):
    min_distance = float('inf')
    closest_word = None
    closest_word_vector = None

    for index, row in embeddings_df.iterrows():
        word_vector = np.array([row['Dimension 1'], row['Dimension 2'], row['Dimension 3']])
        distance = euclidean_distance(vector, word_vector)
        if distance < min_distance:
            min_distance = distance
            closest_word = row['Word']
            closest_word_vector = word_vector

    return closest_word, closest_word_vector

# Функція для інтерактивного графічного відображення векторів слів
def plot_word_vectors_interactive(target_vector, embeddings_df, closest_word_vector):
    # Створюємо списки для всіх векторів
    x_vals = []
    y_vals = []
    z_vals = []
    text_vals = []

    # Збираємо дані для векторів слів
    for index, row in embeddings_df.iterrows():
        word_vector = np.array([row['Dimension 1'], row['Dimension 2'], row['Dimension 3']])
        x_vals.append(word_vector[0])
        y_vals.append(word_vector[1])
        z_vals.append(word_vector[2])
        text_vals.append(row['Word'])

    # Створюємо слід для всіх слів
    trace_words = go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='markers+text',
        marker=dict(size=5, color='blue'),
        text=text_vals
    )

    # Додаємо цільовий вектор
    trace_target = go.Scatter3d(
        x=[target_vector[0]], y=[target_vector[1]], z=[target_vector[2]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Target Vector'
    )

    # Додаємо найближчий вектор
    trace_closest = go.Scatter3d(
        x=[closest_word_vector[0]], y=[closest_word_vector[1]], z=[closest_word_vector[2]],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Closest Word'
    )

    # Створюємо графік
    layout = go.Layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=[trace_words, trace_target, trace_closest], layout=layout)

    # Відображення інтерактивного графіку
    fig.show()

# Приклад вектора для пошуку найближчого слова
test_vector = np.array([0.1, 0.2, 0.3])
closest_word, closest_word_vector = find_closest_word(test_vector, three_dim_embeddings_df)
print(f"Найближче слово для вектора {test_vector}: {closest_word}")

# Інтерактивне відображення графіку
plot_word_vectors_interactive(test_vector, three_dim_embeddings_df, closest_word_vector)

"""3. Обчисліть векторний добуток для знаходження ортогонального слова:

  - Виберіть декілька довільних пар слів.
  - Обчисліть векторний добуток для кожної пари слів та використайте раніше написану функцію для знаходження найближчого слова.
  - Проаналізуйте результати та спробуйте їх інтерпретувати.
"""

import numpy as np

# Функція для обчислення векторного добутку
def vector_cross_product(word1_vector, word2_vector):
    return np.cross(word1_vector, word2_vector)

# Вибираємо дві пари слів та обчислюємо їх вектори
def get_word_vector(word, embeddings_df):
    # Отримуємо вектор для заданого слова
    word_row = embeddings_df[embeddings_df['Word'] == word]
    if not word_row.empty:
        return np.array([word_row['Dimension 1'].values[0], word_row['Dimension 2'].values[0], word_row['Dimension 3'].values[0]])
    else:
        raise ValueError(f"Слово '{word}' не знайдено у векторній моделі.")

# Приклад вибору слів для векторного добутку
word1 = 'city'
word2 = 'country'

word1_vector = get_word_vector(word1, three_dim_embeddings_df)
word2_vector = get_word_vector(word2, three_dim_embeddings_df)

# Обчислюємо векторний добуток
cross_product_vector = vector_cross_product(word1_vector, word2_vector)

# Використовуємо раніше написану функцію для пошуку найближчого слова до векторного добутку
closest_word, _ = find_closest_word(cross_product_vector, three_dim_embeddings_df)
print(f"Найближче слово до векторного добутку {word1} і {word2}: {closest_word}")

# Тестуємо ще одну пару слів
word3 = 'London'
word4 = 'Canada'

word3_vector = get_word_vector(word3, three_dim_embeddings_df)
word4_vector = get_word_vector(word4, three_dim_embeddings_df)

# Обчислюємо векторний добуток для другої пари
cross_product_vector2 = vector_cross_product(word3_vector, word4_vector)

# Шукаємо найближче слово до другого векторного добутку
closest_word2, _ = find_closest_word(cross_product_vector2, three_dim_embeddings_df)
print(f"Найближче слово до векторного добутку {word3} і {word4}: {closest_word2}")

import numpy as np
import plotly.graph_objs as go

# Функція для відображення векторів та їх векторних добутків
def plot_vectors_with_cross_products(word1, word2, cross_product_word, word3, word4, cross_product_word2, embeddings_df):
    fig = go.Figure()

    # Отримуємо вектори для обраних слів
    word1_vector = get_word_vector(word1, embeddings_df)
    word2_vector = get_word_vector(word2, embeddings_df)
    word3_vector = get_word_vector(word3, embeddings_df)
    word4_vector = get_word_vector(word4, embeddings_df)
    cross_product_vector = get_word_vector(cross_product_word, embeddings_df)
    cross_product_vector2 = get_word_vector(cross_product_word2, embeddings_df)

    # Створюємо вектори для перших двох слів і їхнього векторного добутку
    fig.add_trace(go.Scatter3d(x=[0, word1_vector[0]], y=[0, word1_vector[1]], z=[0, word1_vector[2]],
                               mode='lines+markers', name=f'Vector: {word1}', line=dict(color='blue', width=5)))
    fig.add_trace(go.Scatter3d(x=[0, word2_vector[0]], y=[0, word2_vector[1]], z=[0, word2_vector[2]],
                               mode='lines+markers', name=f'Vector: {word2}', line=dict(color='red', width=5)))
    fig.add_trace(go.Scatter3d(x=[0, cross_product_vector[0]], y=[0, cross_product_vector[1]], z=[0, cross_product_vector[2]],
                               mode='lines+markers', name=f'Cross Product: {word1} x {word2} -> {cross_product_word}',
                               line=dict(color='green', width=5)))

    # Створюємо вектори для наступних двох слів і їхнього векторного добутку
    fig.add_trace(go.Scatter3d(x=[0, word3_vector[0]], y=[0, word3_vector[1]], z=[0, word3_vector[2]],
                               mode='lines+markers', name=f'Vector: {word3}', line=dict(color='purple', width=5)))
    fig.add_trace(go.Scatter3d(x=[0, word4_vector[0]], y=[0, word4_vector[1]], z=[0, word4_vector[2]],
                               mode='lines+markers', name=f'Vector: {word4}', line=dict(color='orange', width=5)))
    fig.add_trace(go.Scatter3d(x=[0, cross_product_vector2[0]], y=[0, cross_product_vector2[1]], z=[0, cross_product_vector2[2]],
                               mode='lines+markers', name=f'Cross Product: {word3} x {word4} -> {cross_product_word2}',
                               line=dict(color='green', width=5)))

    # Налаштовуємо осі
    fig.update_layout(scene=dict(
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        zaxis_title='Dimension 3'
    ))

    # Відображаємо графік
    fig.show()

# Викликаємо функцію для відображення векторів
plot_vectors_with_cross_products('city', 'country', 'Guinea', 'London', 'Canada', 'Liberia', three_dim_embeddings_df)

"""## Аналіз результатів:
1. Пара слів: "city" та "country"

  - Після обчислення векторного добутку для цих слів ми отримуємо новий вектор, який ортогональний до обох векторів, що представляють "city" і "country".
  - Пошук найближчого слова до цього векторного добутку може дати слово, яке пов'язане з обома поняттями, наприклад, слово, яке має спільні риси або відображає властивості, притаманні як містам, так і країнам.
  - Інтерпретація: Якщо найближчим словом виявиться щось на зразок "region" або "capital", це може свідчити про зв'язок між містом і країною як географічними поняттями. Ортогональний вектор може відображати загальні риси цих двох понять, наприклад, адміністративний поділ або територіальні одиниці.
2. Пара слів: "London" та "Canada"

  - Після обчислення векторного добутку для цих двох слів, ми отримуємо вектор, ортогональний до векторів, що представляють "London" і "Canada".
  - Найближче слово до цього векторного добутку може мати зв'язок із Лондоном і Канадою, наприклад, історичний або культурний аспект.
  - Інтерпретація: Якщо найближчим словом буде щось типу "Commonwealth", це може відображати історичний зв'язок між Канадою та Лондоном (Великобританією), враховуючи, що Канада була частиною Британської імперії, і Лондон є столицею цієї імперії. Ортогональний вектор може показувати деяку унікальну характеристику цього зв'язку.
## Загальні висновки:
  - Векторний добуток дає вектор, який є ортогональним до двох обраних векторів. Це означає, що він відображає щось, що не є прямо пов'язаним з початковими векторами, але водночас знаходиться у просторі, визначеному обома векторами.
  - Найближче слово до цього вектора може мати певні спільні риси з обома обраними словами або їх узагальнення.
  - Результати можуть відрізнятися залежно від специфіки векторної моделі, тому деякі слова можуть бути більш зрозумілими, інші — менш очевидними.

4. Напишіть функції визначення кута між словами:

  - Розробіть функцію, яка обчислює кут між векторами для довільних двох слів.
  - Протестируйте цю функцію для різних пар слів.
  - Розгляньте отримані результати та спробуйте визначити їхню інтерпретацію.
"""

import numpy as np

# Функція для обчислення кута між двома векторами
def calculate_angle_between_words(word1, word2, embeddings_df):
    # Отримуємо вектори для двох слів
    word1_vector = get_word_vector(word1, embeddings_df)
    word2_vector = get_word_vector(word2, embeddings_df)

    # Обчислюємо скалярний добуток
    dot_product = np.dot(word1_vector, word2_vector)

    # Обчислюємо довжини векторів
    norm_word1 = np.linalg.norm(word1_vector)
    norm_word2 = np.linalg.norm(word2_vector)

    # Обчислюємо косинус кута
    cos_theta = dot_product / (norm_word1 * norm_word2)

    # Обчислюємо кут у радіанах і переводимо в градуси
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Використовуємо np.clip, щоб уникнути помилок округлення
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

# Тестуємо функцію для кількох пар слів
word_pairs = [('city', 'country'), ('London', 'Canada'), ('Guinea', 'Liberia'), ('city', 'London')]

for word1, word2 in word_pairs:
    angle = calculate_angle_between_words(word1, word2, three_dim_embeddings_df)
    print(f"Кут між '{word1}' і '{word2}': {angle:.2f} градусів")

"""1. Кут між 'city' і 'country': 31.42 градусів

  - Це порівняно малий кут, що свідчить про близькість між поняттями "місто" (city) і "країна" (country). Обидва ці слова є географічними термінами, які часто використовуються разом або в контексті адміністративного поділу. Тому не дивно, що векторні моделі відображають їх близькість у семантичному просторі. Вони не є тотожними, але мають багато спільних аспектів.

2. Кут між 'London' і 'Canada': 109.52 градусів

  - Це досить великий кут, що вказує на те, що ці два поняття є досить різними в семантичному просторі. "Лондон" — це столиця Великобританії, тоді як "Канада" — це ціла країна, яка географічно та культурно знаходиться на значній відстані від Лондона. Їхнє значення не дуже тісно пов'язане, але, враховуючи історичні зв'язки між Канадою та Великобританією (через Співдружність), цей кут не максимальний.

3. Кут між 'Guinea' і 'Liberia': 68.28 градусів

  - Ці два слова представляють країни, які розташовані в Західній Африці. Кут у 68 градусів вказує на те, що, хоча ці країни можуть мати певні спільні риси (географічна близькість, спільні історичні та культурні аспекти), вони не є дуже схожими в семантичному просторі. Це може бути пов'язано з різними політичними, економічними або іншими характеристиками, що відрізняють ці країни.
  
4. Кут між 'city' і 'London': 120.08 градусів

  - Це досить великий кут, що вказує на те, що "місто" як загальне поняття і "Лондон" як конкретне місто мають значні відмінності у своїх значеннях. Хоча "Лондон" є містом, він має власний унікальний контекст, історію та характеристики, які не можна повністю узагальнити терміном "місто". Тому модель правильно відображає цей великий кут між загальним і конкретним поняттям.

## Загальна інтерпретація:

  - Малий кут між словами вказує на те, що ці слова мають близьке або схоже значення. Наприклад, "city" і "country" є близькими поняттями, оскільки обидва пов'язані з географічними термінами.
  - Великий кут між словами вказує на суттєві відмінності в їхньому семантичному значенні. Наприклад, "London" і "Canada" мають великий кут, оскільки це абсолютно різні за суттю поняття (місто проти країни).
  - Середні кути (наприклад, 68 градусів між "Guinea" і "Liberia") можуть вказувати на часткову схожість, але з суттєвими відмінностями, що можуть бути культурними, політичними або іншими.

## Отримані результати допомагають побачити, як слова співвідносяться у векторному просторі й показують їхню семантичну близькість або відмінність.
"""