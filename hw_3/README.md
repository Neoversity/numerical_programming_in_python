
# Word Embeddings and Vector Operations

## Домашнє завдання Тема 3

Цей домашнє завдання Тема 3 призначений для роботи з векторами слів (word embeddings) та виконання різних операцій з ними, таких як:
- Пошук найближчого слова до векторного добутку пар слів.
- Визначення кута між векторами слів.
- Обчислення векторних добутків слів для знаходження ортогональних векторів.

Проєкт використовує модель векторів слів, представлену в тривимірному просторі, щоб досліджувати семантичні зв'язки між словами.

## Встановлення

1. Склонуйте репозиторій:

    ```bash
    git clone https://github.com/yourusername/yourproject.git
    ```

2. Встановіть необхідні залежності:

    ```bash
    pip install -r requirements.txt
    ```

## Використання

### Пошук найближчого слова
Скрипт дозволяє знайти найближче слово до тривимірного вектора за допомогою функції `find_closest_word`. Наприклад:

```python
test_vector = [0.1, 0.2, 0.3]
closest_word = find_closest_word(test_vector, embeddings_df)
print(f"Найближче слово: {closest_word}")
```

### Обчислення кута між двома словами
Ви можете обчислити кут між двома словами за допомогою функції `calculate_angle_between_words`:

```python
angle = calculate_angle_between_words('city', 'country', embeddings_df)
print(f"Кут між 'city' і 'country': {angle} градусів")
```

### Обчислення векторного добутку
Для обчислення векторного добутку двох слів та знаходження найближчого слова до результату можна використовувати функцію `vector_cross_product`:

```python
word1 = 'city'
word2 = 'country'
cross_product_vector = vector_cross_product(word1_vector, word2_vector)
closest_word = find_closest_word(cross_product_vector, embeddings_df)
print(f"Найближче слово до векторного добутку 'city' і 'country': {closest_word}")
```

## Візуалізація векторів
Також проект включає можливість графічної візуалізації векторів у 3D-просторі за допомогою `plotly`:

```python
plot_word_vectors(test_vector, embeddings_df, closest_word_vector)
```

## Ліцензія

Цей проект ліцензовано за умовами MIT License.

## Автори

- **Антон Бабенко** - [bobantonbob](https://github.com/bobantonbob)
