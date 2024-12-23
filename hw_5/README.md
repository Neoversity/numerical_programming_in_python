
# Поліноміальна регресія з варіантами градієнтного спуску

## Огляд

Цей проєкт демонструє реалізацію різних алгоритмів оптимізації градієнтного спуску для поліноміальної регресії. До реалізованих методів належать:

- Градієнтний спуск (GD)
- Стохастичний градієнтний спуск (SGD)
- RMSProp
- Adam
- Nadam

Кожен метод застосовується до синтетичного набору даних, згенерованого для задачі поліноміальної регресії другого порядку. Порівнюється продуктивність кожного методу за ефективністю обчислень і кількістю ітерацій для збіжності.

## Методи

1. **Градієнтний спуск (GD)**: Використовує весь набір даних для обчислення градієнтів і оновлення параметрів моделі.
2. **Стохастичний градієнтний спуск (SGD)**: Випадковим чином вибирає один приклад для оновлення параметрів на кожній ітерації, що робить метод швидшим, але менш стабільним.
3. **RMSProp**: Адаптує швидкість навчання для кожного параметра на основі середнього квадратичного значення градієнтів.
4. **Adam**: Поєднує методи моментуму і RMSProp для швидкої та стабільної збіжності.
5. **Nadam**: Додає моментум Нестерова до алгоритму Adam для покращення швидкості збіжності.

## Кроки виконання коду

1. **Генерація даних**:
   - Використовується функція `np.random.rand()` для генерації двох ознак (x1 і x2), після чого обчислюється цільова змінна `y` за допомогою полінома другого степеня.

2. **Розширення поліноміальних ознак**:
   - Використовується `PolynomialFeatures` з бібліотеки `sklearn` для створення додаткових ознак для поліноміальної регресії.

3. **Реалізація градієнтного спуску**:
   - Кожен варіант градієнтного спуску (GD, SGD, RMSProp, Adam, Nadam) реалізовано в Python для оптимізації параметрів моделі поліноміальної регресії.

4. **Порівняння ефективності**:
   - Побудовано криві навчання для кожного методу, щоб візуалізувати зменшення помилки на кожній ітерації.
   - Для порівняння ефективності кожного методу вимірюється час виконання за допомогою `%timeit`.

## Результати

- **GD**: Стабільний, але повільний, особливо для великих наборів даних.
- **SGD**: Швидший, але потребує більше ітерацій для збіжності через нестабільність оновлень.
- **RMSProp**: Адаптивна швидкість навчання призводить до швидшої збіжності у порівнянні з GD та SGD.
- **Adam**: Забезпечує баланс між швидкістю та стабільністю.
- **Nadam**: Покращує Adam, забезпечуючи ще швидшу збіжність завдяки використанню моментуму Нестерова.

## Вимоги

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn

## Як запустити

1. Встановіть необхідні бібліотеки:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```

2. Запустіть Python-скрипт `hw_5.py` або використовуйте Jupyter/Colab notebook для виконання поліноміальної регресії та порівняння методів оптимізації.

## Висновок

Адаптивні методи, такі як RMSProp, Adam та Nadam, значно перевершують класичний градієнтний спуск та SGD за швидкістю збіжності та стабільністю. Надм, зокрема, демонструє найшвидшу збіжність з мінімальними осциляціями.

