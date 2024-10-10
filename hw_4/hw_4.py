# -*- coding: utf-8 -*-
"""hw_4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qYYVWhtPUVWt5Ese1a_9yjn5_LZepnRn

1. Завантажте та ознайомтесь з даними:

  - Завантажте набір даних Breast Cancer за допомогою функції load_breast_cancer() з бібліотеки sklearn.
  - Перегляньте опис набору даних, щоб зрозуміти структуру та характеристики.
"""

# Імпортуємо необхідну бібліотеку з sklearn
from sklearn.datasets import load_breast_cancer

# Завантажуємо набір даних
breast_cancer_data = load_breast_cancer()

# Оглянемо структуру даних
print(breast_cancer_data.DESCR)  # Опис набору даних

# Виведемо основні ключі, щоб побачити, що міститься в наборі
print(breast_cancer_data.keys())


# Перевіримо перші кілька записів для більш детального ознайомлення
import pandas as pd
data = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
data['target'] = breast_cancer_data.target

# Виводимо перші 5 рядків
data.head()

"""2. Створіть DataFrame:

  - Створіть DataFrame, використовуючи дані з набору Breast Cancer.
"""

import pandas as pd

# Завантажуємо набір даних
breast_cancer_data = load_breast_cancer()

# Створюємо DataFrame з ознак (features)
data = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)

# Додаємо стовпчик з цільовими значеннями (target)
data['target'] = breast_cancer_data.target

# Переглянемо перші кілька рядків DataFrame
data.head()

"""3. Виведіть інформацію про дані:

  - Використовуйте функцію info() для виведення інформації про типи стовпців та кількість непустих значень в кожному стовпці.
"""

# Виводимо інформацію про DataFrame
data.info()

"""4. Виведіть описові статистики:

  - Використовуйте функцію describe() для виведення описових статистик даних.
"""

# Виводимо описові статистики для набору даних
data.describe()

"""5. Стандартизуйте дані:

  - Застосуйте процес стандартизації даних за допомогою функцій з конспекту або бібліотеки sklearn.
"""

# Імпортуємо необхідну бібліотеку для стандартизації
from sklearn.preprocessing import StandardScaler

# Створюємо об'єкт StandardScaler
scaler = StandardScaler()

# Відокремлюємо ознаки від цільового стовпця
features = data.drop('target', axis=1)

# Застосовуємо стандартизацію до ознак
scaled_features = scaler.fit_transform(features)

# Перетворимо стандартизовані дані назад у DataFrame для зручності
scaled_data = pd.DataFrame(scaled_features, columns=features.columns)

# Додаємо назад стовпчик 'target' до стандартизованого набору
scaled_data['target'] = data['target'].values

# Виводимо перші кілька рядків стандартизованого набору
scaled_data.head()

"""6. Побудуйте точкові діаграми:

  - Використайте функцію pairplot() з бібліотеки seaborn для побудови точкових діаграм між усіма стовпцями.
"""

# Імпортуємо необхідні бібліотеки
import seaborn as sns
import matplotlib.pyplot as plt

# Для побудови точкових діаграм використовуємо стандартизовані дані
# Оскільки функція pairplot() працює ефективніше з невеликою кількістю стовпців,
# оберемо лише декілька для візуалізації

# Візьмемо лише кілька перших ознак і стовпчик 'target'
selected_columns = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'target']

# Створюємо новий DataFrame з цих ознак
plot_data = scaled_data[selected_columns]

# Використовуємо pairplot з розрізненням за стовпчиком 'target'
sns.pairplot(plot_data, hue='target', diag_kind='kde')

# Відображаємо графік
plt.show()

"""Діаграма точок  демонструє взаємозв’язки між обраними ознаками. Вона також відображає різні категорії (target) злоякісних та доброякісних пухлин за допомогою кольорів. Це надає можливість аналізувати, які ознаки можуть краще диференціювати ці дві категорії.

7. Обчисліть матриці відстаней:

  - Використовуйте алгоритми та функції з конспекту для обчислення матриці відстаней для різних метрик: cityblock, cosine, euclidean, l1, manhattan.
"""

# Імпортуємо необхідні бібліотеки
from scipy.spatial.distance import pdist, squareform
import pandas as pd

# Обираємо лише ознаки для обчислення відстаней (без стовпця 'target')
features_only = scaled_data.drop('target', axis=1)

# Обчислюємо матрицю відстаней для різних метрик

# Cityblock (або manhattan, l1)
cityblock_dist = squareform(pdist(features_only, metric='cityblock'))
cosine_dist = squareform(pdist(features_only, metric='cosine'))
euclidean_dist = squareform(pdist(features_only, metric='euclidean'))

# Створимо DataFrame для кожної метрики для зручного перегляду
cityblock_df = pd.DataFrame(cityblock_dist)
cosine_df = pd.DataFrame(cosine_dist)
euclidean_df = pd.DataFrame(euclidean_dist)

# Виводимо перші кілька рядків кожної з матриць відстаней
print("Cityblock Distance Matrix:")
print(cityblock_df.head())

print("\nCosine Distance Matrix:")
print(cosine_df.head())

print("\nEuclidean Distance Matrix:")
print(euclidean_df.head())

"""8. Візуалізуйте отримані матриці:

  - Використайте функцію heatmap з бібліотеки seaborn або інші методи візуалізації для представлення отриманих матриць відстаней.
"""

# Імпортуємо бібліотеки
import seaborn as sns
import matplotlib.pyplot as plt

# Вибираємо перші 20 рядків і стовпців для кожної матриці
subset_cityblock = cityblock_df.iloc[:20, :20]
subset_cosine = cosine_df.iloc[:20, :20]
subset_euclidean = euclidean_df.iloc[:20, :20]
subset_l1 = cityblock_df.iloc[:20, :20]  # L1 це те саме, що і Cityblock але вивиду для нагляду
subset_manhattan = cityblock_df.iloc[:20, :20]  # Manhattan це теж Cityblock але вивиду для нагляду

# Функція для створення теплової карти
def plot_heatmap(data, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, linewidths=0.5)
    plt.title(title)
    plt.show()

# Візуалізація для Cityblock Distance Matrix (перші 20 зразків)
plot_heatmap(subset_cityblock, "Cityblock Distance Matrix (Subset of 20 samples)")

# Візуалізація для Cosine Distance Matrix (перші 20 зразків)
plot_heatmap(subset_cosine, "Cosine Distance Matrix (Subset of 20 samples)")

# Візуалізація для Euclidean Distance Matrix (перші 20 зразків)
plot_heatmap(subset_euclidean, "Euclidean Distance Matrix (Subset of 20 samples)")

# Візуалізація для L1 Distance Matrix (перші 20 зразків)
plot_heatmap(subset_l1, "L1 Distance Matrix (Subset of 20 samples)")

# Візуалізація для Manhattan Distance Matrix (перші 20 зразків)
plot_heatmap(subset_manhattan, "Manhattan Distance Matrix (Subset of 20 samples)")

# Імпортуємо бібліотеки
import seaborn as sns
import matplotlib.pyplot as plt

# Візуалізація для Cityblock Distance Matrix (всі зразки, збільшуємо розмір графіка)
plt.figure(figsize=(20, 16))
sns.heatmap(cityblock_df, cmap='coolwarm', cbar=True)
plt.title("Cityblock Distance Matrix (Full Data)")
plt.show()

# Візуалізація для Cosine Distance Matrix (всі зразки, збільшуємо розмір графіка)
plt.figure(figsize=(20, 16))
sns.heatmap(cosine_df, cmap='coolwarm', cbar=True)
plt.title("Cosine Distance Matrix (Full Data)")
plt.show()

# Візуалізація для Euclidean Distance Matrix (всі зразки, збільшуємо розмір графіка)
plt.figure(figsize=(20, 16))
sns.heatmap(euclidean_df, cmap='coolwarm', cbar=True)
plt.title("Euclidean Distance Matrix (Full Data)")
plt.show()

# Імпортуємо бібліотеки
import seaborn as sns
import matplotlib.pyplot as plt

# Створюємо кореляційну матрицю
correlation_matrix = scaled_data.corr()

# Створюємо теплову карту
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)

# Додаємо заголовок
plt.title("Correlation Matrix Heatmap")
plt.show()

"""9. Зробіть висновок:

  - Зробіть висновок на основі аналізу отриманих результатів та порівняйте ефективність різних метрик відстаней для даного набору даних.

## Висновок на основі матриць відстаней:
1. Cityblock (або Manhattan) Distance:

  - Ця метрика обчислює суму абсолютних різниць між координатами двох точок. Вона добре підходить для задач, де важливі зміни в кожній окремій ознаці.
  - Переваги: Чутлива до відмінностей у кожній ознаці незалежно від їхніх масштабів. Це корисно для наборів даних, у яких значення можуть суттєво відрізнятися за різними ознаками.
  - Недоліки: Може бути менш ефективною, коли дані мають сильну кореляцію між ознаками або коли напрямок ознак є важливим.
2. Cosine Distance:

  - Косинусна відстань вимірює кут між двома векторами (двома точками). Ця метрика більше фокусується на напрямку векторів, ніж на їхній довжині.
  - Переваги: Ідеально підходить для задач, де важливий напрямок ознак, а не їхній масштаб (наприклад, для текстових даних або векторних представлень). Може виявити схожість між зразками, навіть якщо вони мають різні масштаби ознак.
  - Недоліки: Якщо важливі абсолютні значення ознак (замість їх напрямку), ця метрика може не бути кращим вибором.
3. Euclidean Distance:

  - Це класична метрика, яка вимірює "прямолінійну" відстань між двома точками в просторі. Вона використовується для простих задач класифікації та кластеризації.
  - Переваги: Легко інтерпретується і часто використовується для вимірювання відстані між точками. Добре підходить для наборів даних, у яких ознаки мають однаковий масштаб.
  - Недоліки: Якщо ознаки мають різні масштаби, Евклідова відстань може бути менш чутливою до суттєвих відмінностей у даних. Потрібна стандартизація ознак для коректного використання.
4. L1 Distance:

  - Це те ж саме, що і Cityblock або Manhattan відстань, тому висновки залишаються ті самі: це метрика, яка підсумовує абсолютні різниці між координатами двох точок.
  - Переваги: Особливо корисна для задач, де різниці між значеннями ознак є ключовими для моделі.
5. Manhattan Distance:

  - Як уже було згадано, це те ж саме, що і Cityblock або L1, тому всі переваги та недоліки застосовуються і до цієї метрики.

## Порівняння ефективності метрик:
- Cityblock/L1/Manhattan: Ці метрики добре працюють для задач, де важливо оцінювати зміни в кожній окремій ознаці. Якщо набори даних мають відмінні значення ознак або вам важливо враховувати кожну відмінність, вони будуть корисні.
-
Cosine Distance: Ця метрика більше орієнтована на напрямок векторів і підходить для задач, де схожість визначається не стільки масштабом, скільки напрямком ознак. Вона чудово працює в текстових задачах або з векторними представленнями.

- Euclidean Distance: Якщо ознаки у вашому наборі даних мають однакові масштаби, Евклідова відстань може бути найкращим вибором. Але для неоднорідних наборів даних варто спочатку стандартизувати ознаки, щоб уникнути проблем з різними масштабами.

## Загальні рекомендації:

  - Якщо у вашому наборі даних важливі зміни між окремими ознаками, варто використовувати Cityblock/L1/Manhattan.
  - Якщо важливий напрямок векторів або ви працюєте з текстовими даними, Cosine Distance буде корисною.
  - Якщо ви маєте стандартизовані або рівномірно масштабовані дані, Euclidean Distance буде інтуїтивно зрозумілою та легкою в інтерпретації.

### Таким чином, для даного набору даних можна порекомендувати використовувати Cityblock/L1/Manhattan для задач, де важливі абсолютні зміни в ознаках. Cosine Distance краще підходить для більш високовимірних даних або задач, де важливий напрямок ознак, а не їхній масштаб.
"""