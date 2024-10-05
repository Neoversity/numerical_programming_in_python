import pandas as pd
from sklearn.datasets import load_iris

# Завантажуємо дані Iris
iris = load_iris()
# Створюємо DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Виводимо перші рядки для перевірки
df.head()
