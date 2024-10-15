# # Import Numpy package and the norm function
# import numpy as np
# from numpy.linalg import norm


# # Define a vector
# v = np.array([2,3,1,0])

# # Take the q-norm which p=2
# p = 2
# v_norm = norm(v, ord=p)

# # Print values
# print('The vector: ', v)
# print('The vector norm: ', v_norm)


# # max norm of a vector
# from numpy import inf
# from numpy import array
# from numpy.linalg import norm
# v = array([1, 2, 3])
# print(v)
# maxnorm = norm(v, inf)
# print(maxnorm)


# from math import sqrt
# import numpy as np


# # Function to return the Frobenius
# # Norm of the given matrix
# def frobeniusNorm(mat):


#     row = np.shape(mat)[0]
#     col = np.shape(mat)[1]
#     # To store the sum of squares of the
#     # elements of the given matrix
#     sumSq = 0
#     for i in range(row):
#         for j in range(col):
#             sumSq += pow(mat[i][j], 2)


#     # Return the square root of
#     # the sum of squares
#     res = sqrt(sumSq)
#     return round(res, 5)


# # Driver code


# mat = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]


# print(frobeniusNorm(mat))


# from scipy.spatial.distance import cityblock
# import pandas as pd

# # define DataFrame
# df = pd.DataFrame({"A": [2, 4, 4, 6], "B": [5, 5, 7, 8], "C": [9, 12, 12, 13]})

# # calculate Manhattan distance between columns A and B
# cityblock(df.A, df.B)




# from scipy.spatial import distance
# distance.dice([1, 0, 0], [0, 1, 0])

# distance.dice([1, 0, 0], [1, 1, 0])

# distance.dice([1, 0, 0], [2, 0, 0])



# def data_scale(data, scaler_type='minmax'):
#     from sklearn.preprocessing import MinMaxScaler
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.preprocessing import Normalizer
#     if scaler_type == 'minmax':
#         scaler = MinMaxScaler()
#     if scaler_type == 'std':
#         scaler = StandardScaler()
#     if scaler_type == 'norm':
#         scaler = Normalizer()

#     scaler.fit(data)
#     res = scaler.transform(data)
#     return res




# from scipy.stats import kendalltau

# # Використовуємо ті самі дані X і Y

# # Розрахунок коефіцієнта кореляції Кендала
# kendall_coefficient, _ = kendalltau(X, Y)

# print(f"Коефіцієнт кореляції Кендала: {kendall_coefficient}")

# #https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016
# import seaborn as sb
# from google.colab import drive
# drive.mount('/content/drive')

# df = pd.read_csv("/content/drive/MyDrive/!Kafedra/GOIT/woolf/numprog/mod2_2/suicide-rates.csv")
