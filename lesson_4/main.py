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




from math import sqrt
import numpy as np


# Function to return the Frobenius
# Norm of the given matrix
def frobeniusNorm(mat):


    row = np.shape(mat)[0]
    col = np.shape(mat)[1]
    # To store the sum of squares of the
    # elements of the given matrix
    sumSq = 0
    for i in range(row):
        for j in range(col):
            sumSq += pow(mat[i][j], 2)


    # Return the square root of
    # the sum of squares
    res = sqrt(sumSq)
    return round(res, 5)


# Driver code


mat = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]


print(frobeniusNorm(mat))
