import numpy as np

A = np.array([[66, -10, 16, 14, 65], [-91, 28, 97, -42, 4], [13, 50, -96, 92, -85], [21, -96, 49, 34, 93], [-53, 96, 80, 96, -68]])
B = np.array([[1, 1, 0, 1, 0], [1, 0, 1, 1, 1], [0, 0, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]])

C = np.multiply(A, B)
print(C[:, 2])

innerProduct = np.dot(C[:, 2], C[4, :])
print(innerProduct)

maxValueFourthColumnC = np.amax(C[:, 3])
print(maxValueFourthColumnC)
indexMaxValueFourthColumnC = np.argmax(C[:, 3])
print(indexMaxValueFourthColumnC)
maxColumnIndex = 3 #trivial
maxColumnRow = indexMaxValueFourthColumnC - 1


firstRowC = C[0, :]

DRow0 = np.multiply(firstRowC, firstRowC)
DRow1 = np.multiply(firstRowC, C[1, :])
DRow2 = np.multiply(firstRowC, C[2, :])
DRow3 = np.multiply(firstRowC, C[3, :])
DRow4 = np.multiply(firstRowC, C[4, :])

D = np.vstack((DRow0, DRow1, DRow2, DRow3, DRow4))
print(D)

firstRowD = D[0, :]

ERow0 = np.multiply(firstRowD, firstRowD)
ERow1 = np.multiply(firstRowD, D[1, :])
ERow2 = np.multiply(firstRowD, D[2, :])
ERow3 = np.multiply(firstRowD, D[3, :])
ERow4 = np.multiply(firstRowD, D[4, :])

E = np.vstack((ERow0, ERow1, ERow2, ERow3, ERow4))
print(E)

