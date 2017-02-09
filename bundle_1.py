## BUNDLE ADJUSTMENT - 2016

import numpy as np
import pylab as plt
import scipy.linalg as sl

plt.show()

## Programme Python qui renvoie l'évolution de la plus petite valeur propre, de la plus grande valeur propre, et du conditionnement de la matrice A^T * A en fonction du nombre de points en correspondances DANS LE CAS DU CERCLE

# n : nombre de points (sans unité)
# f : distance focale (mètres)
# a : distance entre le plan des caméras et le plan des points (mètres)
# r : rayon du cercle formé par les points (mètres)
# b : distance entre les quatre caméras (mètres)

def coord_points_cercle(n, a, r, b, j):
    x, y = f * r * np.cos(2 * np.pi / n * j), f * r * np.sin(2 * np.pi / n * j)
    return np.array([x+f*b, x, x-f*b, x, y, y+f*b, y, y-f*b]) / a

def matrice_Aj(coord, b):
    x1, x2, x3, x4 = coord[0], coord[1],  coord[2], coord[3]
    y1, y2, y3, y4 = coord[4], coord[5], coord[6], coord[7]
    return b / f**2 * np.array([[y1*y2 + f**2 + x2*y1, -x1*y2 - x1*x2 - f**2, f*(-x1 + y1), -y1*y2 - f**2 - x1*y2, x2*y1 + x1*x2 + f**2, f*(x2 - y2), 0, 0, 0, 0, 0, 0], 
        [2*(y1*y3 + f**2), -2*x1*y3, -2*f*x1, 0, 0, 0, -2*(y1*y3 + f**2), 2*x3*y1, 2*f*x3, 0, 0, 0], 
        [y1*y4 + f**2 - x4*y1, -x1*y4 + x1*x4 + f**2, -f*(x1 + y1), 0, 0, 0, 0, 0, 0, -y1*y4 - f**2 + x1*y4, x4*y1 - x1*x4 - f**2, f*(x4+y4)], 
        [0, 0, 0, y2*y3 + f**2 - x3*y2, -x2*y3 + x2*x3 + f**2, -f*(x2 + y2), -y2*y3 - f**2 + x2*y3, x3*y2 - x2*x3 - f**2, f*(x3 + y3), 0, 0, 0], 
        [0, 0, 0, -2*x4*y2, 2*(x2*x4 + f**2), -2*f*y2, 0, 0, 0, 2*x2*y4, -2*(x2*x4 + f**2), 2*f*y4], 
        [0, 0, 0, 0, 0, 0, - y3*y4 - f**2 - x4*y3, x3*y4 + x3*x4 + f**2, f*(x3 - y3), y3*y4 + f**2 + x3*y4, -x4*y3 - x3*x4 - f**2, f*(-x4 + y4)]])

def matrice_A_cercle(n, a, r, b):
    A = np.zeros((6*n, 12))
    for j in range(n):
        A[6*j:6*(j+1), :12] = matrice_Aj(coord_points_cercle(n, a, r, b, j), b)
    return A

def conditionnement_cercle(n, a, r, b):
    return sl.svd(matrice_A_cercle(n, a, r, b))[1]

N, A, B, C = [], [], [], []
f = 1

for i in range(60):
    n = 50 * (i+1)
    c = conditionnement_cercle(n, 800000, 10000, 100000)
    N.append(n), A.append(c[0]), B.append(c[-1]), C.append(c[0]/c[-1])
    print(n)

plt.figure(1)
plt.clf()

plt.subplot(311)
plt.plot(N, B, marker='+', label="plus petite valeur propre")
plt.legend(loc='best')

plt.subplot(312)
plt.plot(N, A, marker='+', label="plus grande valeur propre")
plt.legend(loc='best')

plt.subplot(313)
plt.plot(N, C, marker='+', label="conditionnement")
plt.legend(loc='best')