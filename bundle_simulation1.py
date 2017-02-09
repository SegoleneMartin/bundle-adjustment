## BUNDLE ADJUSTMENT - 2016

import numpy as np
import pylab as plt
import scipy.linalg as sl

plt.show()

## Programme Python qui renvoie l'évolution de la plus petite valeur propre, de la plus grande valeur propre, et du conditionnement de la matrice A^T * A en fonction du nombre de points en correspondance DANS LE CAS DU CERCLE

# n : nombre de points (sans unité)
# f : distance focale (mètres)
# a : distance entre le plan des caméras et le plan des points (mètres)
# r : rayon du cercle formé par les points (mètres)
# b : distance entre les quatre caméras (mètres)

def coord_points_cercle(n, a, r, b, j):
    x, y = f*r * np.cos(2 * np.pi / n * j), f*r * np.sin(2 * np.pi / n * j)
    return np.array([x+f*b, x, x-f*b, x, y, y+f*b, y, y-f*b]) / a

def matrice_Aj(coord, b):
    x1, x2, x3, x4 = coord[0], coord[1],  coord[2], coord[3]
    y1, y2, y3, y4 = coord[4], coord[5], coord[6], coord[7]
    Aj = np.zeros((6, 12))
    Aj[0, 0:3] = y1*y2+f**2+x2*y1, -x1*y2-x1*x2-f**2, f*(-x1+y1)
    Aj[0, 3:6] = -y1*y2-f**2-x1*y2, x2*y1+x1*x2+f**2, f*(x2-y2)
    Aj[1, 0:3] = 2*(y1*y3+f**2), -2*x1*y3, -2*f*x1
    Aj[1, 6:9] = -2*(y1*y3+f**2), 2*x3*y1, 2*f*x3
    Aj[2, 0:3] = y1*y4+f**2-x4*y1, -x1*y4+x1*x4+f**2, -f*(x1+y1)
    Aj[2, 9:12] = -y1*y4-f**2+x1*y4, x4*y1-x1*x4-f**2, f*(x4+y4)
    Aj[3, 3:6] = y2*y3+f**2-x3*y2, -x2*y3+x2*x3+f**2, -f*(x2+y2)
    Aj[3, 6:9] = -y2*y3-f**2+x2*y3, x3*y2-x2*x3-f**2, f*(x3+y3)
    Aj[4, 3:6] = -2*x4*y2, 2*(x2*x4+f**2), -2*f*y2
    Aj[4, 9:12] = 2*x2*y4, -2*(x2*x4+f**2), 2*f*y4
    Aj[5, 6:9] = -y3*y4-f**2-x4*y3, x3*y4+x3*x4+f**2, f*(x3-y3)
    Aj[5, 9:12] = y3*y4+f**2+x3*y4, -x4*y3-x3*x4-f**2, f*(-x4+y4)
    return b / f**2 * Aj

def matrice_A_cercle(n, a, r, b):
    A = np.zeros((6*n, 12))
    for j in range(n):
        A[6*j:6*(j+1), :12] = matrice_Aj(coord_points_cercle(n, a, r, b, j), b)
    return A

def conditionnement_cercle(n, a, r, b):
    return sl.svd(matrice_A_cercle(n, a, r, b))[1]
    
f=1
"""
N, A, B, C = [], [], [], []


for i in range(60):
    n = 50 * (i+1)
    c = conditionnement_cercle(n, 800000, 10000, 100000)
    N.append(n), A.append(c[0]), B.append(c[-1]), C.append(c[0]/c[-1])
    print(n)

F, (fig1, fig2, fig3) = plt.subplots(3, sharex=True)

fig1.plot(N, B, marker='+')
fig1.set_ylabel('VP MINIMALE')

fig2.plot(N, A, marker='+')
fig2.set_ylabel('VP MAXIMALE')

fig3.plot(N, C, marker='+')
fig3.set_ylabel('CONDITIONNEMENT')
fig3.set_xlabel('n')


"""


## Programme Python qui compare les plus petites valeurs propres, les plus grandes valeurs propres, et les conditionnements des matrices A associés respectivement aux problèmes lorsque la dispostion est SUR UN CERCLE ou SUR UNE GRILLE en fonction du nombres de points en correspondance

# n^2 : nombre de points (sans unité)
# f : distance focale (mètres)
# a : distance entre le plan des caméras et plan des points (mètres)
# r : moitié de la largeur de la grille (mètres)
# b : distance entre les deux caméras (mètres)

def coord_points_grille(n, a, r, b, jx, jy):
    x = r*(-1 + 2*jx/(n-1))
    y = r*(-1 + 2*jy/(n-1))
    return np.array([x+b, x, x-b, x, y, y+b, y, y-b]) / a

def matrice_A_grille(n, a, r, b):
    A = np.zeros((6*n*n, 12))
    for jx in range(n):
        for jy in range(n):
            A[6*(n*jx+jy):6*((n*jx+jy)+1), :12] = matrice_Aj(coord_points_grille(n, a, r, b, jx, jy), b)
    return A
    
def conditionnement_grille(n, a, r, b):
    return sl.svd(matrice_A_grille(n, a, r, b))[1]

N2, A1, B1, C1, A2, B2, C2 = [], [], [], [], [], [], []

"""
for i in range(25):
    n = 5 + 2*i
    N2.append(n**2)
    print(n**2)
    c = conditionnement_cercle(n**2, 800000, 10000, 100000)
    A1.append(c[0])
    B1.append(c[-1])
    C1.append(c[0]/c[-1])
    c = conditionnement_grille(n, 800000, 10000, 100000)
    A2.append(c[0])
    B2.append(c[-1])
    C2.append(c[0]/c[-1])

F, (fig1, fig2, fig3) = plt.subplots(3, sharex=True)

fig1.plot(N2, B1, marker='+', color='r')
fig1.plot(N2, B2, marker='+', color='g')
fig1.set_ylabel('VP MINIMALE')

fig2.plot(N2, A1, marker='+', color='r', label='cercle')
fig2.plot(N2, A2, marker='+', color='g', label='grille')
fig2.legend(loc=4)
fig2.set_ylabel('VP MAXIMALE')

fig3.plot(N2, C1, marker='+', color='r')
fig3.plot(N2, C2, marker='+', color='g')
fig3.set_ylabel('CONDITIONNEMENT')
fig3.set_xlabel('n')

plt.show()
"""

## Programme Python qui renvoie l'évolution de la plus petite somme des carrés des lignes M, de la plus grande somme des carrés des lignes de M en fonction du nombre de points en correspondances

def matrice_B(n, f, b):
    B_j = np.array([[1, 1, -1, -1, 0, 0, 0, 0],
                  [0, 2, 0, 0, 0, -2, 0, 0],
                  [-1, 1, 0, 0, 0, 0, 1, -1],
                  [0, 0, -1, 1, 1, -1, 0, 0],
                  [0, 0, -2, 0, 0, 0, 2, 0],
                  [0, 0, 0, 0, -1, -1, 1, 1]
                 ]) * b/f
    B = np.zeros((6*n, 8*n))
    for j in range(n):
        B[6*j:6*(j+1), 8*j:8*(j+1)] = B_j
    return B

def matrice_M(n, f, a, r, b):
    A = matrice_A_cercle(n, a, r, b)
    B = matrice_B(n, f, b)
    AT = np.transpose(A)
    M = -np.dot(np.linalg.inv(np.dot(AT, A)), np.dot(AT, B))
    return M

"""
N, Min, Max = [], [], []

Mk = np.zeros(12)

for i in range(140):
    n = 505 + 15*i
    N.append(n)
    M = matrice_M(n, 10, 800000, 10000, 100000)
    for k in range(12):
        Mk[k] = np.dot(np.transpose(M[:,k]), M[:,k])
    Min.append(min(Mk))
    Max.append(max(Mk))
    print(n)
"""
F, (fig1, fig2, fig3, fig4) = plt.subplots(4, sharex=True)

fig1.plot(N, Min, '+-', color='r')
fig1.set_ylabel('MINIMUM')

fig2.plot(N, Max, '+-', color='r')
fig2.set_ylabel('MAXIMUM')

fig3.plot(N[100:], Min[100:], '+-', color='r')
fig3.set_ylabel('MINIMUM')

fig4.plot(N[100:], Max[100:], '+-', color='r')
fig4.set_ylabel('MAXIMUM')


plt.show()




## Programme Python qui simule la correction des erreurs

from random import normalvariate
from scipy.optimize import minimize

def RX(t):
    return(np.array([[1, 0, 0], 
                     [0, np.cos(t), -np.sin(t)], 
                     [0, np.sin(t), np.cos(t)]]))
def RY(t):
    return(np.array([[np.cos(t), 0, np.sin(t)], 
                     [0, 1, 0], 
                     [-np.sin(t), 0, np.cos(t)]]))
def RZ(t):
    return(np.array([[np.cos(t), -np.sin(t), 0], 
                     [np.sin(t), np.cos(t), 0], 
                     [0, 0, 1]]))
    

def R(theta, i): #theta est un tableau 4x3
    return(np.dot(np.dot(RX(theta[i-1, 0]), RY(theta[i-1, 1])), 
    	RZ(theta[i-1, 2])))

def matrice_K(f):
    return(np.diag(np.array([f,f,1])))

def matrice_C(b):
    c1=np.array([[0,0,0,0][1,0,0,0][2,1,0,0][1,0,-1,0]])
    c2=np.array([[0,0,0,0][-1,0,0,0][0,1,0,0][1,2,1,0]])
    return(np.array([c1,c2])*b)



def det(x, theta, j, k, l, b):
    K=matrice_K(f)
    col1 = np.dot(R(-theta,1), np.dot(1/K, x[j,k]))
    col2 = np.dot(R(-theta,2), np.dot(1/K, x[j,l]))
    centre=matrice_C(b)
    col3 = [centre[0,l,k], centre[1, l, k], 0]
    M = [C1, C2, C3]
    return(sl.det(M))


def F(x, theta, b):
    n = len(x)
    Y = np.zeros([6*n])
    for j in range(n):
        Y[6*j: 6*(j+1)]=[det(x, theta, 1, 2, j, b), det(x, theta,1, 3, j, b),
                         det(x, theta,1, 4, j, b), det(x, theta, 2, 3, j, b),
                         det(x, theta, 2, 4, j, b), det(x, theta, 3, 4, j, b)]
    return(Y)

def objective_function(theta_optimized, theta_fixed, x, b):
    theta = np.hstack((theta_optimized, theta_fixed))
    return sl.norm(F(x, np.reshape(theta * 1e-6, (4, 3)), b)) ** 2

def main():
    # n : nombre de points 
    # f : distance focale
    # a : distance entre le milieu des caméra et le sol 
    # r : rayon du cercle formé par les points
    # b : distance entre les deux caméras
    # s : écart type de l'érreur 
    
    n = 1
    f = 10
    a = 800000
    r = 10000
    b = 100000
    s = 1e-5
    
    # mise en place
    
    X = np.zeros([n,4])
    for k in range(n):
        X[k, :]=[r*np.cos(2*np.pi*k/n), r*np.sin(2*np.pi*k/n), a, 1]
    
    P=[] 
    P.append([[1, 0, 0, b], [0, 1, 0, 0], [0, 0, 1, 0]])
    P.append([[1, 0, 0, 0], [0, 1, 0, b], [0, 0, 1, 0]])
    P.append([[1, 0, 0, -b], [0, 1, 0, 0], [0, 0, 1, 0]])
    P.append([[1, 0, 0, 0], [0, 1, 0, -b], [0, 0, 1, 0]])
    
    delta = np.zeros([n, 4, 2])
    theta = np.zeros([4, 3])
    
    K = matrice_K(f)
    
    x = np.zeros([n, 4, 3])  #ensemble des coordonnées des projections:n=nombre de sous matrices 1 matrice=1 projeté, lignes = caméras, colonnes = coordonnées homogenes dans le plan
    for j in range(n):
        for i in range(4):
            #delta[i, j] = [normalvariate(0, s), normalvariate(0, s)] 
            x[j, i] = np.dot(1/K, np.dot(P[i], X[j]))
            x[j, i] = x[j, i] / x[j, i, 2] 
            	+ [delta[j, i][0], delta[j, i][1], 0]  #on ajoute des perturbations

    Y = F(x, theta, b) 
    #print(Y)

    # Minimisation de F
    #theta0 = np.zeros(12)
    np.random.seed(0)
    J=[]
    Z=[]
    for j in range(1, 12):
        print(j)
        theta0 = (2*np.random.rand(j) - 1) * 50
        print('Initial value of F', objective_function(theta0, 
        	np.zeros(12-j), x, b))
        print('value of F in zero', objective_function(np.zeros(j), 
        	np.zeros(12-j), x, b))
        k = 0
        for minimization_method in ['Nelder-Mead', 'Powell', 'CG', 
        	'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']:
            k += 1
            print(minimization_method)
            y = minimize(objective_function, x0=theta0, 
            	method=minimization_method, args=(np.zeros(12-j), x, b), 
                options={'xtol': 1e-15, 'ftol': 1e-15}).x
            y = abs(np.array(y))
            y = np.reshape(y, (j))
            z = max(y)
            J.append(j+k/10)
            Z.append(z)
            print(z)
    plt.figure()
    plt.plot(J, Z, 'o')
    
