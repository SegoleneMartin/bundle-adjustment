# Resolution du problème de minimisation en utilisant le solveur de python avec la méthode Powell

import os
os.chdir("C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes")

## bundle adjusment de la manière la plus basique possible: on fait newton directement sans passer par les determinants

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
import scipy.linalg as sl
from BA_fonctions import *
import scipy.optimize
import time

## simulations

# on suppose les points de correspondances situés dans une zone [0, X0] x [0, Y0] x [0, Z0]
# on suppose les caméras situées dans une zone [0, X0] x [0, Y0] x [Z1, Z2] avec Z0 << Z1
# on suppose les angles des caméras variant dans ???

#np.random.seed(0)

K = 6
N = 10
f = 10**6
X0, Y0, Z0, Z1, Z2 = 20000, 20000, 5000, 700000, 800000

sigma_x = 0.1
sigma_theta = 0.00001


# génération aléatoire des paramètres initiaux
C_reel = rand(K, 3)
C_reel[:, 0] = (C_reel[:, 0] - 0.5) *2*X0
C_reel[:, 1] *= (C_reel[:, 1] - 0.5) *2*Y0
C_reel[:, 2] = C_reel[:, 2] * (Z2-Z1) + Z1

theta_reel = np.zeros((K, 3))
for i in range(K): # on fait pointer les caméras vers le point (0, 0, 0)
    X, Y, Z = -C_reel[i] / sl.norm(C_reel[i])
    beta = np.arcsin(-X)
    alpha = np.arcsin(Y/np.cos(beta))
    if np.cos(alpha)*np.cos(beta)*Z < 0: alpha = np.pi - alpha
    theta_reel[i, 0] = alpha
    theta_reel[i, 1] = beta

X_reel = (rand(3, N) - 0.5)   # X_reel est un vecteur 3xN
X_reel[0, :] *= X0
X_reel[1, :] *= Y0
X_reel[2, :] *= Z0

matrices_P_reel = []
for k in range(K):
    matrices_P_reel.append(matrice_P(C_reel [k], theta_reel[k], f))

x_reel = np.zeros((N, K, 2))
for j in range(N):
    for i in range(K):
        point = matrices_P_reel[i].dot(np.array([X_reel[0,j], X_reel[1,j], X_reel[ 2,j], 1]))
        x_reel[j, i] = point[:-1] / point[-1]

x_0 = np.copy(x_reel) + normal(0, sigma_x, (N, K, 2))
theta_0 = np.copy(theta_reel) + normal(0, sigma_theta, (K, 3))
#theta_0=theta_reel
#x_0=x_reel

matrices_P_0=[]   # matrices de projections crées à partir des angles theta_0
for k in range(K):
    matrices_P_0.append(matrice_P(C_reel[k], theta_0[k], f))

X_0=triangulation(matrices_P_0, x_0)   # reprojection: obtention des points X_0 de l'espace à partir des x_0 et theta_0
X_0=X_0[:3,:]



#Fonction à minimiser

def S(var,*x_0): 

    """
    var = [theta vecteur K*3, X vecteur 3*N] (pour utiliser le solveur on a besoin que les variables soient rentrées sous forme de vecteur)
    x_0 = valeur initiale passée en argument optionnel
    """

    theta = var[:3*K].reshape(K,3)
    X = var[3*K:].reshape(3,N)
    
    matrices_P = []
    for k in range(K):
        matrices_P.append(matrice_P(C_reel[k], theta[k], f))
    x = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P[i].dot(np.array([X[0, j], X[1,j], X[2,j], 1]))
            x[j, i] = point[:-1] / point[-1]
            
    return np.sum((x-x_0)**2)

### MINIMISATION DE S PAR LE SOLVEUR PYTHON

var_initiale = np.append(theta_0.reshape(3*K), X_0.reshape(3*N))

solution = scipy.optimize.minimize(S, var_initiale, args=x_0, method='Powell' ).x
theta_a = solution[:3*K].reshape(K,3)
X_a = solution[3*K:].reshape(3,N)

matrices_P_a=[]   # matrices de projections construite à partir de la solution theta_a
for k in range(K):
    matrices_P_a.append(matrice_P(C_reel[k], theta_a[k], f))
    
x_a = np.zeros((N, K, 2))  # on projète pour obtenir x_a
for j in range(N):
    for i in range(K):
        point = matrices_P_a[i].dot(np.array([X_a[0,j], X_a[1,j], X_a[ 2,j], 1]))
        x_a[j, i] = point[:-1] / point[-1]

n_0 = [sl.norm(x_reel-x_0), sl.norm(theta_reel-theta_0)]
n_a = [sl.norm(x_reel-x_a), sl.norm(theta_reel-theta_a)]
n_i = np.array([sl.norm(x_0 -x_a), sl.norm(theta_0-theta_a)])
print('n_0 =', n_0)
print('n_a =', n_a)
# print('n_i =', n_i)

n_0 = np.array([np.average(sl.norm((x_reel-x_0), axis=2)), np.average(sl.norm(theta_reel-theta_0, axis=1))])
n_a = (np.array([np.average(sl.norm((x_reel-x_a), axis=2)), np.average(sl.norm(theta_reel-theta_a, axis=1))]))
print('n_0 =', n_0)
print('n_a =', n_a)





### Comparaison des deux algos (le solveur: sans determinant, le notre: avec determinant) en faisant varier N et comparaison des temps d'execution

sigma_x = 0.1
sigma_theta = 0.00001

lamb_theta=0.02
lamb_x=0.02
 


def comparaison_solveur():
    
    X0, Y0, Z0, Z1, Z2 = 20000, 20000, 5000, 700000, 800000
    
    C_reel = rand(K, 3)
    C_reel[:, 0] = (C_reel[:, 0] - 0.5) *2*X0
    C_reel[:, 1] *= (C_reel[:, 1] - 0.5) *2*Y0
    C_reel[:, 2] = C_reel[:, 2] * (Z2-Z1) + Z1
    
    theta_reel = np.zeros((K, 3))
    for i in range(K): # on fait pointer les caméras vers le point (0, 0, 0)
        X, Y, Z = -C_reel[i] / sl.norm(C_reel[i])
        beta = np.arcsin(-X)
        alpha = np.arcsin(Y/np.cos(beta))
        if np.cos(alpha)*np.cos(beta)*Z < 0: alpha = np.pi - alpha
        theta_reel[i, 0] = alpha
        theta_reel[i, 1] = beta
    
    X_reel = (rand(N, 3) - 0.5)
    X_reel[:, 0] *= X0
    X_reel[:, 1] *= Y0
    X_reel[:, 2] *= Z0
    
    matrices_P = []
    for i in range(K):
        matrices_P.append(matrice_P(C_reel[i], theta_reel[i], f))
    
    x_reel = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
            x_reel[j, i] = point[:-1] / point[-1]
            
    x_0 = x_reel + normal(0, sigma_x, (N, K, 2))
    theta_0 = theta_reel + normal(0, sigma_theta, (K, 3))
    
    matrices_P_0=[]
    for k in range(K):
        matrices_P_0.append(matrice_P(C_reel[k], theta_0[k], f))
    
    X_0 = triangulation(matrices_P_0, x_0)
    X_0 = X_0[:3,:]
    
    
    # METHODE 1: sans determinant
    
    def S(var): 
    
        """
        var = [theta vecteur K*3, X vecteur 3*N] (pour utiliser le solveur on a besoin que les variables soient rentrées sous forme de vecteur)
        x_0 = valeur initiale passée en argument optionnel
        """
    
        theta = var[:3*K].reshape(K,3)
        X = var[3*K:].reshape(3,N)
        
        matrices_P = []
        for k in range(K):
            matrices_P.append(matrice_P(C_reel[k], theta[k], f))
        x = np.zeros((N, K, 2))
        for j in range(N):
            for i in range(K):
                point = matrices_P[i].dot(np.array([X[0, j], X[1,j], X[2,j], 1]))
                x[j, i] = point[:-1] / point[-1]
                
        return np.sum((x-x_0)**2)
    
    var_initiale = np.append(theta_0.reshape(3*K), X_0.reshape(3*N))
    
    tmp1 = time.clock()

    sol = scipy.optimize.minimize(S, var_initiale, method='Powell' )
    solution = sol.x
    theta_a1 = solution[:3*K].reshape(K,3)
    X_a1 = solution[3*K:].reshape(3,N)

    matrices_P_a1=[]
    for k in range(K):
        matrices_P_a1.append(matrice_P(C_reel[k], theta_a1[k], f))
    x_a1 = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P_a1[i].dot(np.array([X_a1[0,j], X_a1[1,j], X_a1[ 2,j], 1]))
            x_a1[j, i] = point[:-1] / point[-1]
    print((np.array([np.average(sl.norm((x_reel-x_a1), axis=2)), np.average(sl.norm(theta_reel-theta_a1, axis=1))])))
            
    tmp2 = time.clock()
    temps_a1 = tmp2-tmp1
    
    # METHODE 2: avec determinant
    
    tmp1 = time.clock()
    sig = np.zeros(3*K+2*K*N)
    sig[0:3*K] = sigma_theta
    sig[3*K:] = sigma_x
    
    lamb = np.zeros(3*K+2*K*N)
    lamb[0:3*K] = lamb_theta
    lamb[3*K:] = lamb_x
    
    # matrices A, B et M
    A, B = matrices_A_B(C_reel, theta_0, x_0, f)
    M = np.concatenate((A, B), axis=1)
    
    # reduction des vecteurs b (tq MX=b), sig et lamb compte tenu des paramètres fixés
    b = np.zeros((K*(K-1)/2*N+3*K+2*N*K))
    b[0:K*(K-1)/2*N] = -F(C_reel, theta_0, x_0, f)
    
    # ajout du terme forcant sur la matrice M
    contrainte = np.eye(3*K+2*N*K)/sig*lamb
    M = np.append(M,contrainte, axis=0)
    
    
    # Recherche de la solution qui minimise
    piM = np.linalg.pinv(M)
    erreur = piM.dot(b)
    
    theta_a2 = theta_0 + erreur[:3*K].reshape((K, 3))
    x_a2 = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            x_a2[j, i, 0] = x_0[j, i, 0] + erreur[3*K + 2*K*j + 2*i]
            x_a2[j, i, 1] = x_0[j, i, 1] + erreur[3*K + 2*K*j + 2*i + 1]
    tmp2 = time.clock()
    temps_a2 = tmp2-tmp1
    
    return C_reel, X_reel, theta_reel, theta_0, theta_a1, theta_a2, x_reel, x_0, x_a1, x_a2, temps_a1, temps_a2
    


###  AFFICHAGE

K = 6
cas = "satellite"
liste_N = np.arange(10, 80, 5)
Nexp = 10

n0_x_max, n0_x_min, n0_x_moy = [], [], []
na1_x_max, na1_x_min, na1_x_moy = [], [], []
na2_x_max, na2_x_min, na2_x_moy = [], [], []
n0_theta_max, n0_theta_min, n0_theta_moy = [], [], []
na1_theta_max, na1_theta_min, na1_theta_moy = [], [], []
na2_theta_max, na2_theta_min, na2_theta_moy = [], [], []

for N in liste_N:
    print(N)
    
    n0_x_max.append(0), n0_x_min.append(0), n0_x_moy.append(0)
    na1_x_max.append(0), na1_x_min.append(0), na1_x_moy.append(0)
    na2_x_max.append(0), na2_x_min.append(0), na2_x_moy.append(0)
    n0_theta_max.append(0), n0_theta_min.append(0), n0_theta_moy.append(0)
    na1_theta_max.append(0), na1_theta_min.append(0), na1_theta_moy.append(0)
    na2_theta_max.append(0), na2_theta_min.append(0), na2_theta_moy.append(0)
    
    for exp in range(Nexp):
        print(exp)
        C_reel, X_reel, theta_reel, theta_0, theta_a1, theta_a2, x_reel, x_0, x_a1, x_a2, temps_a1, temps_a2 = comparaison_solveur()
        
        n_0_x, n_a1_x, n_a2_x = sl.norm(x_reel-x_0, axis=2), sl.norm(x_reel-x_a1, axis=2), sl.norm(x_reel-x_a2, axis=2)
        n_0_theta, n_a1_theta, n_a2_theta = sl.norm(theta_reel-theta_0, axis=1), sl.norm(theta_reel-theta_a1, axis=1), sl.norm(theta_reel-theta_a2, axis=1)
        
        n0_x_max[-1] += np.max(n_0_x)
        n0_x_min[-1] += np.min(n_0_x)
        n0_x_moy[-1] += np.average(n_0_x)
        na1_x_max[-1] += np.max(n_a1_x)
        na1_x_min[-1] += np.min(n_a1_x)
        na1_x_moy[-1] += np.average(n_a1_x)
        na2_x_max[-1] += np.max(n_a2_x)
        na2_x_min[-1] += np.min(n_a2_x)
        na2_x_moy[-1] += np.average(n_a2_x)
        
        n0_theta_max[-1] += np.max(n_0_theta)
        n0_theta_min[-1] += np.min(n_0_theta)
        n0_theta_moy[-1] += np.average(n_0_theta)
        na1_theta_max[-1] += np.max(n_a1_theta)
        na1_theta_min[-1] += np.min(n_a1_theta)
        na1_theta_moy[-1] += np.average(n_a1_theta)
        na2_theta_max[-1] += np.max(n_a2_theta)
        na2_theta_min[-1] += np.min(n_a2_theta)
        na2_theta_moy[-1] += np.average(n_a2_theta)
    
    
n0_x_max = np.asarray(n0_x_max) / Nexp
n0_x_min = np.asarray(n0_x_min) / Nexp
n0_x_moy = np.asarray(n0_x_moy) / Nexp
na1_x_max = np.asarray(na1_x_max) / Nexp
na1_x_min = np.asarray(na1_x_min) / Nexp
na1_x_moy = np.asarray(na1_x_moy) / Nexp
na2_x_max = np.asarray(na2_x_max) / Nexp
na2_x_min = np.asarray(na2_x_min) / Nexp
na2_x_moy = np.asarray(na2_x_moy) / Nexp

n0_theta_max = np.asarray(n0_theta_max) / Nexp
n0_theta_min = np.asarray(n0_theta_min) / Nexp
n0_theta_moy = np.asarray(n0_theta_moy) / Nexp
na1_theta_max = np.asarray(na1_theta_max) / Nexp
na1_theta_min = np.asarray(na1_theta_min) / Nexp
na1_theta_moy = np.asarray(na1_theta_moy) / Nexp
na2_theta_max = np.asarray(na2_theta_max) / Nexp
na2_theta_min = np.asarray(na2_theta_min) / Nexp
na2_theta_moy = np.asarray(na2_theta_moy) / Nexp

###

figure2, (fig2, fig1) = plt.subplots(2, 1, sharex=True)
fig1.semilogy(liste_N, n0_x_moy, color="b", marker="s", markersize=5, linestyle="None", label="x_0 - x_reel")
fig1.semilogy(liste_N, n0_x_min, color="b", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_N, n0_x_max, color="b", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_N, na1_x_moy, color="g", marker="s", markersize=5, linestyle="None", label="x_a1 - x_reel (solver)")
fig1.semilogy(liste_N, na1_x_min, color="g", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_N, na1_x_max, color="g", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_N, na2_x_moy, color="r", marker="s", markersize=5, linestyle="None", label="x_a2 - x_reel (déterminants)")
fig1.semilogy(liste_N, na2_x_min, color="r", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_N, na2_x_max, color="r", marker="_", markersize=5, linestyle="None")
fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig1.set_ylabel("selon x")
fig1.set_xlabel("N")


fig2.set_title("{} simulations par N avec K = {}, sigma_x = {} (cas du {})".format(Nexp, K, sigma_x, cas))
fig2.semilogy(liste_N, n0_theta_moy, color="b", marker="s", markersize=5, linestyle="None", label="theta_0 - theta_reel")
fig2.semilogy(liste_N, n0_theta_min, color="b", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_N, n0_theta_max, color="b", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_N, na1_theta_moy, color="g", marker="s", markersize=5, linestyle="None", label="theta_a1 - theta_reel (solveur)")
fig2.semilogy(liste_N, na1_theta_min, color="g", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_N, na1_theta_max, color="g", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_N, na2_theta_moy, color="r", marker="s", markersize=5, linestyle="None", label="theta_a2 - theta_reel (déterminants)")
fig2.semilogy(liste_N, na2_theta_min, color="r", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_N, na2_theta_max, color="r", marker="_", markersize=5, linestyle="None")
fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig2.set_ylabel("selon theta")
fig2.set_xlim(min(liste_N)-1, max(liste_N)+1)


plt.show()


