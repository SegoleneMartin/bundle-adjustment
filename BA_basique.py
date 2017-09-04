# BASIQUE POUR BARRIER 

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
C_reel[:, 0] *= X0
C_reel[:, 1] *= Y0
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

n_0 = [sl.norm(x_reel[:, :, 0]-x_0[:, :, 0]), sl.norm(x_reel[:, :, 1]-x_0[:, :, 1]), sl.norm(theta_reel-theta_0)]
n_a = [sl.norm(x_reel[:, :, 0]-x_a[:, :, 0]), sl.norm(x_reel[:, :, 1]-x_a[:, :, 1]), sl.norm(theta_reel-theta_a)]
n_i = np.array([sl.norm(x_0[:, :, 0]-x_a[:, :, 0]), sl.norm(x_0[:, :, 1]-x_a[:, :, 1]), sl.norm(theta_0-theta_a)])

print('n_0 =', n_0)
print('n_a =', n_a)
print('n_i =', n_i)



### Comparaison des deux algos (le solveur: sans determinant, le notre: avec determinant) en faisant varier N et comparaison des temps d'execution
K = 6
f = 50000
X0, Y0, Z0, Z1, Z2 = 100000, 100000, 5000, 700000, 800000
nb_experience=10

sigma_x = 0.1
sigma_theta = 0.00001

lamb_theta=0.02
lamb_x=0.02



liste_N=[]      # liste des N utilisés 
liste_n_0=[]    # solution initiale
liste_n_a1=[]   # a1 solution sans det
liste_n_a2=[]   # a2 solution sans det
liste_F_0= []   # norme de F(sol initiale)
liste_F_a1= []  # norme de F(a1 solution sans det)
liste_F_a2= []  # norme de F(a2 solution avec det
liste_temps_a1=[]
liste_temps_a2=[]

for N in range(10, 80, 5):
    
    print("N", N)
    liste_N.append(N)
    n_0 = np.zeros(2)
    n_a1 = np.zeros(2)
    n_a2 = np.zeros(2)
    F_0 = 0
    F_a1 = 0
    F_a2 = 0
    temps_a1 = 0
    temps_a2 = 0
    
    
    for experience in range(nb_experience):
        
        print("experience", experience)
        
        # génération aléatoire des paramètres initiaux: X_reel, x_reel, theta_reel, X_0, x_0, theta_0
        C_reel = rand(K, 3)
        C_reel[:, 0] *= X0
        C_reel[:, 1] *= Y0
        C_reel[:, 2] = C_reel[:, 2] * (Z2-Z1) + Z1
        
        theta_reel = np.zeros((K, 3))
        for i in range(K): # on fait pointer les caméras vers le point (0, 0, 0)
            X, Y, Z = -C_reel[i] / sl.norm(C_reel[i])
            beta = np.arcsin(-X)
            alpha = np.arcsin(Y/np.cos(beta))
            if np.cos(alpha)*np.cos(beta)*Z < 0: alpha = np.pi - alpha
            theta_reel[i, 0] = alpha
            theta_reel[i, 1] = beta
        
        X_reel = rand(3, N) - 0.5
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
        
        matrices_P_0=[]
        for k in range(K):
            matrices_P_0.append(matrice_P(C_reel[k], theta_0[k], f))
        
        X_0 = triangulation(matrices_P_0, x_0)
        X_0 = X_0[:3,:]
        
        
        # METHODE 1: sans determinant
        
        var_initiale = np.append(theta_0.reshape(3*K), X_0.reshape(3*N))
        
        tmp1 = time.clock()

        sol = scipy.optimize.minimize(S, var_initiale, args=x_0, method='Powell' )
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
                
        tmp2 = time.clock()
        
        F_0+ = sl.norm(F(C_reel, theta_0, x_0, f))
        F_a1+ = sl.norm(F(C_reel, theta_a1, x_a1, f))
        n_0 + = np.array([sl.norm(x_reel-x_0), sl.norm(theta_reel-theta_0)])
        n_a1+ = np.array([sl.norm(x_reel-x_a1), sl.norm(theta_reel-theta_a1)])
        temps_a1+ = tmp2-tmp1
        
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
        
        n_a2+ = np.array([sl.norm(x_reel-x_a2),  sl.norm(theta_reel-theta_a2)])
        F_a2+ = sl.norm(F(C_reel, theta_a2, x_a2, f))
        temps_a2+ = tmp2-tmp1
        
    liste_F_0.append(F_0/nb_experience)
    liste_F_a1.append(F_a1/nb_experience)
    liste_F_a2.append(F_a2/nb_experience)
    liste_n_0.append(n_0/nb_experience)
    liste_n_a1.append(n_a1/nb_experience)
    liste_n_a2.append(n_a2/nb_experience)
    liste_temps_a1.append(temps_a1/nb_experience)
    liste_temps_a2.append(temps_a2/nb_experience)

liste_n_0 = np.array(liste_n_0) 
liste_n_a1 = np.array(liste_n_a1)
liste_n_a2 = np.array(liste_n_a2)
    


###  AFFICHAGE
'''
#fig1.set_title('{} simulations satellitaires avec K = {}, N = {}, lamb_theta={}, lamb_x={}, sigma_theta={}, sigma_x={}, f={}'.format(indice+1, K, N, lamb_theta, lamb_x, sigma_theta, sigma_x, f))
 
plt.figure(1)
plt.semilogy(liste_N, liste_F_0,  color='b', marker='+', markersize=10, linestyle='None', label='|| F_0 ||')
plt.semilogy(liste_N, liste_F_a1,  color='r', marker='+', markersize=10, linestyle='None', label='|| F_a || (sans determinant)')
plt.semilogy(liste_N, liste_F_a2,  color='g', marker='+', markersize=10, linestyle='None', label='|| F_a || (avec determinant)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('F')
plt.xlim(min(liste_N)-1, max(liste_N)+1)
plt.xlabel('N')


plt.show()

plt.clf()
plt.figure(2)
    
plt.semilogy(liste_N, liste_n_0[:,0], color='b', marker='+', markersize=10, linestyle='None', label='|| X_0 - X_reel ||')
plt.semilogy(liste_N,liste_n_a1[:,0], color='r', marker='+', markersize=10, linestyle='None', label='|| X_a - X_reel || (sans determinant)')
plt.semilogy(liste_N,liste_n_a2[:,0], color='g', marker='+', markersize=10, linestyle='None', label='|| X_a - X_reel || (avec determinant)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('selon X')
plt.xlabel('N')
plt.xlim(min(liste_N)-1, max(liste_N)+1)

plt.show()

plt.clf()
plt.figure(3)

plt.semilogy(liste_N, liste_n_0[:,1], color='b', marker='+', markersize=10, linestyle='None', label='|| theta_0 - theta_reel ||')
plt.semilogy(liste_N,liste_n_a1[:,1],color='r', marker='+', markersize=10, linestyle='None', label='|| theta_a - theta_reel || (sans determinant)')
plt.semilogy(liste_N,liste_n_a2[:,1], color='g', marker='+', markersize=10, linestyle='None', label='|| theta_a - theta_reel || (avec determinant)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('selon theta')
plt.xlabel('N')
plt.xlim(min(liste_N)-1, max(liste_N)+1)

plt.show()

'''


figure, (fig4, fig2, fig1) = plt.subplots(3, 1, sharex=True)
fig4.set_title('simulations satellitaires avec K = {}, lamba={}, sigma_theta={}, sigma_x={}, nombre d\'experiences'.format(K, lamb_theta, sigma_theta, sigma_x, nb_experience))
 
fig4.semilogy(liste_N, liste_F_0,  color='b', marker='+', markersize=10, linestyle='None', label='|| F_0 ||')
fig4.semilogy(liste_N, liste_F_a1,  color='r', marker='+', markersize=10, linestyle='None', label='|| F_a || (sans determinant)')
fig4.semilogy(liste_N, liste_F_a2,  color='g', marker='+', markersize=10, linestyle='None', label='|| F_a || (avec determinant)')
fig4.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig4.set_ylabel('F')
fig4.set_xlim(min(liste_N)-1, max(liste_N)+1)
    
fig2.semilogy(liste_N, liste_n_0[:,0], color='b', marker='+', markersize=10, linestyle='None', label='|| X_0 - X_reel ||')
fig2.semilogy(liste_N,liste_n_a1[:,0], color='r', marker='+', markersize=10, linestyle='None', label='|| X_a - X_reel || (sans determinant)')
fig2.semilogy(liste_N,liste_n_a2[:,0], color='g', marker='+', markersize=10, linestyle='None', label='|| X_a - X_reel || (avec determinant)')
fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig2.set_ylabel('selon X')

fig1.semilogy(liste_N, liste_n_0[:,1], color='b', marker='+', markersize=10, linestyle='None', label='|| theta_0 - theta_reel ||')
fig1.semilogy(liste_N,liste_n_a1[:,1],color='r', marker='+', markersize=10, linestyle='None', label='|| theta_a - theta_reel || (sans determinant)')
fig1.semilogy(liste_N,liste_n_a2[:,1], color='g', marker='+', markersize=10, linestyle='None', label='|| theta_a - theta_reel || (avec determinant)')
fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig1.set_ylabel('selon theta')
fig1.set_xlabel('N')

plt.show()

plt.figure(1)
plt.semilogy(liste_N, liste_temps_a1,  color='r', marker='+', markersize=10, linestyle='None', label='temps 1 (sans determinant)')
plt.semilogy(liste_N, liste_temps_a2,  color='g', marker='+', markersize=10, linestyle='None', label='temps 2 (avec determinant)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('temps')
plt.xlim(min(liste_N)-1, max(liste_N)+1)
plt.xlabel('N')

plt.show()

