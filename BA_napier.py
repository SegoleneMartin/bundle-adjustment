## bundle adjusment avec terme forcant et possibilité de fixer les paramètres

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
import random
import scipy.linalg as sl
from numpy import cos, sin
from math import atan2
import os
os.chdir("C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes")
from decomposition_QR import *
from BA_fonctions import *
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from scipy.sparse import csr_matrix



## application à des données réelles: vérification croisée. 
#os.chdir("C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images")
os.chdir("C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images/matches/matches_ransac_0.2")
#os.chdir("C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images/matches")

np.random.seed(1)

K = 6
N = 280  #Nombre de points dans les sous ensembles à partir desquels on estime une solution
M = 1 # nombre de sous ensembles
f=1.398267e+06

sigma_x = 0.1
sigma_theta = 0.00001
sig=np.zeros(3*K+2*K*N)
sig[0:3*K]=sigma_theta
sig[3*K:]=sigma_x

lamb_theta=0.03
lamb_x=0.03
lamb=np.zeros(3*K+2*K*N)
lamb[0:3*K]=lamb_theta
lamb[3*K:]=lamb_x

# paramètre fixé: 1 pour parametre fixés
theta_fixes = np.zeros((K, 3))
x_fixes = np.zeros((N, K, 2))

# recupération des correspondances
#correspondances = np.loadtxt('correspondances_M6.txt')
#correspondances = np.loadtxt('Correspondances.txt')
correspondances = np.loadtxt('Correspondances_0.2.txt')

alea=np.arange(0, np.shape(correspondances)[0], 1)
alea=list(alea)
echantillon = random.sample(alea, N) 
x_0=correspondances[echantillon]
x_0=x_0.reshape((N, 6, 2), order='F')
x_0=np.array(x_0)
 
# récupération des angles
os.chdir("C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images")
P1, P2, P3, P4, P5, P6 = np.loadtxt('P_12.txt'), np.loadtxt('P_13.txt'), np.loadtxt('P_14.txt'), np.loadtxt('P_18.txt'), np.loadtxt('P_19.txt'), np.loadtxt('P_20.txt')
P=[P1, P2, P3, P4, P5, P6]
theta_0=[]
C_reel=[]
for i,Pi in enumerate(P):
    C, R, mat_K = parametres(Pi)
    
    px=mat_K[0,2]
    py=mat_K[1,2]
    x_0[:, i, 0]-=px
    x_0[:, i, 1]-=py
    x_0[:, i, 1]*=-1
    mat_K[1,1]*=-1
    mat_K[0,2], mat_K[1,2], mat_K[0,1] = 0, 0, 0
    
    a=atan2(R[2,1], R[2,2])
    b=atan2(-R[2,0], np.sqrt(R[2,1]**2 +R[2,2]**2))
    g=atan2(R[1,0],R[0,0])
    theta_0.append([a, b, g])
    C_reel.append(C)
    
theta_0=np.array(theta_0)
C_reel=np.array(C_reel)


# matrices A, B et M
A, B = matrices_A_B(C_reel, theta_0, x_0, f)

for k in range(K-1, -1, -1):
    for angle in range(2, -1, -1):
        if theta_fixes[k, angle] == 1:
            A = np.delete(A, (3*k+angle), axis=1)
for n in range(N-1, -1, -1):
    for k in range(K-1, -1, -1):
        for i in range(1, -1, -1):
            if x_fixes[n, k, i] == 1:
                B = np.delete(B, (2*K*n + 2*k + i), axis=1)

M = np.concatenate((A, B), axis=1)

# reduction des vecteurs b (tq MX=b), sig et lamb compte tenu des paramètres fixés
b=np.zeros((K*(K-1)/2*N+3*K+2*N*K))
b[0:K*(K-1)/2*N]=-F(C_reel, theta_0, x_0, f)
b1=b[0:K*(K-1)/2*N]
b2=b[K*(K-1)/2*N:K*(K-1)/2*N+3*K]
b3=b[K*(K-1)/2*N+3*K:]

count=0
sig1=sig[:3*K]
sig2=sig[3*K:]
lamb1=lamb[:3*K]
lamb2=lamb[3*K:]
for k in range(K-1, -1, -1):
    for angle in range(2, -1, -1):
        if theta_fixes[k, angle] == 1:
            b2=np.delete(b2, 3*k+angle)
            sig1=np.delete(sig1, 3*k+angle)
            lamb1=np.delete(lamb1, 3*k+angle)
            count+=1
for n in range(N-1, -1, -1):
    for k in range(K-1, -1, -1):
        for i in range(1, -1, -1):
            if x_fixes[n, k, i] == 1:
                b3=np.delete(b3, 2*K*n + 2*k + i)
                sig2=np.delete(sig2, 2*K*n + 2*k + i)
                lamb2=np.delete(lamb2,2*K*n + 2*k + i)
                count+=1

sig=np.concatenate((sig1, sig2), axis=0)
lamb=np.concatenate((lamb1, lamb2), axis=0)
b=np.concatenate((b1, np.concatenate((b2, b3), axis=0)), axis=0)

# ajout du terme forcant sur la matrice M
contrainte=np.eye(3*K+2*N*K-count)/sig*lamb
M=np.append(M,contrainte, axis=0)
M=csr_matrix(M)


# Recherche de la solution qui minimise

erreur1 = ssl.lsqr(M,b, conlim=1.0e+8)[0]
erreur = np.zeros((3*K + 2*N*K))
i = 0
for p in range(3*K):
    if theta_fixes[p//3, p%3] == 0:
        erreur[p] = erreur1[i]
        i += 1
for p in range(3*K, 3*K + 2*N*K):
    q = p - 3*K
    if x_fixes[q//(2*K), (q%(2*K))//2, q%2] == 0:
        erreur[p] = erreur1[i]
        i += 1

theta_a = theta_0 + erreur[:3*K].reshape((K, 3))
x_a = np.zeros((N, K, 2))
for j in range(N):
    for i in range(K):
        x_a[j, i, 0] = x_0[j, i, 0] + erreur[3*K + 2*K*j + 2*i]
        x_a[j, i, 1] = x_0[j, i, 1] + erreur[3*K + 2*K*j + 2*i + 1]


n_i = np.array([sl.norm(x_0[:, :, 0]-x_a[:, :, 0]), sl.norm(x_0[:, :, 1]-x_a[:, :, 1]), sl.norm(theta_0-theta_a)])
n_i_bis = np.array([sl.norm(x_0-x_a), sl.norm(theta_0-theta_a)])

print('norm F_0 =', sl.norm(F(C_reel, theta_0, x_0, f)))
print('norm F_a =', sl.norm(F(C_reel, theta_a, x_a, f)))
print('n_i =', n_i)
print('n_i_bis =', n_i_bis)














