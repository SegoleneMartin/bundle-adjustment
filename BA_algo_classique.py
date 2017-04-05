## bundle adjusment classique

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
import scipy.linalg as sl
from numpy import cos, sin
from numpy.linalg import det
from scipy.linalg import solve, rq


    

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

def matrice_R(theta):
    return np.dot(RZ(theta[2]), np.dot(RY(theta[1]), RX(theta[0])))

def matrice_P(C, theta, f):
    R = matrice_R(theta)
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    P = np.zeros((3, 4))
    P[:, :3] = np.dot(K, R)
    P[:, 3] = np.dot(K, R).dot(-C)
    return P
    
def parametres(P):
    ''' P matrice 3x4 associée à une caméra, retourne les paramètres de la caméra '''
    
    X = det([P[:, 1], P[:, 2], P[:, 3]])    
    Y = -det([P[:, 0], P[:, 2], P[:, 3]])    
    Z = det([P[:, 0], P[:, 1], P[:, 3]])    
    T = -det([P[:, 0], P[:, 1], P[:, 2]])
    
    C = np.array([X/T, Y/T, Z/T])
    
    [K, R] = rq(P[:, :-1])
    a = K[2, 2]
    K /= a
    R *= a
    K[0,0] *= np.sign(K[0,0])
    R[0, :] *= np.sign(K[0,0])
    K[1,1] *= np.sign(K[1,1])
    R[1, :] *= np.sign(K[1,1])
    return C, R, K
    

    
## simulations

# on suppose les points de correspondances situés dans une zone [0, X0] x [0, Y0] x [0, Z0]
# on suppose les caméras situées dans une zone [0, X0] x [0, Y0] x [Z1, Z2] avec Z0 << Z1
# on suppose les angles des caméras variant dans ???

#np.random.seed(0)

K = 6
N = 4
f = 50000
X0, Y0, Z0, Z1, Z2 = 100000, 100000, 5000, 700000, 800000

sigma_x = 0.1
sigma_theta = 0.00001

# paramètre fixé: 1 pour parametre fixés
theta_fixes = np.zeros((K, 3))
x_fixes = np.zeros((N, K, 2))


# génération aléatoire des paramètres initiaux
C_reel = rand(K, 3)
C_reel[:, 0] *= X0
C_reel[:, 1] *= Y0
C_reel[:, 2] = C_reel[:, 2] * (Z2-Z1) + Z1

theta_reel = rand(K, 3)
theta_reel[:, 0] = (theta_reel[:, 0] - 0.5) * np.pi/50 + np.pi
theta_reel[:, 1] = 0
theta_reel[:, 2] = theta_reel[:, 2] * 2*np.pi

X_reel = rand(N, 3)
X_reel[:, 0] *= X0
X_reel[:, 1] *= Y0
X_reel[:, 2] *= Z0



matrices_P = []
for k in range(K):
    matrices_P.append(matrice_P(C_reel[k], theta_reel[k], f))

x_reel = np.zeros((N, K, 2))
for j in range(N):
    for i in range(K):
        point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
        x_reel[j, i] = point[:-1] / point[-1]
        

x_0_initiaux = np.copy(x_reel) + normal(0, sigma_x, (N, K, 2))


# ETAPE 1: isotropic scaling
x_0=np.copy(x_0_initiaux)
translations = np.average(x_0, axis=0)
x_0 = x_0 - translations
x_0_norme=np.zeros((N,K,1))
for j in range(N):
    x_0_norme[j]=np.linalg.norm(x_0[j], axis=1).reshape(6,1)
distance_moyenne=np.average(x_0_norme, axis=0)
x_0=x_0*(np.sqrt(2))/distance_moyenne


# ETAPE 2: initial estimate of the projectiv depth
"""on prend une depth de 1 pour tous les points image. Le x_0 actuel est en coordonnées non homogènes: on doit donc rajouter des colonnes de 1"""
x_0_homogene=np.zeros((N, K, 3))
for j in range(N):
    x_0_homogene[j]=np.concatenate((x_0[j], np.ones((K,1))), axis=1)

# itérations
for iteration in range(1):
    
    W = np.zeros((3*K, N))
    for j in range(K):
        W[3*j:3*j+3, :] = x_0_homogene[:, j, :].T

    # ETAPE 3: normalisation des lambdas
    for passes in range(3):
        W_norme_colonne=np.linalg.norm(W, axis=0)
        W=W/W_norme_colonne[None, :]
        for i in range(K):
            W[3*i:3*i+3,:]/=np.linalg.norm(W[3*i:3*i+3,:])

    # ETAPE 4: factorisation
    u, s, v = sl.svd(W)
    
    M = u[:, :4]
    M[:, 0] *= s[0]
    M[:, 1] *= s[1]
    M[:, 2] *= s[2]
    M[:, 3] *= s[3]

    X = np.transpose(v[:, :4])
    
    x=np.dot(M,X)
    
    """On crée une nouvelle matrice x_0_homogene pour la prochaine itération"""
    for j in range(N):
        x_0_homogene[j,:, :2]=x_0[j]
        for i in range(K):
            x_0_homogene[j,i, 2]=x[2+3*i,j]
    


# Utilisation du résultat
x_a=np.zeros((N,K,2))
for j in range(N):
    for i in range(K):
        x_a[j,i,:]=x[3*i:3*i+2, j]/x[3*i+2,j]
        
x_a=x_a*distance_moyenne/np.sqrt(2)
x_a+=translations


P = np.zeros((K, 3, 4))
for j in range(K):
    P[j] = M[3*j:3*j+3, :]

C, R, K = parametres(P[0])


    
n_0=sl.norm(x_reel-x_0_initiaux)
n_a=sl.norm(x_reel-x_a)
n_i=sl.norm(x_0_initiaux-x_a)
  
        
print("n_0 = ", n_0)
print("n_a = ", n_a)
print("n_i = ", n_i)


