################################################################################
################################################################################
##                   Bundle Adjustment With Known Positions                   ##
##                                                                            ##
##                       SIMULATIONS SANS TERME FORCANT                       ##
################################################################################
################################################################################

## packages

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
import scipy.linalg as sl
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as ssl
import os

plt.ion(), plt.show()

## import de BA_fonctions

dossier = "C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes"

os.chdir(dossier)
from BA_fonctions import *

################################################################################
##                           FONCTIONS DE SIMULATION                          ##
################################################################################

## rappels

# on suppose les points de correspondances situés dans la zone 
# [-X0/2, X0/2] x [-Y0/2, Y0/2] x [-Z0/2, Z0/2]

# on suppose les caméras situées dans la zone 
# [-X0, X0] x [-Y0, Y0] x [Z1, Z2] avec Z0 << Z1

# on suppose que les caméras pointent vers le point (0, 0, 0)
# pour cela, connaissant les coordonnées de la caméra, on fixe l'angle gamma à 0
# puis alpha et beta sont alors déterminés de manière unique

## fonctions

def genere_donnees():
    """ crée aléatoirement des données 
    
    SORTIES
    C_reel [array (K, 3)] : coordonnées des caméras
    X_reel [array (N, 3)] : coordonnées des points
    theta_reel [array (K, 3)] : angles de rotation des caméras initiaux
    
    """
    
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
    
    return C_reel, X_reel, theta_reel

def scene_3D():
    """ crée une figure 3D correspondant aux données """
    
    Z_cam = []
    for theta in theta_reel:
        Z_cam.append(np.linalg.inv(matrice_R(theta)).dot(np.array([0, 0, 1])))
    
    plt.figure().gca(projection = "3d")
    plt.xlabel("X"), plt.ylabel("Y")
    plt.plot([0], [0], [0], color="k", marker="+", markersize=10)
    plt.plot(X_reel[:, 0], X_reel[:, 1], X_reel[:, 2], marker="+", markersize=3, color="b", label="points", linestyle="None")
    plt.plot(C_reel[:, 0], C_reel[:, 1], C_reel[:, 2], marker="+", markersize=10, color="r", label="cameras", linestyle="None")
    plt.legend(loc="best")
    plt.xlim(-X0, X0), plt.ylim(-Y0, Y0)
    
    t = 200000
    for C, Z in zip(C_reel, Z_cam):
        ZZ = C + t*Z
        plt.plot([C[0], ZZ[0]], [C[1], ZZ[1]], [C[2], ZZ[2]], color="k")    # trace les axes Z_cam
        
        
        
        

def reestimation_globale(x_exact=False, theta_exact=False, affichage=False, AC=False):
    """ réalise une simulation globale (création des données + réestimation de tous les paramètres)
    
    PARAMETRES
    x_exact [bool] : si vrai, fixe x_0 à x_reel (pas de perturbation des points)
    theta_exact [bool] : si vrai, fixe theta_0 à theta_reel (pas de perturbation des angles)
    affichage [bool] : si vrai, affiche les données créées sur une figure 3D
    AC [bool] : si vrai, rajoute une contrainte 
    
    SORTIES
    C_reel [array (K, 3)] : coordonnées des caméras
    X_reel [array (N, 3)] : coordonnées des points
    theta_reel, theta_0, theta_a [array (K, 3)] : angles de rotation des caméras, respectivement initiaux, perturbés et réestimés
    x_reel, x_0, x_a [array (N, K, 2)] : coordonnées des images des points, respectivement initiaux, perturbés et réestimés
    
    """
    
    C_reel, X_reel, theta_reel = genere_donnees()
    
    if affichage: scene_3D()
    
    matrices_P = []
    for i in range(K):
        matrices_P.append(matrice_P(C_reel[i], theta_reel[i], f))
    
    x_reel = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
            x_reel[j, i] = point[:-1] / point[-1]
    
    # perturbation des coordonnées et des angles
    
    x_0 = np.copy(x_reel)
    if not x_exact: x_0 += normal(0, sigma_x, (N, K, 2))
    theta_0 = np.copy(theta_reel)
    if not theta_exact: theta_0 += normal(0, sigma_theta, (K, 3))
    
    # minimisation de F
    
    A, B = matrices_A_B_sparse(C_reel, theta_0, x_0, f)
    M = ss.hstack([A, B])
    
    if AC == True:
    
        sig = np.zeros(3*K+2*K*N)
        sig[0:3*K] = sigma_theta
        sig[3*K:] = sigma_x
        
        lamb = np.zeros(3*K+2*K*N) 
        lamb[0:3*K] = lamb_theta
        lamb[3*K:] = lamb_x
        
        b = np.zeros((K*(K-1)/2*N+3*K+2*N*K))
        b[0:K*(K-1)/2*N] = -F(C_reel, theta_0, x_0, f)
        contrainte = np.eye(3*K+2*N*K)/sig*lamb
        M = ss.vstack([M,contrainte])
        erreur = ssl.lsqr(M,b, conlim=1.0e+8)[0]
        
    else :
        erreur = ssl.lsqr(M, -F(C_reel, theta_0, x_0, f), conlim=1.0e+8)[0]
    
    theta_a = theta_0 + erreur[:3*K].reshape((K, 3))
    x_a = np.copy(x_0)
    for j in range(N):
        for i in range(K):
            x_a[j, i, 0] += erreur[3*K + 2*K*j + 2*i]
            x_a[j, i, 1] += erreur[3*K + 2*K*j + 2*i + 1]
    
    return C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a
    
    
    
    
    
def reestimation_globale_comparaison(x_exact=False, theta_exact=False, affichage=False): # pour comparer les resultats avec et sans contrainte
    """ réalise une simulation globale (création des données + réestimation de tous les paramètres)
    
    PARAMETRES
    x_exact [bool] : si vrai, fixe x_0 à x_reel (pas de perturbation des points)
    theta_exact [bool] : si vrai, fixe theta_0 à theta_reel (pas de perturbation des angles)
    affichage [bool] : si vrai, affiche les données créées sur une figure 3D
    
    SORTIES
    C_reel [array (K, 3)] : coordonnées des caméras
    X_reel [array (N, 3)] : coordonnées des points
    theta_reel, theta_0, theta_a [array (K, 3)] : angles de rotation des caméras, respectivement initiaux, perturbés et réestimés
    x_reel, x_0, x_a [array (N, K, 2)] : coordonnées des images des points, respectivement initiaux, perturbés et réestimés
    
    """
    
    C_reel, X_reel, theta_reel = genere_donnees()
    
    if affichage: scene_3D()
    
    matrices_P = []
    for i in range(K):
        matrices_P.append(matrice_P(C_reel[i], theta_reel[i], f))
    
    x_reel = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
            x_reel[j, i] = point[:-1] / point[-1]
    
    # perturbation des coordonnées et des angles
    
    x_0 = np.copy(x_reel)
    if not x_exact: x_0 += normal(0, sigma_x, (N, K, 2))
    theta_0 = np.copy(theta_reel)
    if not theta_exact: theta_0 += normal(0, sigma_theta, (K, 3))
    
    # minimisation de F
    
    A, B = matrices_A_B_sparse(C_reel, theta_0, x_0, f)
    M = ss.hstack([A, B])
    erreur1 = ssl.lsqr(M, -F(C_reel, theta_0, x_0, f), conlim=1.0e+8)[0]
    theta_a1 = theta_0 + erreur1[:3*K].reshape((K, 3))
    x_a1 = np.copy(x_0)
    for j in range(N):
        for i in range(K):
            x_a1[j, i, 0] += erreur1[3*K + 2*K*j + 2*i]
            x_a1[j, i, 1] += erreur1[3*K + 2*K*j + 2*i + 1]
    
    sig = np.zeros(3*K+2*K*N)
    sig[0:3*K] = sigma_theta
    sig[3*K:] = sigma_x
    
    lamb = np.zeros(3*K+2*K*N) 
    lamb[0:3*K] = lamb_theta
    lamb[3*K:] = lamb_x
    
    b = np.zeros((K*(K-1)/2*N+3*K+2*N*K))
    b[0:K*(K-1)/2*N] = -F(C_reel, theta_0, x_0, f)
    contrainte = np.eye(3*K+2*N*K)/sig*lamb
    M = ss.vstack([M,contrainte])
    erreur2 = ssl.lsqr(M,b, conlim=1.0e+8)[0]
    
    theta_a2 = theta_0 + erreur2[:3*K].reshape((K, 3))
    x_a2 = np.copy(x_0)
    for j in range(N):
        for i in range(K):
            x_a2[j, i, 0] += erreur2[3*K + 2*K*j + 2*i]
            x_a2[j, i, 1] += erreur2[3*K + 2*K*j + 2*i + 1]
    
    return C_reel, X_reel, theta_reel, theta_0, theta_a1, theta_a2, x_reel, x_0, x_a1, x_a2
    
    
    
    

def reestimation_x(theta_exact=False, affichage=False):
    """ réalise une simulation ne réestimant que les x (création des données + réestimation des x)
    
    PARAMETRES
    x_exact [bool] : si vrai, fixe x_0 à x_reel (pas de perturbation des points)
    theta_exact [bool] : si vrai, fixe theta_0 à theta_reel (pas de perturbation des angles)
    affichage [bool] : si vrai, affiche les données créées sur une figure 3D
    
    SORTIES
    C_reel [array (K, 3)] : coordonnées des caméras
    X_reel [array (N, 3)] : coordonnées des points
    theta_reel, theta_0, theta_a [array (K, 3)] : angles de rotation des caméras, respectivement initiaux, perturbés et réestimés
    x_reel, x_0, x_a [array (N, K, 2)] : coordonnées des images des points, respectivement initiaux, perturbés et réestimés
    
    """
    
    C_reel, X_reel, theta_reel = genere_donnees()
    
    if affichage: scene_3D()
    
    matrices_P = []
    for i in range(K):
        matrices_P.append(matrice_P(C_reel[i], theta_reel[i], f))
    
    x_reel = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
            x_reel[j, i] = point[:-1] / point[-1]
    
    # perturbation des coordonnées et des angles
    
    x_0 = x_reel + normal(0, sigma_x, (N, K, 2))
    theta_0 = np.copy(theta_reel)
    if not theta_exact: theta_0 += normal(0, sigma_theta, (K, 3))
    
    # minimisation de F
    
    B = csr_matrix(matrice_B(C_reel, theta_0, x_0, f))
    
    erreur = ssl.lsqr(B, -F(C_reel, theta_0, x_0, f), conlim=1.0e+8)[0]
    
    theta_a = np.copy(theta_0)
    x_a = np.copy(x_0)
    for j in range(N):
        for i in range(K):
            x_a[j, i, 0] += erreur[2*K*j + 2*i]
            x_a[j, i, 1] += erreur[2*K*j + 2*i + 1]
    
    return C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a




def reestimation_theta(x_exact=False, affichage=False):
    """ réalise une simulation ne réestimant que les angles (création des données + réestimation des angles)
    
    PARAMETRES
    x_exact [bool] : si vrai, fixe x_0 à x_reel (pas de perturbation des points)
    theta_exact [bool] : si vrai, fixe theta_0 à theta_reel (pas de perturbation des angles)
    affichage [bool] : si vrai, affiche les données créées sur une figure 3D
    
    SORTIES
    C_reel [array (K, 3)] : coordonnées des caméras
    X_reel [array (N, 3)] : coordonnées des points
    theta_reel, theta_0, theta_a [array (K, 3)] : angles de rotation des caméras, respectivement initiaux, perturbés et réestimés
    x_reel, x_0, x_a [array (N, K, 2)] : coordonnées des images des points, respectivement initiaux, perturbés et réestimés
    
    """
    
    C_reel, X_reel, theta_reel = genere_donnees()
    
    if affichage: scene_3D()
    
    matrices_P = []
    for i in range(K):
        matrices_P.append(matrice_P(C_reel[i], theta_reel[i], f))
    
    x_reel = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
            x_reel[j, i] = point[:-1] / point[-1]
    
    # perturbation des coordonnées et des angles
    
    x_0 = np.copy(x_reel)
    if not x_exact: x_0 += normal(0, sigma_x, (N, K, 2))
    theta_0 = theta_reel + normal(0, sigma_theta, (K, 3))
    
    # minimisation de F
    
    A = csr_matrix(matrice_A(C_reel, theta_0, x_0, f))
    
    erreur = ssl.lsqr(A, -F(C_reel, theta_0, x_0, f), conlim=1.0e+8)[0]
    
    theta_a = theta_0 + erreur.reshape((K, 3))
    x_a = np.copy(x_0)
    
    return C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a
    
    
    

def reestimation_parametres_choisis(x_fixes, theta_fixes, affichage=False):
    """ réalise une simulation qui ne réestime que les paramètres souhaités
    
    PARAMETRES
    x_fixes [array (N, K, 2)] : array de 0 et de 1. Les 1 correspondent aux coordonnées qu'il ne faut pas réestimer
    theta_fixes [array (K, 3)] : array de 0 et de 1. Les 1 correspondent aux angles qu'il ne faut pas réestimer
    affichage [bool] : si vrai, affiche les données créées sur une figure 3D
    
    SORTIES
    C_reel [array (K, 3)] : coordonnées des caméras
    X_reel [array (N, 3)] : coordonnées des points
    theta_reel, theta_0, theta_a [array (K, 3)] : angles de rotation des caméras, respectivement initiaux, perturbés et réestimés
    x_reel, x_0, x_a [array (N, K, 2)] : coordonnées des images des points, respectivement initiaux, perturbés et réestimés
    
    """
    
    C_reel, X_reel, theta_reel = genere_donnees()
    
    if affichage: scene_3D()
    
    matrices_P = []
    for i in range(K):
        matrices_P.append(matrice_P(C_reel[i], theta_reel[i], f))
    
    x_reel = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
            x_reel[j, i] = point[:-1] / point[-1]
    
    # perturbation des coordonnées et des angles
    
    x_0 = x_reel + normal(0, sigma_x, (N, K, 2))
    theta_0 = theta_reel + normal(0, sigma_theta, (K, 3))
    
    for n in range(N):
        for k in range(K):
            for i in range(2):
                if x_fixes[n, k, i] == 1:
                    x_0[n, k, i] = x_reel[n, k, i]
    for k in range(K):
        for angle in range(3):
            if theta_fixes[k, angle] == 1:
                theta_0[k, angle] = theta_reel[k, angle]
    
    # minimisation de F
    
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
    
    erreur1 = ssl.lsqr(M, -F(C_reel, theta_0, x_0, f), conlim=1.0e+8)[0]
    
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
    x_a = np.copy(x_0)
    for j in range(N):
        for i in range(K):
            x_a[j, i, 0] += erreur[3*K + 2*K*j + 2*i]
            x_a[j, i, 1] += erreur[3*K + 2*K*j + 2*i + 1]

    return C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a





################################################################################
##                            SIMULATIONS GLOBALES                            ##
################################################################################

## paramètres

K = 6
N = 500
Nexp = 15

cas = "satellite"
f = 10**6
X0, Y0, Z0, Z1, Z2 = 20000, 20000, 5000, 700000, 800000
sigma_x = 10**-2
sigma_theta = 10**-5

lamb_theta = 0.02
lamb_x = 0.02

# cas = "drône"
# f = 10**4
# X0, Y0, Z0, Z1, Z2 = 100, 100, 40, 50, 75
# sigma_x = 10**-2
# sigma_theta = 10**-2

## réestimation globale

liste_exp = np.arange(Nexp) + 1
F0_max, F0_min, F0_moy = [], [], []
Fa_max, Fa_min, Fa_moy = [], [], []
n0_theta_max, n0_theta_min, n0_theta_moy = [], [], []
n0_x_max, n0_x_min, n0_x_moy = [], [], []
na_theta_max, na_theta_min, na_theta_moy = [], [], []
na_x_max, na_x_min, na_x_moy = [], [], []

for n in range(Nexp):
    C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a = reestimation_globale(AC=True)
    
    F_0, F_a, n_0, n_a = abs(F(C_reel, theta_0, x_0, f)), abs(F(C_reel, theta_a, x_a, f)), (np.array([sl.norm((x_reel-x_0), axis=2), sl.norm(theta_reel-theta_0, axis=1)])), (np.array([sl.norm((x_reel-x_a), axis=2), sl.norm(theta_reel-theta_a, axis=1)]))
    
    F0_moy.append(np.average(F_0)), F0_min.append(np.min(F_0)), F0_max.append(np.max(F_0))
    Fa_moy.append(np.average(F_a)), Fa_min.append(np.min(F_a)), Fa_max.append(np.max(F_a))
    n0_theta_max.append(np.max(n_0[1])), n0_theta_min.append(np.min(n_0[1])), n0_theta_moy.append(np.average(n_0[1]))
    n0_x_max.append(np.max(n_0[0])), n0_x_min.append(np.min(n_0[0])), n0_x_moy.append(np.average(n_0[0]))
    na_theta_max.append(np.max(n_a[1])), na_theta_min.append(np.min(n_a[1])), na_theta_moy.append(np.average(n_a[1]))
    na_x_max.append(np.max(n_a[0])), na_x_min.append(np.min(n_a[0])), na_x_moy.append(np.average(n_a[0]))

figure1, (fig1, fig2, fig3) = plt.subplots(3, 1, sharex=True)
fig1.set_title("{} simulations avec K = {}, N = {}, sigma_theta = {}, sigma_x = {} (cas du {})".format(Nexp, K, N, sigma_theta, sigma_x, cas))
fig1.semilogy(liste_exp, n0_theta_moy, color="b", marker="s", markersize=5, linestyle="None", label="theta_0 - theta_reel")
fig1.semilogy(liste_exp, n0_theta_min, color="b", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_exp, n0_theta_max, color="b", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_exp, na_theta_moy, color="r", marker="s", markersize=5, linestyle="None", label="theta_a - theta_reel")
fig1.semilogy(liste_exp, na_theta_min, color="r", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_exp, na_theta_max, color="r", marker="_", markersize=5, linestyle="None")
fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig1.set_ylabel("selon theta")
fig2.semilogy(liste_exp, n0_x_moy, color="b", marker="s", markersize=5, linestyle="None", label="x_0 - x_reel")
fig2.semilogy(liste_exp, n0_x_min, color="b", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_exp, n0_x_max, color="b", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_exp, na_x_moy, color="r", marker="s", markersize=5, linestyle="None", label="x_a - x_reel")
fig2.semilogy(liste_exp, na_x_min, color="r", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_exp, na_x_max, color="r", marker="_", markersize=5, linestyle="None")
fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig2.set_ylabel("selon x")
fig3.semilogy(liste_exp, F0_moy, color="b", marker="s", markersize=5, linestyle="None", label="F_0")
fig3.semilogy(liste_exp, F0_min, color="b", marker="_", markersize=5, linestyle="None")
fig3.semilogy(liste_exp, F0_max, color="b", marker="_", markersize=5, linestyle="None")
fig3.semilogy(liste_exp, Fa_moy, color="r", marker="s", markersize=5, linestyle="None", label="F_a")
fig3.semilogy(liste_exp, Fa_min, color="r", marker="_", markersize=5, linestyle="None")
fig3.semilogy(liste_exp, Fa_max, color="r", marker="_", markersize=5, linestyle="None")
fig3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig3.set_ylabel("F")
fig3.set_xlabel("simulation")

## reestimation globale comparaison avec contrainte/ sans contrainte

liste_exp = np.arange(Nexp) + 1
F0_max, F0_min, F0_moy = [], [], []
Fa1_max, Fa1_min, Fa1_moy = [], [], []
Fa2_max, Fa2_min, Fa2_moy = [], [], []
n0_theta_max, n0_theta_min, n0_theta_moy = [], [], []
n0_x_max, n0_x_min, n0_x_moy = [], [], []
na1_theta_max, na1_theta_min, na1_theta_moy = [], [], []
na2_theta_max, na2_theta_min, na2_theta_moy = [], [], []
na1_x_max, na1_x_min, na1_x_moy = [], [], []
na2_x_max, na2_x_min, na2_x_moy = [], [], []

for n in range(Nexp):
    C_reel, X_reel, theta_reel, theta_0, theta_a1, theta_a2, x_reel, x_0, x_a1, x_a2 = reestimation_globale_comparaison()
    
    F_0, F_a1, F_a2, n_0, n_a1, n_a2 = abs(F(C_reel, theta_0, x_0, f)), abs(F(C_reel, theta_a1, x_a1, f)),  abs(F(C_reel, theta_a2, x_a2, f)), (np.array([sl.norm((x_reel-x_0), axis=2), sl.norm(theta_reel-theta_0, axis=1)])), (np.array([sl.norm((x_reel-x_a1), axis=2), sl.norm(theta_reel-theta_a1, axis=1)])), (np.array([sl.norm((x_reel-x_a2), axis=2), sl.norm(theta_reel-theta_a2, axis=1)]))
    
    F0_moy.append(np.average(F_0)), F0_min.append(np.min(F_0)), F0_max.append(np.max(F_0))
    Fa1_moy.append(np.average(F_a1)), Fa1_min.append(np.min(F_a1)), Fa1_max.append(np.max(F_a1))
    Fa2_moy.append(np.average(F_a2)), Fa2_min.append(np.min(F_a2)), Fa2_max.append(np.max(F_a2))
    n0_theta_max.append(np.max(n_0[1])), n0_theta_min.append(np.min(n_0[1])), n0_theta_moy.append(np.average(n_0[1]))
    n0_x_max.append(np.max(n_0[0])), n0_x_min.append(np.min(n_0[0])), n0_x_moy.append(np.average(n_0[0]))
    na1_theta_max.append(np.max(n_a1[1])), na1_theta_min.append(np.min(n_a1[1])), na1_theta_moy.append(np.average(n_a1[1]))
    na2_theta_max.append(np.max(n_a2[1])), na2_theta_min.append(np.min(n_a2[1])), na2_theta_moy.append(np.average(n_a2[1]))
    na1_x_max.append(np.max(n_a1[0])), na1_x_min.append(np.min(n_a1[0])), na1_x_moy.append(np.average(n_a1[0]))
    na2_x_max.append(np.max(n_a2[0])), na2_x_min.append(np.min(n_a2[0])), na2_x_moy.append(np.average(n_a2[0]))

figure1, (fig1, fig2, fig3) = plt.subplots(3, 1, sharex=True)
fig1.set_title("{} simulations avec K = {}, N = {}, sigma_theta = {}, sigma_x = {} (cas du {})".format(Nexp, K, N, sigma_theta, sigma_x, cas))
fig1.semilogy(liste_exp, n0_theta_moy, color="b", marker="s", markersize=5, linestyle="None", label="theta_0 - theta_reel")
fig1.semilogy(liste_exp, n0_theta_min, color="b", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_exp, n0_theta_max, color="b", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_exp, na1_theta_moy, color="r", marker="s", markersize=5, linestyle="None", label="theta_a1 - theta_reel (sans contrainte)")
fig1.semilogy(liste_exp, na1_theta_min, color="r", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_exp, na1_theta_max, color="r", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_exp, na2_theta_moy, color="g", marker="s", markersize=5, linestyle="None", label="theta_a2 - theta_reel (avec contrainte)")
fig1.semilogy(liste_exp, na2_theta_min, color="g", marker="_", markersize=5, linestyle="None")
fig1.semilogy(liste_exp, na2_theta_max, color="g", marker="_", markersize=5, linestyle="None")

fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig1.set_ylabel("selon theta")
fig2.semilogy(liste_exp, n0_x_moy, color="b", marker="s", markersize=5, linestyle="None", label="x_0 - x_reel")
fig2.semilogy(liste_exp, n0_x_min, color="b", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_exp, n0_x_max, color="b", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_exp, na1_x_moy, color="r", marker="s", markersize=5, linestyle="None", label="x_a1 - x_reel (sans contrainte)")
fig2.semilogy(liste_exp, na1_x_min, color="r", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_exp, na1_x_max, color="r", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_exp, na2_x_moy, color="g", marker="s", markersize=5, linestyle="None", label="x_a2 - x_reel (avec contrainte)")
fig2.semilogy(liste_exp, na2_x_min, color="g", marker="_", markersize=5, linestyle="None")
fig2.semilogy(liste_exp, na2_x_max, color="g", marker="_", markersize=5, linestyle="None")
fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig2.set_ylabel("selon x")
fig3.semilogy(liste_exp, F0_moy, color="b", marker="s", markersize=5, linestyle="None", label="F_0")
fig3.semilogy(liste_exp, F0_min, color="b", marker="_", markersize=5, linestyle="None")
fig3.semilogy(liste_exp, F0_max, color="b", marker="_", markersize=5, linestyle="None")
fig3.semilogy(liste_exp, Fa1_moy, color="r", marker="s", markersize=5, linestyle="None", label="F_a1 (sans contrainte)")
fig3.semilogy(liste_exp, Fa1_min, color="r", marker="_", markersize=5, linestyle="None")
fig3.semilogy(liste_exp, Fa1_max, color="r", marker="_", markersize=5, linestyle="None")
fig3.semilogy(liste_exp, Fa2_moy, color="g", marker="s", markersize=5, linestyle="None", label="F_a2 (avec contrainte)")
fig3.semilogy(liste_exp, Fa2_min, color="g", marker="_", markersize=5, linestyle="None")
fig3.semilogy(liste_exp, Fa2_max, color="g", marker="_", markersize=5, linestyle="None")
fig3.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig3.set_ylabel("F")
fig3.set_xlabel("simulation")

################################################################################
##                      SIMULATIONS AVEC PARAMETRES FIXES                     ##
################################################################################


## paramètres

K = 6
liste_N_x = [50, 100, 250, 500, 750, 1000, 1500, 2000]
liste_N_theta = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000]
Nexp = 10

cas = "satellite"
f = 10**6
X0, Y0, Z0, Z1, Z2 = 20000, 20000, 5000, 700000, 800000
sigma_x = 10**-2
sigma_theta = 10**-5

# cas = "drône"
# f = 10**4
# X0, Y0, Z0, Z1, Z2 = 100, 100, 40, 50, 75
# sigma_x = 10**-2
# sigma_theta = 10**-2

## réestimation des x à theta_0 = theta_reel

F0_max, F0_min, F0_moy = [], [], []
Fa_max, Fa_min, Fa_moy = [], [], []
n0_x_max, n0_x_min, n0_x_moy = [], [], []
na_x_max, na_x_min, na_x_moy = [], [], []

for N in liste_N_x:
    print(N)
    
    F0_max.append(0), F0_min.append(0), F0_moy.append(0)
    Fa_max.append(0), Fa_min.append(0), Fa_moy.append(0)
    n0_x_max.append(0), n0_x_min.append(0), n0_x_moy.append(0)
    na_x_max.append(0), na_x_min.append(0), na_x_moy.append(0)
    
    for exp in range(Nexp):
        print(exp)
        C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a = reestimation_x(theta_exact=True)
        
        F_0, F_a, n_0, n_a = abs(F(C_reel, theta_0, x_0, f)), abs(F(C_reel, theta_a, x_a, f)), sl.norm(x_reel-x_0, axis=2), sl.norm(x_reel-x_a, axis=2)
        
        F0_max[-1] += np.max(F_0)
        F0_min[-1] += np.min(F_0)
        F0_moy[-1] += np.average(F_0)
        Fa_max[-1] += np.max(F_a)
        Fa_min[-1] += np.min(F_a)
        Fa_moy[-1] += np.average(F_a)
        n0_x_max[-1] += np.max(n_0)
        n0_x_min[-1] += np.min(n_0)
        n0_x_moy[-1] += np.average(n_0)
        na_x_max[-1] += np.max(n_a)
        na_x_min[-1] += np.min(n_a)
        na_x_moy[-1] += np.average(n_a)
    
F0_max = np.asarray(F0_max) / Nexp
F0_min = np.asarray(F0_min) / Nexp
F0_moy = np.asarray(F0_moy) / Nexp
Fa_max = np.asarray(Fa_max) / Nexp
Fa_min = np.asarray(Fa_min) / Nexp
Fa_moy = np.asarray(Fa_moy) / Nexp
n0_x_max = np.asarray(n0_x_max) / Nexp
n0_x_min = np.asarray(n0_x_min) / Nexp
n0_x_moy = np.asarray(n0_x_moy) / Nexp
na_x_max = np.asarray(na_x_max) / Nexp
na_x_min = np.asarray(na_x_min) / Nexp
na_x_moy = np.asarray(na_x_moy) / Nexp

figure2, (fig1, fig2) = plt.subplots(2, 1, sharex=True)
fig1.set_title("{} simulations par N avec K = {}, sigma_x = {} (cas du {})".format(Nexp, K, sigma_x, cas))
fig1.loglog(liste_N_x, n0_x_moy, color="b", marker="s", markersize=5, linestyle="None", label="x_0 - x_reel")
fig1.loglog(liste_N_x, n0_x_min, color="b", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_x, n0_x_max, color="b", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_x, na_x_moy, color="r", marker="s", markersize=5, linestyle="None", label="x_a - x_reel")
fig1.loglog(liste_N_x, na_x_min, color="r", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_x, na_x_max, color="r", marker="_", markersize=5, linestyle="None")
fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig1.set_ylabel("selon x")
fig2.loglog(liste_N_x, F0_moy, color="b", marker="s", markersize=5, linestyle="None", label="F_0")
fig2.loglog(liste_N_x, F0_min, color="b", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_x, F0_max, color="b", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_x, Fa_moy, color="r", marker="s", markersize=5, linestyle="None", label="F_a")
fig2.loglog(liste_N_x, Fa_min, color="r", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_x, Fa_max, color="r", marker="_", markersize=5, linestyle="None")
fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig2.set_ylabel("F")
fig2.set_xlabel("N")

## réestimation des x à theta_0 != theta_reel

F0_max, F0_min, F0_moy = [], [], []
Fa_max, Fa_min, Fa_moy = [], [], []
n0_x_max, n0_x_min, n0_x_moy = [], [], []
na_x_max, na_x_min, na_x_moy = [], [], []

for N in liste_N_x:
    print(N)
    
    F0_max.append(0), F0_min.append(0), F0_moy.append(0)
    Fa_max.append(0), Fa_min.append(0), Fa_moy.append(0)
    n0_x_max.append(0), n0_x_min.append(0), n0_x_moy.append(0)
    na_x_max.append(0), na_x_min.append(0), na_x_moy.append(0)
    
    for exp in range(Nexp):
        print(exp)
        C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a = reestimation_x()
        
        F_0, F_a, n_0, n_a = abs(F(C_reel, theta_0, x_0, f)), abs(F(C_reel, theta_a, x_a, f)), sl.norm(x_reel-x_0, axis=2), sl.norm(x_reel-x_a, axis=2)
        
        F0_max[-1] += np.max(F_0)
        F0_min[-1] += np.min(F_0)
        F0_moy[-1] += np.average(F_0)
        Fa_max[-1] += np.max(F_a)
        Fa_min[-1] += np.min(F_a)
        Fa_moy[-1] += np.average(F_a)
        n0_x_max[-1] += np.max(n_0)
        n0_x_min[-1] += np.min(n_0)
        n0_x_moy[-1] += np.average(n_0)
        na_x_max[-1] += np.max(n_a)
        na_x_min[-1] += np.min(n_a)
        na_x_moy[-1] += np.average(n_a)
    
F0_max = np.asarray(F0_max) / Nexp
F0_min = np.asarray(F0_min) / Nexp
F0_moy = np.asarray(F0_moy) / Nexp
Fa_max = np.asarray(Fa_max) / Nexp
Fa_min = np.asarray(Fa_min) / Nexp
Fa_moy = np.asarray(Fa_moy) / Nexp
n0_x_max = np.asarray(n0_x_max) / Nexp
n0_x_min = np.asarray(n0_x_min) / Nexp
n0_x_moy = np.asarray(n0_x_moy) / Nexp
na_x_max = np.asarray(na_x_max) / Nexp
na_x_min = np.asarray(na_x_min) / Nexp
na_x_moy = np.asarray(na_x_moy) / Nexp

figure3, (fig1, fig2) = plt.subplots(2, 1, sharex=True)
fig1.set_title("{} simulations par N avec K = {}, sigma_theta = {}, sigma_x = {} (cas du {})".format(Nexp, K, sigma_theta, sigma_x, cas))
fig1.loglog(liste_N_x, n0_x_moy, color="b", marker="s", markersize=5, linestyle="None", label="x_0 - x_reel")
fig1.loglog(liste_N_x, n0_x_min, color="b", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_x, n0_x_max, color="b", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_x, na_x_moy, color="r", marker="s", markersize=5, linestyle="None", label="x_a - x_reel")
fig1.loglog(liste_N_x, na_x_min, color="r", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_x, na_x_max, color="r", marker="_", markersize=5, linestyle="None")
fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig1.set_ylabel("selon x")
fig2.loglog(liste_N_x, F0_moy, color="b", marker="s", markersize=5, linestyle="None", label="F_0")
fig2.loglog(liste_N_x, F0_min, color="b", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_x, F0_max, color="b", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_x, Fa_moy, color="r", marker="s", markersize=5, linestyle="None", label="F_a")
fig2.loglog(liste_N_x, Fa_min, color="r", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_x, Fa_max, color="r", marker="_", markersize=5, linestyle="None")
fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig2.set_ylabel("F")
fig2.set_xlabel("N")

## réestimation des angles à x_0 = x_reel

F0_max, F0_min, F0_moy = [], [], []
Fa_max, Fa_min, Fa_moy = [], [], []
n0_theta_max, n0_theta_min, n0_theta_moy = [], [], []
na_theta_max, na_theta_min, na_theta_moy = [], [], []

for N in liste_N_theta:
    print(N)
    
    F0_max.append(0), F0_min.append(0), F0_moy.append(0)
    Fa_max.append(0), Fa_min.append(0), Fa_moy.append(0)
    n0_theta_max.append(0), n0_theta_min.append(0), n0_theta_moy.append(0)
    na_theta_max.append(0), na_theta_min.append(0), na_theta_moy.append(0)
    
    for exp in range(Nexp):
        print(exp)
        C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a = reestimation_theta(x_exact=True)
        
        F_0, F_a, n_0, n_a = abs(F(C_reel, theta_0, x_0, f)), abs(F(C_reel, theta_a, x_a, f)), sl.norm(theta_reel-theta_0, axis=1), sl.norm(theta_reel-theta_a, axis=1)
        
        F0_max[-1] += np.max(F_0)
        F0_min[-1] += np.min(F_0)
        F0_moy[-1] += np.average(F_0)
        Fa_max[-1] += np.max(F_a)
        Fa_min[-1] += np.min(F_a)
        Fa_moy[-1] += np.average(F_a)
        n0_theta_max[-1] += np.max(n_0)
        n0_theta_min[-1] += np.min(n_0)
        n0_theta_moy[-1] += np.average(n_0)
        na_theta_max[-1] += np.max(n_a)
        na_theta_min[-1] += np.min(n_a)
        na_theta_moy[-1] += np.average(n_a)
    
F0_max = np.asarray(F0_max) / Nexp
F0_min = np.asarray(F0_min) / Nexp
F0_moy = np.asarray(F0_moy) / Nexp
Fa_max = np.asarray(Fa_max) / Nexp
Fa_min = np.asarray(Fa_min) / Nexp
Fa_moy = np.asarray(Fa_moy) / Nexp
n0_theta_max = np.asarray(n0_theta_max) / Nexp
n0_theta_min = np.asarray(n0_theta_min) / Nexp
n0_theta_moy = np.asarray(n0_theta_moy) / Nexp
na_theta_max = np.asarray(na_theta_max) / Nexp
na_theta_min = np.asarray(na_theta_min) / Nexp
na_theta_moy = np.asarray(na_theta_moy) / Nexp

figure4, (fig1, fig2) = plt.subplots(2, 1, sharex=True)
fig1.set_title("{} simulations par N avec K = {}, sigma_theta = {} (cas du {})".format(Nexp, K, sigma_theta, cas))
fig1.loglog(liste_N_theta, n0_theta_moy, color="b", marker="s", markersize=5, linestyle="None", label="theta_0 - theta_reel")
fig1.loglog(liste_N_theta, n0_theta_min, color="b", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_theta, n0_theta_max, color="b", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_theta, na_theta_moy, color="r", marker="s", markersize=5, linestyle="None", label="theta_a - theta_reel")
fig1.loglog(liste_N_theta, na_theta_min, color="r", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_theta, na_theta_max, color="r", marker="_", markersize=5, linestyle="None")
fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig1.set_ylabel("selon theta")
fig2.loglog(liste_N_theta, F0_moy, color="b", marker="s", markersize=5, linestyle="None", label="F_0")
fig2.loglog(liste_N_theta, F0_min, color="b", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_theta, F0_max, color="b", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_theta, Fa_moy, color="r", marker="s", markersize=5, linestyle="None", label="F_a")
fig2.loglog(liste_N_theta, Fa_min, color="r", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_theta, Fa_max, color="r", marker="_", markersize=5, linestyle="None")
fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig2.set_ylabel("F")
fig2.set_xlabel("N")

## réestimation des angles à x_0 != x_reel

F0_max, F0_min, F0_moy = [], [], []
Fa_max, Fa_min, Fa_moy = [], [], []
n0_theta_max, n0_theta_min, n0_theta_moy = [], [], []
na_theta_max, na_theta_min, na_theta_moy = [], [], []

for N in liste_N_theta:
    print(N)
    
    F0_max.append(0), F0_min.append(0), F0_moy.append(0)
    Fa_max.append(0), Fa_min.append(0), Fa_moy.append(0)
    n0_theta_max.append(0), n0_theta_min.append(0), n0_theta_moy.append(0)
    na_theta_max.append(0), na_theta_min.append(0), na_theta_moy.append(0)
    
    for exp in range(Nexp):
        print(exp)
        C_reel, X_reel, theta_reel, theta_0, theta_a, x_reel, x_0, x_a = reestimation_theta()
        
        F_0, F_a, n_0, n_a = abs(F(C_reel, theta_0, x_0, f)), abs(F(C_reel, theta_a, x_a, f)), sl.norm(theta_reel-theta_0, axis=1), sl.norm(theta_reel-theta_a, axis=1)
        
        F0_max[-1] += np.max(F_0)
        F0_min[-1] += np.min(F_0)
        F0_moy[-1] += np.average(F_0)
        Fa_max[-1] += np.max(F_a)
        Fa_min[-1] += np.min(F_a)
        Fa_moy[-1] += np.average(F_a)
        n0_theta_max[-1] += np.max(n_0)
        n0_theta_min[-1] += np.min(n_0)
        n0_theta_moy[-1] += np.average(n_0)
        na_theta_max[-1] += np.max(n_a)
        na_theta_min[-1] += np.min(n_a)
        na_theta_moy[-1] += np.average(n_a)
    
F0_max = np.asarray(F0_max) / Nexp
F0_min = np.asarray(F0_min) / Nexp
F0_moy = np.asarray(F0_moy) / Nexp
Fa_max = np.asarray(Fa_max) / Nexp
Fa_min = np.asarray(Fa_min) / Nexp
Fa_moy = np.asarray(Fa_moy) / Nexp
n0_theta_max = np.asarray(n0_theta_max) / Nexp
n0_theta_min = np.asarray(n0_theta_min) / Nexp
n0_theta_moy = np.asarray(n0_theta_moy) / Nexp
na_theta_max = np.asarray(na_theta_max) / Nexp
na_theta_min = np.asarray(na_theta_min) / Nexp
na_theta_moy = np.asarray(na_theta_moy) / Nexp

figure5, (fig1, fig2) = plt.subplots(2, 1, sharex=True)
fig1.set_title("{} simulations par N avec K = {}, sigma_theta = {}, sigma_x = {} (cas du {})".format(Nexp, K, sigma_theta, sigma_x, cas))
fig1.loglog(liste_N_theta, n0_theta_moy, color="b", marker="s", markersize=5, linestyle="None", label="theta_0 - theta_reel")
fig1.loglog(liste_N_theta, n0_theta_min, color="b", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_theta, n0_theta_max, color="b", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_theta, na_theta_moy, color="r", marker="s", markersize=5, linestyle="None", label="theta_a - theta_reel")
fig1.loglog(liste_N_theta, na_theta_min, color="r", marker="_", markersize=5, linestyle="None")
fig1.loglog(liste_N_theta, na_theta_max, color="r", marker="_", markersize=5, linestyle="None")
fig1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig1.set_ylabel("selon theta")
fig2.loglog(liste_N_theta, F0_moy, color="b", marker="s", markersize=5, linestyle="None", label="F_0")
fig2.loglog(liste_N_theta, F0_min, color="b", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_theta, F0_max, color="b", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_theta, Fa_moy, color="r", marker="s", markersize=5, linestyle="None", label="F_a")
fig2.loglog(liste_N_theta, Fa_min, color="r", marker="_", markersize=5, linestyle="None")
fig2.loglog(liste_N_theta, Fa_max, color="r", marker="_", markersize=5, linestyle="None")
fig2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig2.set_ylabel("F")
fig2.set_xlabel("N")





################################################################################
##                          ETUDE DU CONDITIONNEMENT                          ##
################################################################################

def matrices_A_B_M(x_exact=False, theta_exact=False):
    """ Simule une matrice M aléatoire en créant des données associées au paramètres
    
    PARAMETRES
    x_exact [bool] : si vrai, fixe x_0 à x_reel (pas de perturbation des points)
    theta_exact [bool] : si vrai, fixe theta_0 à theta_reel (pas de perturbation des angles)
    affichage [bool] : si vrai, affiche les données créées sur une figure 3D
    
    SORTIES
    A [array (K*(K-1)//2*N, 3*K)] : matrice A
    B [array (K*(K-1)//2*N, 2*K*N)] : matrice B
    M [array (K*(K-1)//2*N, 3*K+2*K*N)] : matrice M
    
    """
    
    C_reel, X_reel, theta_reel = genere_donnees()
    
    matrices_P = []
    for i in range(K):
        matrices_P.append(matrice_P(C_reel[i], theta_reel[i], f))
    
    x_reel = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
            x_reel[j, i] = point[:-1] / point[-1]
    
    # perturbation des coordonnées et des angles
    
    x_0 = x_reel
    if not x_exact: x_0 += normal(0, sigma_x, (N, K, 2))
    theta_0 = theta_reel
    if not theta_exact: theta_0 += normal(0, sigma_theta, (K, 3))
    
    # minimisation de F
    
    A, B = matrices_A_B(C_reel, theta_0, x_0, f)
    
    return A, B, np.concatenate((A, B), axis=1)

# paramètres

K = 6

cas = "satellite"
f = 10**6
X0, Y0, Z0, Z1, Z2 = 20000, 20000, 5000, 700000, 800000
sigma_x = 10**-2
sigma_theta = 10**-5

# cas = "drône"
# f = 10**4
# X0, Y0, Z0, Z1, Z2 = 100, 100, 40, 50, 75
# sigma_x = 10**-2
# sigma_theta = 10**-2

liste_N = 2**np.arange(3, 9)
Nexp = 10

# calculs

VSmin_A, VSmax_A, condA = [], [], []
VSmin_B, VSmax_B, condB = [], [], []
VSmin_M, VSmax_M, condM = [], [], []
liste_N_repetee = []
for N in liste_N:
    print(N)
    for i in range(Nexp):
        A, B, M = matrices_A_B_M()
        cA = sl.svd(A, compute_uv=False)
        cB = sl.svd(B, compute_uv=False)
        cM = sl.svd(M, compute_uv=False)
        liste_N_repetee.append(N)
        VSmin_A.append(cA[-1]), VSmax_A.append(cA[0]), condA.append(cA[0]/cA[-1])
        VSmin_B.append(cB[-1]), VSmax_B.append(cB[0]), condB.append(cB[0]/cB[-1])
        VSmin_M.append(cM[-1]), VSmax_M.append(cM[0]), condM.append(cM[0]/cM[-1])

plot1, (fig1, fig2) = plt.subplots(2, sharex=True)
fig1.set_title("Etude de A, K = {}, sigma_x = {}, sigma_theta = {} (cas du {})".format(K, sigma_x, sigma_theta, cas)), 
fig1.set_ylabel("valeurs singulières")
fig1.set_ylim(min(VSmin_A)/10, max(VSmax_A)*10)
fig1.loglog(liste_N_repetee, VSmax_A, marker="+", markersize=10, linestyle="None", color="b", label="VS maximale")
fig1.loglog(liste_N_repetee, VSmin_A, marker="+", markersize=10, linestyle="None", color="r", label="VS minimale")
fig1.legend(loc=6)
fig2.set_ylabel("conditionnement")
fig2.set_xlabel("N")
fig2.loglog(liste_N_repetee, condA, marker="+", markersize=10, linestyle="None", color="g")

plot2, (fig1, fig2) = plt.subplots(2, sharex=True)
fig1.set_title("Etude de B, K = {}, sigma_x = {}, sigma_theta = {} (cas du {})".format(K, sigma_x, sigma_theta, cas))
fig1.set_ylabel("valeurs singulières")
fig1.set_ylim(min(VSmin_B)/10, max(VSmax_B)*10)
fig1.loglog(liste_N_repetee, VSmax_B, marker="+", markersize=10, linestyle="None", color="b", label="VS maximale")
fig1.loglog(liste_N_repetee, VSmin_B, marker="+", markersize=10, linestyle="None", color="r", label="VS minimale")
fig1.legend(loc=6)
fig2.set_ylabel("conditionnement")
fig2.set_xlabel("N")
fig2.loglog(liste_N_repetee, condB, marker="+", markersize=10, linestyle="None", color="g")

plot3, (fig1, fig2) = plt.subplots(2, sharex=True)
fig1.set_title("Etude de M, K = {}, sigma_x = {}, sigma_theta = {} (cas du {})".format(K, sigma_x, sigma_theta, cas))
fig1.set_ylabel("valeurs singulières")
fig1.set_ylim(min(VSmin_M)/10, max(VSmax_M)*10)
fig1.loglog(liste_N_repetee, VSmax_M, marker="+", markersize=10, linestyle="None", color="b", label="VS maximale")
fig1.loglog(liste_N_repetee, VSmin_M, marker="+", markersize=10, linestyle="None", color="r", label="VS minimale")
fig1.legend(loc=6)
fig2.set_ylabel("conditionnement")
fig2.set_xlabel("N")
fig2.loglog(liste_N_repetee, condM, marker="+", markersize=10, linestyle="None", color="g")