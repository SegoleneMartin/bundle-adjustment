################################################################################
################################################################################
##                   Bundle Adjustment With Known Positions                   ##
##                                                                            ##
##                          DONNEES REELLES - NAPIER                          ##
################################################################################
################################################################################

## packages

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
import random
import scipy.linalg as sl
from numpy import cos, sin
from math import atan2
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as ssl
from scipy.linalg import solve, rq 
from math import atan2
import os

dossier = "C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes"
dir = "C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images"
dir_corres_02 = "C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images/matches/matches_ransac_0.2"

## import de BA_fonctions

# dossier contient le chemin du fichier maching
os.chdir(dir)
from matching import *
# dossier contient le chemin du fichier BA_fonctions
os.chdir(dossier)
from BA_fonctions import *

## création des fichiers contenant les points C et les matrices K et R

# données

cameras = list(range(1, 21))

# traitement

os.chdir(dir)

cameras.sort()

str_cameras = []
for cam in cameras:
    if cam < 10: str_cameras.append("0" + str(cam))
    else: str_cameras.append(str(cam))

K = len(cameras)

liste_fichiers_P, liste_fichiers_C, liste_fichiers_R, liste_fichiers_K = [], [], [], []
for cam in str_cameras:
    liste_fichiers_P.append("P_" + cam + ".txt")
    liste_fichiers_C.append("C_" + cam + ".txt")
    liste_fichiers_R.append("R_" + cam + ".txt")
    liste_fichiers_K.append("K_" + cam + ".txt")

matrices_P, matrices_C, matrices_R, matrices_K = [], [], [], []
for fichier_P in liste_fichiers_P:
    P = np.loadtxt(fichier_P)
    matrices_P.append(P)

for P in matrices_P:
    Ck, Rk, Kk = kr_from_p(P)
    matrices_C.append(np.asarray(Ck))
    matrices_R.append(np.asarray(Rk))
    matrices_K.append(np.asarray(Kk))

for k in range(K):
    np.savetxt(liste_fichiers_C[k], matrices_C[k])
    np.savetxt(liste_fichiers_R[k], matrices_R[k])
    np.savetxt(liste_fichiers_K[k], matrices_K[k])

## simulations


def initialisation(cameras): # retourne theta_0, x_0 à partir d'un jeu de caméras

    os.chdir(dir)
    
    cameras.sort()
    
    str_cameras = []
    for cam in cameras:
        if cam < 10: str_cameras.append("0" + str(cam))
        else: str_cameras.append(str(cam))
    
    K = len(cameras)
    
    liste_fichiers_P, liste_fichiers_C, liste_fichiers_R, liste_fichiers_K = [], [], [], []
    for cam in str_cameras:
        liste_fichiers_P.append("P_" + cam + ".txt")
        liste_fichiers_C.append("C_" + cam + ".txt")
        liste_fichiers_R.append("R_" + cam + ".txt")
        liste_fichiers_K.append("K_" + cam + ".txt")
    
    matrices_P, matrices_C, matrices_R, matrices_K = [], [], [], []
    for k in range(K):
        matrices_P.append(np.loadtxt(liste_fichiers_P[k]))
        matrices_C.append(np.loadtxt(liste_fichiers_C[k]))
        matrices_R.append(np.loadtxt(liste_fichiers_R[k]))
        matrices_K.append(np.loadtxt(liste_fichiers_K[k]))
    
    os.chdir(dir_corres_02)
    
    fichier_corres = "corres"
    for cam in str_cameras:
        fichier_corres += "_" + cam
    fichier_corres += ".txt"
    
    correspondances = np.loadtxt(fichier_corres)
    
    N = correspondances.shape[0]

    C_reel = np.asarray(matrices_C)
    theta_0 = np.zeros((K, 3))
    
    for k in range(K): theta_0[k] = theta_from_r(matrices_R[k])
    
    x_0 = correspondances.reshape((N, 6, 2), order='F')
    
    for k in range(K):
        [px, py] = matrices_K[k][:-1, -1]
        for n in range(N):
            x_0[n, k] -= [px, py]   # translation
            x_0[n, k, 1] *= -1      # repère image "direct"
            
    liste_f = []
    for k in range(K):
        liste_f.append(abs(matrices_K[k][0, 0]))
        liste_f.append(abs(matrices_K[k][1, 1]))

    f = np.average(liste_f)

    return(theta_0, x_0, f, C_reel)
    
    
    
# tracé sur un globe de Napier et des caméras pointant vers le centre de la Terre

lat, lon = -39.483724 * np.pi / 180, 176.913477 * np.pi / 180       # coord Napier

def scene_3D_reelle(cameras=True, direction=True, earth=True, location=True):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_aspect("equal")
    plt.xlabel("X"), plt.ylabel("Y")
    
    if cameras: plt.plot(C_reel[:, 0], C_reel[:, 1], C_reel[:, 2], marker="+", markersize=10, color="r", label="cameras", linestyle="None")
    
    if direction:
        Z_cam = []
        for k in range(K):
            Z_cam.append(np.linalg.inv(matrices_R[k]).dot(np.array([0, 0, -1])))
        
        t = 700000
        for C, Z in zip(C_reel, Z_cam):
            ZZ = C + t*Z
            plt.plot([C[0], ZZ[0]], [C[1], ZZ[1]], [C[2], ZZ[2]], color="k")
    
    if earth:
        plt.plot([0], [0], [0], color="k", marker="+", markersize=15)
        a = 6378137
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = a*np.cos(u)*np.sin(v)
        y = a*np.sin(u)*np.sin(v)
        z = a*np.cos(v)
        ax.plot_wireframe(x, y, z, color="b")
    
    if location:
        a = 6378137
        f = 1 / 298.257223563 
        b = a * (1-f)
        h = 0
        
        e2 = (a**2-b**2) / b**2;
        W = np.sqrt(1 - e2*np.sin(lat)**2)
        NN = a / W
        X = (NN+h) * np.cos(lon) * np.cos(lat)
        Y = (NN+h) * np.sin(lon) * np.cos(lat)
        Z = (NN*(1-e2)+h) * np.sin(lat)
        
        plt.plot([X], [Y], [Z], marker="+", color="g", markersize=10)



sigma_x = 0.1
sigma_theta = 0.00001

lamb_theta = 0.02
lamb_x = 0.02

def reestimation_theta_x_ref(x_0, theta_0, f, C_reel):  # pour un x de reference, renvoie l'estimation en ce x_ref et aussi theta_a

    N, K = x_0.shape[0], x_0.shape[1]
    
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
    M = csr_matrix(M)
    
    # Recherche de la solution qui minimise
    
    erreur = ssl.lsqr(M,b, conlim=1.0e+8)[0]
    
    theta_a = theta_0 + erreur[:3*K].reshape((K, 3))
    x_a = np.zeros((N, K, 2))
    for j in range(N):
        for i in range(K):
            x_a[j, i, 0] = x_0[j, i, 0] + erreur[3*K + 2*K*j + 2*i]
            x_a[j, i, 1] = x_0[j, i, 1] + erreur[3*K + 2*K*j + 2*i + 1]
    
    return(theta_a, x_a[0,:,:])
    
    

def reestimation_theta_ref(cam_ref, x_0, theta_0, f, C_reel):  # pour une caméra de référence, renvoie l'estimation de l'angle de cette caméra

    N, K = x_0.shape[0], x_0.shape[1]
    
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
    M = csr_matrix(M)
    
    # Recherche de la solution qui minimise
    
    erreur = ssl.lsqr(M,b, conlim=1.0e+8)[0]
    
    theta_a = theta_0 + erreur[:3*K].reshape((K, 3))
    
    return(theta_a[cameras.index(cam_ref),:])
    
    

### Application 1: comparaisons sur différents sous ensembles de points

cameras = [1, 2, 3, 4, 5, 6]
#cameras = [2, 15, 16, 17, 18, 19]
#cameras = [12, 13, 14, 18, 19, 20]

theta_0, x_0, f, C_reel = initialisation(cameras)
N, K = x_0.shape[0], theta_0.shape[0]

# choix de x_ref et des ensembles de points contenant x_ref

alea = random.randint(0, N)
x_ref = x_0[alea]
x_0_new = np.concatenate((x_0[:alea], x_0[alea+1:]), axis=0)

taille_set = 200 #Nombre de points dans les sous ensembles à partir desquels on estime une solution
nb_set = 5
liste_set = []

if (taille_set - 1) * nb_set > N:
    print("Impossible d'avoir {} sets de taille {} tout en les voulant disjoints".format(nb_set, taille_set))

for k in range(nb_set):
    liste_set.append(np.concatenate((x_ref.reshape((1,6,2)), x_0_new[k*(taille_set-1): k*(taille_set-1) + taille_set-1]), axis=0))
    

# application de reestimation_theta_x_ref aux ensembles de points contenus dans liste_set

liste_theta_a = []
liste_x_ref_a = []

for ensemble in liste_set:
    theta_a, x_ref_a = reestimation_theta_x_ref(ensemble, theta_0, f, C_reel)
    liste_theta_a.append(theta_a)
    liste_x_ref_a.append(x_ref_a)
    
liste_theta_a = np.array(liste_theta_a)
liste_x_ref_a = np.array(liste_x_ref_a)

# distance max entre les elements de liste_x_ref_a

def max_dist_x_ref(i, liste_x_ref_a): 
    """ i numero de la camera """
    maximum = 0
    for  k in range(nb_set):
        for p in range(k+1, nb_set):
            candidat = max([sl.norm(liste_x_ref_a[k, i, :] - liste_x_ref_a[p, i, :]) for p in range(k+1, nb_set)])
            if maximum < candidat: maximum = candidat
    return(maximum)

    
### Plot de comparaison des solutions à partir de différents sous ensembles

# plot sur x_ref

for i in range(K):
    plt.subplot(2,3,i+1)
    moyenne = np.average(liste_x_ref_a[:, i, :], axis=0)
    x_ref_max = max_dist_x_ref(i, liste_x_ref_a)
    # print("Distance de la moyenne des x_ref estimés au x_ref initial", sl.norm(moyenne - x_ref[i,:]))
    # print("Distance maximale entre les x_ref estimés", x_ref_max)
    if i == 2:
        plt.plot(moyenne[0], moyenne[1], marker='x', markersize='10', color='r', linestyle='None', label='moyenne')
        plt.plot(liste_x_ref_a[:, i, 0], liste_x_ref_a[:, i, 1], marker='x', markersize='10', linestyle='None', color='b', label='x_ref estimés')
        plt.plot(x_ref[i,0], x_ref[i,1], marker='x', markersize='10', color='g', linestyle='None', label='x_ref initial')
        plt.title('camera {}'.format(cameras[i]),  y=1.04)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    else: 
        plt.plot(moyenne[0], moyenne[1], marker='x', markersize='10', color='r')
        plt.plot(liste_x_ref_a[:, i, 0], liste_x_ref_a[:, i, 1], marker='x', markersize='10', linestyle='None', color='b')
        plt.plot(x_ref[i,0], x_ref[i,1], marker='x', markersize='10', color='g')
        plt.title('camera {}'.format(cameras[i]),  y=1.04)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.suptitle('Nombre d\'ensembles de points: {}. Taille des ensembles: {}'.format(nb_set, taille_set))
    

plt.show()


# plot sur les angles

fig, axarr = plt.subplots(6, 3)
liste_axes = ['X', 'Y', 'Z']

for i in range(K):
    moyenne = np.average(liste_theta_a[:, i, :], axis=0)
    for k in range(3):
        moy = moyenne[k]
        if k == 0:
             axarr[i, k].set_ylabel('camera {}'.format(cameras[i]), rotation=0, labelpad=60, fontsize=17)
             axarr[i, k].yaxis.set_label_position('left')
        if i == 0:
            axarr[i, k].set_xlabel(liste_axes[k], fontsize=17)
            axarr[i, k].xaxis.set_label_position('top') 
        
        if i == 0 and k == 2:
            axarr[i, k].plot(moy, 0, marker='x', markersize='10', color='r', label='moyenne', linestyle='None')
            axarr[i, k].plot(liste_theta_a[:, i, k], np.zeros(len(liste_theta_a[:, i, k])), marker='x', markersize='10', linestyle='None', color='b', label='theta estimés')
            axarr[i, k].plot(theta_0[i,k], 0, marker='x', markersize='10', color='g', label='theta_0', linestyle='None')
            axarr[i, k].axhline(0, color='k')
            axarr[i, k].set_yticklabels([])
            axarr[i, k].legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
            axarr[i, k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        else:
            axarr[i, k].plot(moy, 0, marker='x', markersize='10', color='r')
            axarr[i, k].plot(liste_theta_a[:, i, k], np.zeros(len(liste_theta_a[:, i, k])), marker='x', markersize='10', linestyle='None', color='b')
            axarr[i, k].plot(theta_0[i,k], 0, marker='x', markersize='10', color='g')
            axarr[i, k].axhline(0, color='k')
            axarr[i, k].set_yticklabels([])
            axarr[i, k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.suptitle('Nombre d\'ensembles de points: {}. Taille des ensembles: {}'.format(nb_set, taille_set))

plt.show()


### Application 2: comparaisons sur différents sous ensembles de caméras

liste_cameras = [[1, 2, 3, 4, 5, 6], [2, 15, 16, 17, 18, 19]]
cam_ref = list(set.intersection(*map(set,liste_cameras)))[0]  # intersection

liste_theta_ref_a = []
liste_theta_ref = []

for cameras in liste_cameras:
    matching(cameras) 
    theta_0, x_0, f, C_reel = initialisation(cameras)
    x_0 = x_0[:300]
    theta_ref = theta_0[cameras.index(cam_ref)]
    theta_ref_a = reestimation_theta_ref(cam_ref, x_0, theta_0, f, C_reel)
    liste_theta_ref_a.append(theta_ref_a)
    liste_theta_ref.append(theta_ref)
    
liste_theta_ref_a = np.array(liste_theta_ref_a)
liste_theta_ref = np.array(liste_theta_ref)


### plot
fig, axarr = plt.subplots(1, 3)
liste_axes = ['X', 'Y', 'Z']


for k in range(3):
    if k == 0:
        axarr[k].set_ylabel('camera {}'.format(cam_ref), rotation=0, labelpad=60, fontsize=17)
        axarr[k].yaxis.set_label_position('left')

    axarr[k].set_xlabel(liste_axes[k], fontsize=17)
    axarr[k].xaxis.set_label_position('top') 
    
    if k == 2:
        axarr[k].plot(liste_theta_ref_a[:, k], np.zeros(len(liste_theta_ref_a[:, k])), marker='x', markersize='10', linestyle='None', color='b', label='theta estimés')
        axarr[k].plot(liste_theta_ref[:, k], np.zeros(len(liste_theta_ref[:, k])), marker='x', markersize='10', color='g', label='theta_0', linestyle='None')
        axarr[k].axhline(0, color='k')
        axarr[k].set_yticklabels([])
        axarr[k].legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)
        axarr[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
    else:
        axarr[k].plot(liste_theta_ref_a[:, k], np.zeros(len(liste_theta_ref_a[:, k])), marker='x', markersize='10', linestyle='None', color='b', label='theta estimés')
        axarr[k].plot(liste_theta_ref[:, k], np.zeros(len(liste_theta_ref[:, k])), marker='x', markersize='10', color='g', label='theta_0', linestyle='None')
        axarr[k].axhline(0, color='k')
        axarr[k].set_yticklabels([])
        axarr[k].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.show()



    
