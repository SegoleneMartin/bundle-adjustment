################################################################################
################################################################################
##                   Bundle Adjustment With Known Positions                   ##
##                                                                            ##
##                    RECHERCHE DES MATCHING ENTRE CAMERAS                    ##
################################################################################
################################################################################

## packages

import numpy as np
import os
import pylab as plt
from numpy.random import rand, normal
import random
import networkx as nx
dossier = "C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes"
dir = "C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images"
dir_corres_02 = "C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images/matches/matches_ransac_0.2"
dir_corres = "C:/Users/Ségolène Martin/Documents/ENS/L3/Stage/codes/napier_pleiades_20_images/matches"

################################################################################
##             RECHERCHE DES PLUS LONGS FICHIERS DE CORRESPONDANCE            ##
################################################################################
'''
# caméras
cameras = list(range(1, 21))

# algo

os.chdir(dir_corres_02)

cameras.sort()

str_cameras = []
for cam in cameras:
    if cam < 10: str_cameras.append("0" + str(cam))
    else: str_cameras.append(str(cam))

K = len(cameras)

liste_fichiers = []
for k1 in range(K):
    for k2 in range(k1+1, K):
        liste_fichiers.append((k1, k2, "m_" + str_cameras[k1] + "_" + str_cameras[k2] + ".txt"))

matrice_corres = np.ones((K, K))

for (k1, k2, fichier) in liste_fichiers:
    taille = np.loadtxt(fichier).shape[0]
    matrice_corres[k1, k2], matrice_corres[k2, k1] = taille, taille

# affichage

X, Y = np.meshgrid(np.asarray(cameras + [21]) - 0.5, np.asarray(cameras + [21]) - 0.5)

plt.figure()
plt.axis([0, 21, 0, 21])
plt.pcolor(X, Y, matrice_corres)
plt.colorbar()
plt.title("nombre de correspondances entre les caméras")
plt.xlabel("caméra"), plt.ylabel("caméra")

plt.figure()
plt.axis([0, 21, 0, 21])
plt.pcolor(X, Y, np.log(matrice_corres))
plt.colorbar()
plt.title("log du nombre de correspondances entre les caméras")
plt.xlabel("caméra"), plt.ylabel("caméra")


'''


################################################################################
##                         MATCHING VERSION GRAPHES                           ##
################################################################################


def matching(cameras):

    # algo
    os.chdir(dir_corres_02)
    
    cameras.sort()
    
    str_cameras = []
    for cam in cameras:
        if cam < 10: str_cameras.append("0" + str(cam))
        else: str_cameras.append(str(cam))
    
    K = len(cameras)
    
    liste_fichiers = []
    for k1 in range(K):
        for k2 in range(k1+1, K):
            liste_fichiers.append((k1, k2, "m_" + str_cameras[k1] + "_" + str_cameras[k2] + ".txt"))
            
    G = nx.Graph()
    
    for k1, k2, fichier in liste_fichiers:
        F = np.loadtxt(fichier)
        for j in range(len(F)):
            G.add_edge((F[j, 0], F[j, 1], k1), (F[j, 2], F[j, 3], k2))
    
    compos_connexes = sorted(nx.connected_components(G), key = len, reverse=True)
    print("Il y a", len(compos_connexes), "composantes connexes. ")
    
    compos_connexes_K = list(filter(lambda a: len(a) == K, compos_connexes))
    print("Il y a", len(compos_connexes_K), "composantes connexes de longueur", K, ". ")
    
    # remplissage des données
    
    corres = []
    for compo in compos_connexes_K:
        cam_utilisees, point = [False] * K, [None] * (2*K)
        for (x, y, k) in compo:
            cam_utilisees[k] = True
            point[k], point[k + K] = x, y
        if cam_utilisees == [True] * K: corres.append(point)
    
    print("Il y a", len(corres), "composantes connexes satisfaisantes. ")
    
    corres = np.asarray(corres)
    
    fichier_corres = "corres"
    for cam in str_cameras:
        fichier_corres += "_" + cam
    fichier_corres += '.txt'
    
    np.savetxt(fichier_corres, corres, fmt='%4.3f')
