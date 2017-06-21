import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
import scipy.linalg as sl
from numpy import cos, sin
from math import atan2
import cv2
from scipy.linalg import solve, rq
from numpy.linalg import det, inv, solve
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
from scipy.sparse import csr_matrix, coo_matrix



## fonctions
def det(C1, C2, theta1, theta2, p1, p2, f):
    """ C1, C2 : coordonnées des centres des caméras, 
    theta1, theta2 : angles de rotation associés aux caméras, 
    p1, p2 : coordonnées des images des points sur les plans images des caméras
    
    Retourne la valeur du déterminant associée aux paramètres """
    
  
    ca1, ca2 = np.cos(theta1[0]), np.cos(theta2[0])
    sa1, sa2 = np.sin(theta1[0]), np.sin(theta2[0])
    cb1, cb2 = np.cos(theta1[1]), np.cos(theta2[1])
    sb1, sb2 = np.sin(theta1[1]), np.sin(theta2[1])
    cg1, cg2 = np.cos(theta1[2]), np.cos(theta2[2])
    sg1, sg2 = np.sin(theta1[2]), np.sin(theta2[2])
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    
    #attention: ici on n'a pas multiplié par f**2 comme dans le latex
    
    D1=-((cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f + (y1*(ca1*cg1 + sa1*sb1*sg1))/f)*     ((x2*cb2*cg2)/f - sb2 + (y2*cb2*sg2)/f)) +   ((x1*cb1*cg1)/f - sb1 + (y1*cb1*sg1)/f)*   (cb2*sa2 + (x2*(cg2*sa2*sb2 - ca2*sg2))/f +      (y2*(ca2*cg2 + sa2*sb2*sg2))/f)
    
    D2=-((ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f +        (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)*     ((x2*cb2*cg2)/f - sb2 + (y2*cb2*sg2)/f)) +   ((x1*cb1*cg1)/f - sb1 + (y1*cb1*sg1)/f)*   (ca2*cb2 + (x2*(ca2*cg2*sb2 + sa2*sg2))/f +      (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f)
    
    D3=(cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f +      (y1*(ca1*cg1 + sa1*sb1*sg1))/f)*   (ca2*cb2 + (x2*(ca2*cg2*sb2 + sa2*sg2))/f +      (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f) -   (ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f +      (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)*   (cb2*sa2 + (x2*(cg2*sa2*sb2 - ca2*sg2))/f +      (y2*(ca2*cg2 + sa2*sb2*sg2))/f)
    
    
    m1, m2, m3 = C2[0]-C1[0], C2[1]-C1[1], C2[2]-C1[2]
    return(m1*D3-m2*D2+m3*D1)
    
    
def deriv_det(C1, C2, theta1, theta2, p1, p2, f):
    
    
    #attention: ici on n'a pas multiplié par f**2 comme dans le latex
    ca1, ca2 = np.cos(theta1[0]), np.cos(theta2[0])
    sa1, sa2 = np.sin(theta1[0]), np.sin(theta2[0])
    cb1, cb2 = np.cos(theta1[1]), np.cos(theta2[1])
    sb1, sb2 = np.sin(theta1[1]), np.sin(theta2[1])
    cg1, cg2 = np.cos(theta1[2]), np.cos(theta2[2])
    sg1, sg2 = np.sin(theta1[2]), np.sin(theta2[2])
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    
    
    D1=np.array([
    
    -((ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f + (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)*((x2*cb2*cg2)/f - sb2 + (y2*cb2*sg2)/f)),
    
    
    ((x1*cb1*cg1)/f - sb1 + (y1*cb1*sg1)/f)*(ca2*cb2 +(x2*(ca2*cg2*sb2 + sa2*sg2))/f + (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f),
    
    
    -(((x1*cb1*cg1*sa1)/f - sa1*sb1 +  (y1*cb1*sa1*sg1)/f)* ((x2*cb2*cg2)/f - sb2 + (y2*cb2*sg2)/f)) + (-cb1 - (x1*cg1*sb1)/f -(y1*sb1*sg1)/f)* (cb2*sa2 + (x2*(cg2*sa2*sb2 - ca2*sg2))/f +  (y2*(ca2*cg2 + sa2*sb2*sg2))/f),
    
    
    ((x1*cb1*cg1)/f - sb1 + (y1*cb1*sg1)/f)* ((x2*cb2*cg2*sa2)/f - sa2*sb2 +  (y2*cb2*sa2*sg2)/f) - (cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f +  (y1*(ca1*cg1 + sa1*sb1*sg1))/f)* (-cb2 - (x2*cg2*sb2)/f - (y2*sb2*sg2)/f),
    
    
    -(((y1*(cg1*sa1*sb1 - ca1*sg1))/f +  (x1*(-(ca1*cg1) - sa1*sb1*sg1))/f)* ((x2*cb2*cg2)/f - sb2 + (y2*cb2*sg2)/f)) +((y1*cb1*cg1)/f - (x1*cb1*sg1)/f)* (cb2*sa2 + (x2*(cg2*sa2*sb2 - ca2*sg2))/f +  (y2*(ca2*cg2 + sa2*sb2*sg2))/f),
    
    
    -((cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f +  (y1*(ca1*cg1 + sa1*sb1*sg1))/f)* ((y2*cb2*cg2)/f - (x2*cb2*sg2)/f)) +((x1*cb1*cg1)/f - sb1 + (y1*cb1*sg1)/f)* ((y2*(cg2*sa2*sb2 -ca2*sg2))/f +  (x2*(-(ca2*cg2) - sa2*sb2*sg2))/f),
    
    
    -(((cg1*sa1*sb1 - ca1*sg1)* ((x2*cb2*cg2)/f - sb2 + (y2*cb2*sg2)/f))/f) + (cb1*cg1*(cb2*sa2 +  (x2*(cg2*sa2*sb2 - ca2*sg2))/f +  (y2*(ca2*cg2 + sa2*sb2*sg2))/f))/f,
    
    
    -((cb2*cg2*(cb1*sa1 +  (x1*(cg1*sa1*sb1 - ca1*sg1))/f +  (y1*(ca1*cg1 + sa1*sb1*sg1))/f))/f) + (((x1*cb1*cg1)/f - sb1 +(y1*cb1*sg1)/f)* (cg2*sa2*sb2 - ca2*sg2))/f,
    
    
    -(((ca1*cg1 + sa1*sb1*sg1)* ((x2*cb2*cg2)/f - sb2 +(y2*cb2*sg2)/f))/f) + (cb1*sg1*(cb2*sa2 +  (x2*(cg2*sa2*sb2 - ca2*sg2))/f +  (y2*(ca2*cg2 + sa2*sb2*sg2))/f))/f,
    
    
    -((cb2*(cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f + (y1*(ca1*cg1 + sa1*sb1*sg1))/f)*sg2)/f) + (((x1*cb1*cg1)/f - sb1 +(y1*cb1*sg1)/f)* (ca2*cg2 + sa2*sb2*sg2))/f
    
    ])
    
    
    
    
    D2=np.array([
    -((-(cb1*sa1) + (x1*(-(cg1*sa1*sb1) + ca1*sg1))/ f +(y1*(-(ca1*cg1) - sa1*sb1*sg1))/f)*((x2*cb2*cg2)/f - sb2 +(y2*cb2*sg2)/f)),
    
    
    ((x1*cb1*cg1)/f - sb1 + (y1*cb1*sg1)/f)*(-(cb2*sa2) +(x2*(-(cg2*sa2*sb2) + ca2*sg2))/ f + (y2*(-(ca2*cg2) - sa2*sb2*sg2))/f),
    
    
    -(((x1*ca1*cb1*cg1)/f - ca1*sb1 +  (y1*ca1*cb1*sg1)/f)*((x2*cb2*cg2)/f - sb2 + (y2*cb2*sg2)/f)) + (-cb1 - (x1*cg1*sb1)/f - (y1*sb1*sg1)/f)* (ca2*cb2 + (x2*(ca2*cg2*sb2 + sa2*sg2))/f +  (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f),
    
    
    ((x1*cb1*cg1)/f - sb1 + (y1*cb1*sg1)/f)* ((x2*ca2*cb2*cg2)/f -ca2*sb2 +  (y2*ca2*cb2*sg2)/f) - (ca1*cb1 + (x1*(ca1*cg1*sb1 +sa1*sg1))/f +  (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)* (-cb2 -(x2*cg2*sb2)/f - (y2*sb2*sg2)/f),
    
    
    -(((y1*(ca1*cg1*sb1 + sa1*sg1))/f +  (x1*(cg1*sa1 - ca1*sb1*sg1))/f)* ((x2*cb2*cg2)/f - sb2 + (y2*cb2*sg2)/f)) +((y1*cb1*cg1)/f - (x1*cb1*sg1)/f)* (ca2*cb2 + (x2*(ca2*cg2*sb2 +sa2*sg2))/f +  (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f),
    
    
    -((ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f +  (y1*(-(cg1*sa1) +ca1*sb1*sg1))/f)* ((y2*cb2*cg2)/f - (x2*cb2*sg2)/f)) +((x1*cb1*cg1)/f - sb1 + (y1*cb1*sg1)/f)* ((y2*(ca2*cg2*sb2 +sa2*sg2))/f +  (x2*(cg2*sa2 - ca2*sb2*sg2))/f),
    
    
    -(((ca1*cg1*sb1 + sa1*sg1)* ((x2*cb2*cg2)/f - sb2 +(y2*cb2*sg2)/f))/f) + (cb1*cg1*(ca2*cb2 +  (x2*(ca2*cg2*sb2 +sa2*sg2))/f +  (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f))/f,
    
    
    -((cb2*cg2*(ca1*cb1 +  (x1*(ca1*cg1*sb1 + sa1*sg1))/f +  (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f))/f) + (((x1*cb1*cg1)/f - sb1 +(y1*cb1*sg1)/f)* (ca2*cg2*sb2 + sa2*sg2))/f,
    
    
    -(((-(cg1*sa1) + ca1*sb1*sg1)* ((x2*cb2*cg2)/f - sb2 +(y2*cb2*sg2)/f))/f) + (cb1*sg1*(ca2*cb2 +  (x2*(ca2*cg2*sb2 + sa2*sg2))/f +  (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f))/f,
    
    
    
    -((cb2*(ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f +  (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)*sg2)/f) + (((x1*cb1*cg1)/f -sb1 + (y1*cb1*sg1)/f)* (-(cg2*sa2) + ca2*sb2*sg2))/f
    
    ])
    
    
    
    
    D3=np.array([
    
    
    (ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f + (y1*(-(cg1*sa1) +ca1*sb1*sg1))/f)* (ca2*cb2 + (x2*(ca2*cg2*sb2 + sa2*sg2))/f +(y2*(-(cg2*sa2) + ca2*sb2*sg2))/f) - (-(cb1*sa1) +(x1*(-(cg1*sa1*sb1) + ca1*sg1))/f + (y1*(-(ca1*cg1) -sa1*sb1*sg1))/f)* (cb2*sa2 + (x2*(cg2*sa2*sb2 - ca2*sg2))/f +  (y2*(ca2*cg2 + sa2*sb2*sg2))/f),
    
    
    -((ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f +  (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)* (ca2*cb2 + (x2*(ca2*cg2*sb2 + sa2*sg2))/f +  (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f)) + (cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f +  (y1*(ca1*cg1 + sa1*sb1*sg1))/f)* (-(cb2*sa2) + (x2*(-(cg2*sa2*sb2) + ca2*sg2))/f + (y2*(-(ca2*cg2) - sa2*sb2*sg2))/f),
    
    
    ((x1*cb1*cg1*sa1)/f - sa1*sb1 +  (y1*cb1*sa1*sg1)/f)* (ca2*cb2+ (x2*(ca2*cg2*sb2 + sa2*sg2))/f +  (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f) - ((x1*ca1*cb1*cg1)/f - ca1*sb1 + (y1*ca1*cb1*sg1)/f)* (cb2*sa2 + (x2*(cg2*sa2*sb2 - ca2*sg2))/f + (y2*(ca2*cg2 + sa2*sb2*sg2))/f),
    
    
    (cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f +  (y1*(ca1*cg1 +sa1*sb1*sg1))/f)* ((x2*ca2*cb2*cg2)/f - ca2*sb2 +  (y2*ca2*cb2*sg2)/f) - (ca1*cb1 + (x1*(ca1*cg1*sb1 +sa1*sg1))/f +  (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)* ((x2*cb2*cg2*sa2)/f -sa2*sb2 +  (y2*cb2*sa2*sg2)/f),
    
    
    ((y1*(cg1*sa1*sb1 - ca1*sg1))/f +  (x1*(-(ca1*cg1) - sa1*sb1*sg1))/f)* (ca2*cb2 + (x2*(ca2*cg2*sb2 + sa2*sg2))/f +  (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f) - ((y1*(ca1*cg1*sb1 + sa1*sg1))/f +  (x1*(cg1*sa1 - ca1*sb1*sg1))/f)* (cb2*sa2 +(x2*(cg2*sa2*sb2 - ca2*sg2))/f +  (y2*(ca2*cg2 + sa2*sb2*sg2))/f),
    
    
    (cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f +  (y1*(ca1*cg1 + sa1*sb1*sg1))/f)* ((y2*(ca2*cg2*sb2 + sa2*sg2))/f +(x2*(cg2*sa2 - ca2*sb2*sg2))/f) - (ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f +  (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)* ((y2*(cg2*sa2*sb2 - ca2*sg2))/f +  (x2*(-(ca2*cg2) - sa2*sb2*sg2))/f),
    
    
    ((cg1*sa1*sb1 - ca1*sg1)* (ca2*cb2 + (x2*(ca2*cg2*sb2 + sa2*sg2))/f + (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f))/f -((ca1*cg1*sb1 + sa1*sg1)* (cb2*sa2 + (x2*(cg2*sa2*sb2 - ca2*sg2))/f +  (y2*(ca2*cg2 + sa2*sb2*sg2))/f))/f,
    
    
    -(((ca1*cb1 + (x1*(ca1*cg1*sb1 + sa1*sg1))/f +  (y1*(-(cg1*sa1) + ca1*sb1*sg1))/f)* (cg2*sa2*sb2 - ca2*sg2))/f) + ((cb1*sa1 +(x1*(cg1*sa1*sb1 - ca1*sg1))/f +  (y1*(ca1*cg1 +sa1*sb1*sg1))/f)* (ca2*cg2*sb2 + sa2*sg2))/f,
    
    
    ((ca1*cg1 + sa1*sb1*sg1)* (ca2*cb2 + (x2*(ca2*cg2*sb2 + sa2*sg2))/f + (y2*(-(cg2*sa2) + ca2*sb2*sg2))/f))/f - ((-(cg1*sa1) + ca1*sb1*sg1)* (cb2*sa2 + (x2*(cg2*sa2*sb2 - ca2*sg2))/f +  (y2*(ca2*cg2 + sa2*sb2*sg2))/f))/f,
    
    

    ((cb1*sa1 + (x1*(cg1*sa1*sb1 - ca1*sg1))/f +  (y1*(ca1*cg1 + sa1*sb1*sg1))/f)* (-(cg2*sa2) + ca2*sb2*sg2))/f - ((ca1*cb1 +(x1*(ca1*cg1*sb1 + sa1*sg1))/f +  (y1*(-(cg1*sa1) +ca1*sb1*sg1))/f)* (ca2*cg2 + sa2*sb2*sg2))/f
    
    ])

    D=np.array([D3,-D2,D1])
    
    centre = C2-C1   #matrice 1x3
    return(centre.dot(D))


def genere_liste_couples(K):
    """ crée la liste des couples (i1, i2) pour i1<i2 entre 0 et K-1 """
    liste_couples = []
    for i1 in range(K-1):
        for i2 in range(i1+1, K):
            liste_couples.append((i1, i2))
    return liste_couples

def matrices_Aj_Bj(C, theta, x, j, f, liste_couples):
    """ Calcule les matrices Aj et Bj associées au j-ième point de x """
    K = C.shape[0]
    Aj = np.zeros((K*(K-1)/2, 3*K))
    Bj = np.zeros((K*(K-1)/2, 2*K))
    for (l, (i1, i2)) in enumerate(liste_couples):
        derivees = deriv_det(C[i1], C[i2], theta[i1], theta[i2], x[j, i1], x[j, i2], f)
        Aj[l, 3*i1] = derivees[0]
        Aj[l, 3*i1+1] = derivees[2]
        Aj[l, 3*i1+2] = derivees[4]
        Aj[l, 3*i2] = derivees[1]
        Aj[l, 3*i2+1] = derivees[3]
        Aj[l, 3*i2+2] = derivees[5]
        Bj[l, 2*i1] = derivees[6]
        Bj[l, 2*i1+1] = derivees[8]
        Bj[l, 2*i2] = derivees[7]
        Bj[l, 2*i2+1] = derivees[9]
    return Aj, Bj
    
    
def matrices_Aj(C, theta, x, j, f, liste_couples):
    """ Calcule les matrices Aj et Bj associées au j-ième point de x """
    K = C.shape[0]
    Aj = np.zeros((K*(K-1)/2, 3*K))
    for (l, (i1, i2)) in enumerate(liste_couples):
        derivees = deriv_det(C[i1], C[i2], theta[i1], theta[i2], x[j, i1], x[j, i2], f)
        Aj[l, 3*i1] = derivees[0]
        Aj[l, 3*i1+1] = derivees[2]
        Aj[l, 3*i1+2] = derivees[4]
        Aj[l, 3*i2] = derivees[1]
        Aj[l, 3*i2+1] = derivees[3]
        Aj[l, 3*i2+2] = derivees[5]
    return Aj

    
def matrices_A_B(C, theta, x, f):
    """ C : ensemble des coordonnées des K caméras
    theta : ensemble des angles de rotations associés aux caméras
    x : ensemble des coordonnées des images sur les K caméras des N points
    (array de taille (N, K, 2))
    
    Calcule les matrices A et B """
    
    K, N = C.shape[0], x.shape[0]
    A = np.zeros((K*(K-1)/2*N, 3*K))
    B = np.zeros((K*(K-1)/2*N, 2*K*N))
    liste_couples = genere_liste_couples(K)
    for j in range(N):
        A[K*(K-1)/2*j:K*(K-1)/2*(j+1)], B[K*(K-1)/2*j:K*(K-1)/2*(j+1), 2*K*j:2*K*(j+1)] = matrices_Aj_Bj(C, theta, x, j, f, liste_couples)
    return A, B

def matrices_A_B_sparse(C, theta, x, f):
    """ C : ensemble des coordonnées des K caméras
    theta : ensemble des angles de rotations associés aux caméras
    x : ensemble des coordonnées des images sur les K caméras des N points
    (array de taille (N, K, 2))
    
    Calcule les matrices A et B """

    K, N = C.shape[0], x.shape[0]
    A = []
    diag_B = []
    liste_couples = genere_liste_couples(K)
    for j in range(N):
        Aj, Bj=matrices_Aj_Bj(C, theta, x, j, f, liste_couples)
        if j==0:
            A=Aj
        else:
            A=np.concatenate((A, Aj), axis=0)
        diag_B.append(Bj)
    B=ss.block_diag(diag_B)
    return coo_matrix(A), B
    
def matrice_A(C, theta, x, f):
    """ C : ensemble des coordonnées des K caméras
    theta : ensemble des angles de rotations associés aux caméras
    x : ensemble des coordonnées des images sur les K caméras des N points
    (array de taille (N, K, 2))
    
    Calcule les matrices A et B """
    
    K, N = C.shape[0], x.shape[0]
    A = []
    liste_couples = genere_liste_couples(K)
    for j in range(N):
        Aj = matrices_Aj(C, theta, x, j, f, liste_couples)
        if j == 0:
            A = Aj
        else:
            A = np.concatenate((A, Aj), axis=0)
    return A



def F(C, theta, x, f):
    N, K = x.shape[:-1]
    liste_couples = genere_liste_couples(K)
    FF = []
    for j in range(N):
        for (i1, i2) in liste_couples:
            FF.append(det(C[i1], C[i2], theta[i1], theta[i2], x[j, i1], x[j, i2], f))
    return np.array(FF)

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

def triangulation(matrices_P, x): 
    # retrouve les positions des points 3D à partir des matrices de projection et des coordonnées des points
    # C : centres des cameras, matrice (6,3)
    # matrices_P : liste des matrices de projection, matrice (K,3,4)
    # x : coordonées des points dans les images, matrice (N,K,2) 
    # 
    # Principe: pour un point de l espace donné, pour chaque paire de camera, on trouve une estimation X_i1,i2 puis on fait la moyenne des estimations trouvées.
    N, K = x.shape[:2]
    liste_couples=genere_liste_couples(K)
    Points_camera=[]
    for (i1, i2) in liste_couples:
        projPoints1=x[:,i1,:].ravel().reshape(2,N,order='F')
        projPoints2=x[:,i2,:].ravel().reshape(2,N,order='F')
        p=cv2.triangulatePoints(matrices_P[i1], matrices_P[i2], projPoints1, projPoints2)
        p=p/p[3,:]
        Points_camera.append(p)
    Points_camera=np.array(Points_camera)
    return(np.average(Points_camera, axis=0))
    
    
def kr_from_p(P):
    """ Extract K, R and C from a camera matrix P, such that P = K*R*[eye(3) | -C]. 
    
    K is scaled so that K[2, 2] = 1 and K[0, 0] > 0. 
    
    """
    
    K, R = rq(P[:, :3])
    
    K /= K[2, 2]
    if K[0, 0] < 0:
        D = np.diag([-1, -1, 1])
        K = np.dot(K, D)
        R = np.dot(D, R)
    
    C = -np.linalg.solve(P[:, :3], P[:, 3])
    
    test = np.dot(np.dot(K, R), np.concatenate((np.eye(3), -np.array([C]).T), axis=1))
    np.testing.assert_allclose(test / test[2, 3], P / P[2, 3])
    
    return C, R, K
    

def theta_from_r(R):  # extraire theta à partir de R
    a = atan2(R[2, 1], R[2, 2])
    b = atan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    g = atan2(R[1, 0], R[0, 0])
    
    return a, b, g
        

    
    
    
    
    