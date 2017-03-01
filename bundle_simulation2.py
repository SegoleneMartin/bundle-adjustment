## bundle adjusment

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand, normal
import scipy.linalg as sl

plt.show()

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
    m1, m2, m3 = C2[0]-C1[0], C2[1]-C1[1], C2[2]-C1[2]
    return (m1 * ((ca1*(-sg1*x1+cg1*y1) + sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)) * (sa2*(sg2*x2-cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (ca2*(-sg2*x2+cg2*y2) + sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) * (sa1*(sg1*x1-cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    - m2 * ((cb1*(cg1*x1+sg1*y1) - f*sb1) * (sa2*(sg2*x2-cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (cb2*(cg2*x2+sg2*y2) - f*sb2) * (sa1*(sg1*x1-cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    + m3 * ((cb1*(cg1*x1+sg1*y1) - f*sb1) * (ca2*(-sg2*x2+cg2*y2) + sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (cb2*(cg2*x2+sg2*y2) - f*sb2) * (ca1*(-sg1*x1+cg1*y1) + sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))) / f**2

def deriv_det(C1, C2, theta1, theta2, p1, p2, f):
    """ C1, C2 : coordonnées des centres des caméras, 
    theta1, theta2 : angles de rotation associés aux caméras, 
    p1, p2 : coordonnées des images des points sur les plans images des caméras
    
    Retourne les dérivées partielles du déterminant selon les angles puis les positions 
    dans une matrice D (5, 2) telle qu'on a dans l'ordre et par ligne les dérivées partielles
    par rapport aux alpha_i, beta_i, gamma_i, x_i et y_i """
    
    # on introduit pour simplifier : 
    ca1, ca2 = np.cos(theta1[0]), np.cos(theta2[0])
    sa1, sa2 = np.sin(theta1[0]), np.sin(theta2[0])
    cb1, cb2 = np.cos(theta1[1]), np.cos(theta2[1])
    sb1, sb2 = np.sin(theta1[1]), np.sin(theta2[1])
    cg1, cg2 = np.cos(theta1[2]), np.cos(theta2[2])
    sg1, sg2 = np.sin(theta1[2]), np.sin(theta2[2])
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    m1, m2, m3 = C2[0]-C1[0], C2[1]-C1[1], C2[2]-C1[2]
    k1 = cb1*(cg1*x1+sg1*y1) - f*sb1
    l1 = cb2*(cg2*x2+sg2*y2) - f*sb2
    k2 = ca1*(-sg1*x1+cg1*y1) + sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)
    l2 = ca2*(-sg2*x2+cg2*y2) + sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)
    k3 = sa1*(sg1*x1-cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)
    l3 = sa2*(sg2*x2-cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)
    
    # remplissage de la matrice D
    D = np.zeros((5, 2))
    
    D[0, 0] = (m1 * ((-sa1*(-sg1*x1+cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)) * l3 - l2 * (ca1*(sg1*x1-cg1*y1) - sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    - m2 * (- l1 * (ca1*(sg1*x1-cg1*y1) - sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)))
    + m3 * (- l1 * (-sa1*(-sg1*x1+cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1))))
    
    D[0, 1] = (m1 * (k2 * (ca2*(sg2*x2-cg2*y2) - sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (-sa2*(-sg2*x2+cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) * k3)
    - m2 * (k1 * (ca2*(sg2*x2-cg2*y2) - sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)))
    + m3 * (k1 * (-sa2*(-sg2*x2+cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2))))
    
    D[1, 0] = (m1 * (sa1*(cb1*(cg1*x1+sg1*y1) - f*sb1) * l3 - l2 * (ca1*(cb1*(cg1*x1+sg1*y1) - f*sb1)))
    - m2 * ((-sb1*(cg1*x1+sg1*y1) - f*cb1) * l3 - l1 * (ca1*(cb1*(cg1*x1+sg1*y1) - f*sb1)))
    + m3 * ((-sb1*(cg1*x1+sg1*y1) - f*cb1) * l2 - l1 * (sa1*(cb1*(cg1*x1+sg1*y1) - f*sb1))))
    
    D[1, 1] = (m1 * (k2 * (ca2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (sa2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) * k3)
    - m2 * (k1 * (ca2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (-sb2*(cg2*x2+sg2*y2) - f*cb2) * k3)
    + m3 * (k1 * (sa2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (-sb2*(cg2*x2+sg2*y2) - f*cb2) * k2))
    
    D[2, 0] = (m1 * ((ca1*(-cg1*x1-sg1*y1) + sa1*sb1*(-sg1*x1+cg1*y1)) * l3 - l2 * (sa1*(cg1*x1+sg1*y1) + ca1*sb1*(-sg1*x1+cg1*y1)))
    - m2 * (cb1*(-sg1*x1+cg1*y1) * l3 - l1 * (sa1*(cg1*x1+sg1*y1) + ca1*sb1*(-sg1*x1+cg1*y1)))
    + m3 * (cb1*(-sg1*x1+cg1*y1) * l2 - l1 * (ca1*(-cg1*x1-sg1*y1) + sa1*sb1*(-sg1*x1+cg1*y1))))
    
    D[2, 1] = (m1 * (k2 * (sa2*(cg2*x2+sg2*y2) + ca2*sb2*(-sg2*x2+cg2*y2)) - (ca2*(-cg2*x2-sg2*y2) + sa2*sb2*(-sg2*x2+cg2*y2)) * k3)
    - m2 * (k1 * (sa2*(cg2*x2+sg2*y2) + ca2*sb2*(-sg2*x2+cg2*y2)) - cb2*(-sg2*x2+cg2*y2) * k3)
    + m3 * (k1 * (ca2*(-cg2*x2-sg2*y2) + sa2*sb2*(-sg2*x2+cg2*y2)) - cb2*(-sg2*x2+cg2*y2) * k2))
    
    D[3, 0] = (m1 * ((ca1*-sg1 + sa1*sb1*cg1) * l3 - l2 * (sa1*sg1 + ca1*sb1*cg1))
    - m2 * (cb1*cg1 * l3 - l1 * (sa1*sg1 + ca1*sb1*cg1))
    + m3 * (cb1*cg1 * l2 - l1 * (ca1*-sg1 + sa1*sb1*cg1)))
    
    D[3, 1] = (m1 * (k2 * (sa2*sg2 + ca2*sb2*cg2) - (ca2*-sg2 + sa2*sb2*cg2) * k3)
    - m2 * (k1 * (sa2*sg2 + ca2*sb2*cg2) - cb2*cg2 * k3)
    + m3 * (k1 * (ca2*-sg2 + sa2*sb2*cg2) - cb2*cg2 * k2))
    
    D[4, 0] = (m1 * ((ca1*cg1 + sa1*sb1*sg1) * l3 - l2 * (sa1*-cg1 + ca1*sb1*sg1))
    - m2 * (cb1*sg1 * l3 - l1 * (sa1*-cg1 + ca1*sb1*sg1))
    + m3 * (cb1*sg1 * l2 - l1 * (ca1*cg1 + sa1*sb1*sg1)))
    
    D[4, 1] = (m1 * (k2 * (sa2*-cg2 + ca2*sb2*sg2) - (ca2*cg2 + sa2*sb2*sg2) * k3)
    - m2 * (k1 * (sa2*-cg2 + ca2*sb2*sg2) - cb2*sg2 * k3)
    + m3 * (k1 * (ca2*cg2 + sa2*sb2*sg2) - cb2*sg2 * k2))
    
    return D / f**2

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
        Aj[l, 3*i1] = derivees[0, 0]
        Aj[l, 3*i1+1] = derivees[1, 0]
        Aj[l, 3*i1+2] = derivees[2, 0]
        Aj[l, 3*i2] = derivees[0, 1]
        Aj[l, 3*i2+1] = derivees[1, 1]
        Aj[l, 3*i2+2] = derivees[2, 1]
        Bj[l, 2*i1] = derivees[3, 0]
        Bj[l, 2*i1+1] = derivees[4, 0]
        Bj[l, 2*i2] = derivees[3, 1]
        Bj[l, 2*i2+1] = derivees[4, 1]
    return Aj, Bj

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
    return np.dot(RX(theta[0]), np.dot(RY(theta[1]), RZ(theta[2])))

def matrice_P(C, theta, f):
    R = matrice_R(theta)
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    P = np.zeros((3, 4))
    P[:, :3] = np.dot(K, R)
    P[:, 3] = np.dot(K, R).dot(-C)
    return P

"""
def calcul_D2(C1, C2, theta1, theta2, p1, p2, f):
    ca1, ca2 = np.cos(theta1[0]), np.cos(theta2[0])
    sa1, sa2 = np.sin(theta1[0]), np.sin(theta2[0])
    cb1, cb2 = np.cos(theta1[1]), np.cos(theta2[1])
    sb1, sb2 = np.sin(theta1[1]), np.sin(theta2[1])
    cg1, cg2 = np.cos(theta1[2]), np.cos(theta2[2])
    sg1, sg2 = np.sin(theta1[2]), np.sin(theta2[2])
    #print(ca1, ca2, sa1, sa2, cb1, cb2, sb1, sb2, cg1, cg2, sg1, sg2)
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]
    #print(x1, x2, y1, y2)
    k1 = cb1*(cg1*x1+sg1*y1) - f*sb1
    l1 = cb2*(cg2*x2+sg2*y2) - f*sb2
    k2 = ca1*(-sg1*x1+cg1*y1) + sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)
    l2 = ca2*(-sg2*x2+cg2*y2) + sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)
    k3 = sa1*(sg1*x1-cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)
    l3 = sa2*(sg2*x2-cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)
    #print(k1, l1, k2, l2, k3, l3)
    
    D1=np.array([
    - l1 * (-sa1*(-sg1*x1+cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)),
    
    k1 * (-sa2*(-sg2*x2+cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)),
    
    (-sb1*(cg1*x1+sg1*y1) - f*cb1) * l2 - l1 * (sa1*(cb1*(cg1*x1+sg1*y1) - f*sb1)),
    
    k1 * (sa2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (-sb2*(cg2*x2+sg2*y2) - f*cb2) * k2,
    
    cb1*(-sg1*x1+cg1*y1) * l2 - l1 * (ca1*(-cg1*x1-sg1*y1) + sa1*sb1*(-sg1*x1+cg1*y1)),
    
    k1 * (ca2*(-cg2*x2-sg2*y2) + sa2*sb2*(-sg2*x2+cg2*y2)) - cb2*(-sg2*x2+cg2*y2) * k2,
    
    cb1*cg1 * l2 - l1 * (ca1*-sg1 + sa1*sb1*cg1),
    
    k1 * (ca2*-sg2 + sa2*sb2*cg2) - cb2*cg2 * k2,
    
    cb1*sg1 * l2 - l1 * (ca1*cg1 + sa1*sb1*sg1),
    
    k1 * (ca2*cg2 + sa2*sb2*sg2) - cb2*sg2 * k2
    ])
    
    D2=np.array([
    - l1 * (ca1*(sg1*x1-cg1*y1) - sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)),
    
    k1 * (ca2*(sg2*x2-cg2*y2) - sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)),
    
    (-sb1*(cg1*x1+sg1*y1) - f*cb1) * l3 - l1 * (ca1*(cb1*(cg1*x1+sg1*y1) - f*sb1)),
    
    k1 * (ca2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (-sb2*(cg2*x2+sg2*y2) - f*cb2) * k3,
    
    cb1*(-sg1*x1+cg1*y1) * l3 - l1 * (sa1*(cg1*x1+sg1*y1) + ca1*sb1*(-sg1*x1+cg1*y1)),
    
    k1 * (sa2*(cg2*x2+sg2*y2) + ca2*sb2*(-sg2*x2+cg2*y2)) - cb2*(-sg2*x2+cg2*y2) * k3,
    
    cb1*cg1 * l3 - l1 * (sa1*sg1 + ca1*sb1*cg1),
    
    k1 * (sa2*sg2 + ca2*sb2*cg2) - cb2*cg2 * k3,
    
    cb1*sg1 * l3 - l1 * (sa1*-cg1 + ca1*sb1*sg1),
    
    k1 * (sa2*-cg2 + ca2*sb2*sg2) - cb2*sg2 * k3
    ])
    
    D3=np.array([
    (-sa1*(-sg1*x1+cg1*y1) + ca1*(sb1*(cg1*x1+sg1*y1) + f*cb1)) * l3 - l2 * (ca1*(sg1*x1-cg1*y1) - sa1*(sb1*(cg1*x1+sg1*y1) + f*cb1)),
    
    k2 * (ca2*(sg2*x2-cg2*y2) - sa2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) - (-sa2*(-sg2*x2+cg2*y2) + ca2*(sb2*(cg2*x2+sg2*y2) + f*cb2)) * k3,
    
    sa1*(cb1*(cg1*x1+sg1*y1) - f*sb1) * l3 - l2 * (ca1*(cb1*(cg1*x1+sg1*y1) - f*sb1)),
    
    k2 * (ca2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) - (sa2*(cb2*(cg2*x2+sg2*y2) - f*sb2)) * k3,
    
    (ca1*(-cg1*x1-sg1*y1) + sa1*sb1*(-sg1*x1+cg1*y1)) * l3 - l2 * (sa1*(cg1*x1+sg1*y1) + ca1*sb1*(-sg1*x1+cg1*y1)),
    
    k2 * (sa2*(cg2*x2+sg2*y2) + ca2*sb2*(-sg2*x2+cg2*y2)) - (ca2*(-cg2*x2-sg2*y2) + sa2*sb2*(-sg2*x2+cg2*y2)) * k3,
    
    (ca1*-sg1 + sa1*sb1*cg1) * l3 - l2 * (sa1*sg1 + ca1*sb1*cg1),
    
    k2 * (sa2*sg2 + ca2*sb2*cg2) - (ca2*-sg2 + sa2*sb2*cg2) * k3,
    
    (ca1*cg1 + sa1*sb1*sg1) * l3 - l2 * (sa1*-cg1 + ca1*sb1*sg1),
    
    k2 * (sa2*-cg2 + ca2*sb2*sg2) - (ca2*cg2 + sa2*sb2*sg2) * k3
    ])
    
    D = np.array([D3, -D2, D1]) / f**2
    return(D)
"""










## simulations

# on suppose les points de correspondances situés dans une zone [0, X0] x [0, Y0] x [0, Z0]
# on suppose les caméras situées dans une zone [0, X0] x [0, Y0] x [Z1, Z2] avec Z0 << Z1
# on suppose les angles des caméras variant dans ???

np.random.seed(0)

K = 6
N = 100
f = 12
X0, Y0, Z0, Z1, Z2 = 100000, 100000, 5000, 700000, 800000

C_reel = rand(K, 3)
C_reel[:, 0] *= X0
C_reel[:, 1] *= Y0
C_reel[:, 2] = C_reel[:, 2] * (Z2-Z1) + Z1

theta_reel = rand(K, 3)
theta_reel[:, 0] = theta_reel[:, 0] * np.pi/50 + 99*np.pi/100
theta_reel[:, 1] = (theta_reel[:, 1] - 0.5) * np.pi/100
theta_reel[:, 1:] = 0

X_reel = rand(N, 3)
X_reel[:, 0] *= X0
X_reel[:, 1] *= Y0
X_reel[:, 2] *= Z0

Z_cam = []
for theta in theta_reel:
    Z_cam.append(matrice_R(theta).dot(np.array([0, 0, 1])))

plt.figure().gca(projection = '3d')    # crée une figure 3D
plt.xlabel('X'), plt.ylabel('Y')
plt.plot(X_reel[:, 0], X_reel[:, 1], X_reel[:, 2], marker='+', markersize=3, color='b', label='points', linestyle='None')
plt.plot(C_reel[:, 0], C_reel[:, 1], C_reel[:, 2], marker='+', markersize=10, color='r', label='cameras', linestyle='None')
plt.legend(loc='best')
plt.xlim(0, X0), plt.ylim(0, Y0)

t = 300000
for C, Z in zip(C_reel, Z_cam):
    ZZ = C + t*Z
    plt.plot([C[0], ZZ[0]], [C[1], ZZ[1]], [C[2], ZZ[2]], color='k')    # trace les axes Z_cam

matrices_P = []
for k in range(K):
    matrices_P.append(matrice_P(C_reel[k], theta_reel[k], f))

x_reel = np.zeros((N, K, 2))
for j in range(N):
    for i in range(K):
        point = matrices_P[i].dot(np.array([X_reel[j, 0], X_reel[j, 1], X_reel[j, 2], 1]))
        x_reel[j, i] = point[:-1] / point[-1]

# perturbation des coordonnées et des angles

sigma_x = 0.0001
sigma_theta = 0.0001

x_0 = x_reel + normal(0, sigma_x, (N, K, 2))
theta_0 = theta_reel + normal(0, sigma_theta, (K, 3))

# minimisation de F

A, B = matrices_A_B(C_reel, theta_0, x_0, f)
M = np.concatenate((A, B), axis=1)
piM = np.linalg.pinv(M)

erreur = piM.dot(-F(C_reel, theta_0, x_0, f))


theta_a = theta_0 + erreur[:3*K].reshape((K, 3))
x_a = np.zeros((N, K, 2))
for j in range(N):
    for i in range(K):
        x_a[j, i, 0] = x_0[j, i, 0] + erreur[3*K + 2*K*j + 2*i]
        x_a[j, i, 1] = x_0[j, i, 0] + erreur[3*K + 2*K*j + 2*i + 1]

n_0 = [sl.norm(x_reel[:, :, 0]-x_0[:, :, 0]), sl.norm(x_reel[:, :, 1]-x_0[:, :, 1]), sl.norm(theta_reel-theta_0)]
n_a = [sl.norm(x_reel[:, :, 0]-x_a[:, :, 0]), sl.norm(x_reel[:, :, 1]-x_a[:, :, 1]),sl.norm(theta_reel-theta_a)] 



















"""
## test

K, N, f = 6, 20, 10
C = (rand(K, 3) - 0.5) * 20     # tirage au sort des coordonnées entre -10 et 10
theta = rand(K, 3) * 2 * np.pi  # tirage au sort des coordonnées des angles entre 0 et 2pi
x = (rand(N, K, 2) - 0.5) * 10  # tirage au sort des coordonnées entre -5 et 5

A, B = matrices_A_B(C, theta, x, f)

M = np.c_[A, B]
sigmas = sl.svd(M)[1]

# étude du conditionnement (aléatoire) en fonction de N : 

K = 6
NN, VSmax, VSmin, CC = [], [], [], []
for i in range(9):
    N = 50 * (i+1)
    print(N)
    C = (rand(K, 3) - 0.5) * 20     # tirage au sort des coordonnées entre -10 et 10
    theta = rand(K, 3) * 2 * np.pi  # tirage au sort des coordonnées des angles entre 0 et 2pi
    x = (rand(N, K, 2) - 0.5) * 10  # tirage au sort des coordonnées entre -5 et 5
    A, B = matrices_A_B(C, theta, x, f)
    M = np.c_[A, B]
    VS = sl.svd(M)[1]
    NN.append(N), VSmax.append(VS[0]), VSmin.append(VS[-1]), CC.append(VS[0]/VS[-1])

F, (fig1, fig2, fig3) = plt.subplots(3, sharex=True)

fig1.plot(NN, VSmin, marker='+')
fig1.set_ylabel('VS MINIMALE')

fig2.plot(NN, VSmax, marker='+')
fig2.set_ylabel('VS MAXIMALE')

fig3.plot(NN, CC, marker='+')
fig3.set_ylabel('CONDITIONNEMENT')
fig3.set_xlabel('N')
"""