import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy import cos, sin


def calcul_D(x, theta, j, k ,l):
    
    a1=theta[k,0]
    a2=theta[l,1]
    b1=theta[k,1]
    b2=theta[l,1]
    g1=theta[k,2]
    g2=theta[l,2]
    
    x1=x[j,k,0]
    x2=x[j,l,0]
    y1=x[j,k,1]
    y2=x[j,l,1]
    
    #attention: ici on n'a pas multiplié par f**2 comme dans le latex
    D1=np.array([
    
    -((cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f + (y1*(-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1)))/f)*((x2*cos(b2)*cos(g2))/f - sin(b2) + (y2*cos(b2)*sin(g2))/f)),
    
    
    ((x1*cos(b1)*cos(g1))/f - sin(b1) + (y1*cos(b1)*sin(g1))/f)*(cos(a2)*cos(b2) +(x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f + (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f),
    
    
    -(((x1*cos(b1)*cos(g1)*sin(a1))/f - sin(a1)*sin(b1) +  (y1*cos(b1)*sin(a1)*sin(g1))/f)* ((x2*cos(b2)*cos(g2))/f - sin(b2) + (y2*cos(b2)*sin(g2))/f)) + (-cos(b1) - (x1*cos(g1)*sin(b1))/f -(y1*sin(b1)*sin(g1))/f)* (cos(b2)*sin(a2) + (x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    ((x1*cos(b1)*cos(g1))/f - sin(b1) + (y1*cos(b1)*sin(g1))/f)* ((x2*cos(b2)*cos(g2)*sin(a2))/f - sin(a2)*sin(b2) +  (y2*cos(b2)*sin(a2)*sin(g2))/f) - (cos(b1)*sin(a1) + (x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (y1*(cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1)))/f)* (-cos(b2) - (x2*cos(g2)*sin(b2))/f - (y2*sin(b2)*sin(g2))/f),
    
    
    -(((y1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (x1*(-(cos(a1)*cos(g1)) - sin(a1)*sin(b1)*sin(g1)))/f)* ((x2*cos(b2)*cos(g2))/f - sin(b2) + (y2*cos(b2)*sin(g2))/f)) +((y1*cos(b1)*cos(g1))/f - (x1*cos(b1)*sin(g1))/f)* (cos(b2)*sin(a2) + (x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    -((cos(b1)*sin(a1) + (x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (y1*(cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1)))/f)* ((y2*cos(b2)*cos(g2))/f - (x2*cos(b2)*sin(g2))/f)) +((x1*cos(b1)*cos(g1))/f - sin(b1) + (y1*cos(b1)*sin(g1))/f)* ((y2*(cos(g2)*sin(a2)*sin(b2) -cos(a2)*sin(g2)))/f +  (x2*(-(cos(a2)*cos(g2)) - sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    -(((cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1))* ((x2*cos(b2)*cos(g2))/f - sin(b2) + (y2*cos(b2)*sin(g2))/f))/f) + (cos(b1)*cos(g1)*(cos(b2)*sin(a2) +  (x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f))/f,
    
    
    -(((cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1))* ((x2*cos(b2)*cos(g2))/f - sin(b2) +(y2*cos(b2)*sin(g2))/f))/f) + (cos(b1)*sin(g1)*(cos(b2)*sin(a2) +  (x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f))/f,
    
    
    -((cos(b2)*cos(g2)*(cos(b1)*sin(a1) +  (x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (y1*(cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1)))/f))/f) + (((x1*cos(b1)*cos(g1))/f - sin(b1) +(y1*cos(b1)*sin(g1))/f)* (cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f,
    
    
    -((cos(b2)*(cos(b1)*sin(a1) + (x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f + (y1*(cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1)))/f)*sin(g2))/f) + (((x1*cos(b1)*cos(g1))/f - sin(b1) +(y1*cos(b1)*sin(g1))/f)* (cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f
    
    ])
    
    
    
    
    D2=np.array([
    -((-(cos(b1)*sin(a1)) + (x1*(-(cos(g1)*sin(a1)*sin(b1)) + cos(a1)*sin(g1)))/ f +(y1*(-(cos(a1)*cos(g1)) - sin(a1)*sin(b1)*sin(g1)))/f)*((x2*cos(b2)*cos(g2))/f - sin(b2) +(y2*cos(b2)*sin(g2))/f)),
    
    
    ((x1*cos(b1)*cos(g1))/f - sin(b1) + (y1*cos(b1)*sin(g1))/f)*(-(cos(b2)*sin(a2)) +(x2*(-(cos(g2)*sin(a2)*sin(b2)) + cos(a2)*sin(g2)))/ f + (y2*(-(cos(a2)*cos(g2)) - sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    -(((x1*cos(a1)*cos(b1)*cos(g1))/f - cos(a1)*sin(b1) +  (y1*cos(a1)*cos(b1)*sin(g1))/f)*((x2*cos(b2)*cos(g2))/f - sin(b2) + (y2*cos(b2)*sin(g2))/f)) + (-cos(b1) - (x1*cos(g1)*sin(b1))/f - (y1*sin(b1)*sin(g1))/f)* (cos(a2)*cos(b2) + (x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f +  (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f),
    
    
    ((x1*cos(b1)*cos(g1))/f - sin(b1) + (y1*cos(b1)*sin(g1))/f)* ((x2*cos(a2)*cos(b2)*cos(g2))/f -cos(a2)*sin(b2) +  (y2*cos(a2)*cos(b2)*sin(g2))/f) - (cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) +sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1)))/f)* (-cos(b2) -(x2*cos(g2)*sin(b2))/f - (y2*sin(b2)*sin(g2))/f),
    
    
    -(((y1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (x1*(cos(g1)*sin(a1) - cos(a1)*sin(b1)*sin(g1)))/f)* ((x2*cos(b2)*cos(g2))/f - sin(b2) + (y2*cos(b2)*sin(g2))/f)) +((y1*cos(b1)*cos(g1))/f - (x1*cos(b1)*sin(g1))/f)* (cos(a2)*cos(b2) + (x2*(cos(a2)*cos(g2)*sin(b2) +sin(a2)*sin(g2)))/f +  (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f),
    
    
    -((cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) +cos(a1)*sin(b1)*sin(g1)))/f)* ((y2*cos(b2)*cos(g2))/f - (x2*cos(b2)*sin(g2))/f)) +((x1*cos(b1)*cos(g1))/f - sin(b1) + (y1*cos(b1)*sin(g1))/f)* ((y2*(cos(a2)*cos(g2)*sin(b2) +sin(a2)*sin(g2)))/f +  (x2*(cos(g2)*sin(a2) - cos(a2)*sin(b2)*sin(g2)))/f),
    
    
    -(((cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1))* ((x2*cos(b2)*cos(g2))/f - sin(b2) +(y2*cos(b2)*sin(g2))/f))/f) + (cos(b1)*cos(g1)*(cos(a2)*cos(b2) +  (x2*(cos(a2)*cos(g2)*sin(b2) +sin(a2)*sin(g2)))/f +  (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f))/f,
    
    
    -(((-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1))* ((x2*cos(b2)*cos(g2))/f - sin(b2) +(y2*cos(b2)*sin(g2))/f))/f) + (cos(b1)*sin(g1)*(cos(a2)*cos(b2) +  (x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f +  (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f))/f,
    
    
    -((cos(b2)*cos(g2)*(cos(a1)*cos(b1) +  (x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1)))/f))/f) + (((x1*cos(b1)*cos(g1))/f - sin(b1) +(y1*cos(b1)*sin(g1))/f)* (cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f,
    
    
    -((cos(b2)*(cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1)))/f)*sin(g2))/f) + (((x1*cos(b1)*cos(g1))/f -sin(b1) + (y1*cos(b1)*sin(g1))/f)* (-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f
    
    ])
    
    
    
    
    D3=np.array([
    
    
    (cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f + (y1*(-(cos(g1)*sin(a1)) +cos(a1)*sin(b1)*sin(g1)))/f)* (cos(a2)*cos(b2) + (x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f +(y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f) - (-(cos(b1)*sin(a1)) +(x1*(-(cos(g1)*sin(a1)*sin(b1)) + cos(a1)*sin(g1)))/f + (y1*(-(cos(a1)*cos(g1)) -sin(a1)*sin(b1)*sin(g1)))/f)* (cos(b2)*sin(a2) + (x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    -((cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1)))/f)* (cos(a2)*cos(b2) + (x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f +  (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f)) + (cos(b1)*sin(a1) + (x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (y1*(cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1)))/f)* (-(cos(b2)*sin(a2)) + (x2*(-(cos(g2)*sin(a2)*sin(b2)) + cos(a2)*sin(g2)))/f + (y2*(-(cos(a2)*cos(g2)) - sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    ((x1*cos(b1)*cos(g1)*sin(a1))/f - sin(a1)*sin(b1) +  (y1*cos(b1)*sin(a1)*sin(g1))/f)* (cos(a2)*cos(b2)+ (x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f +  (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f) - ((x1*cos(a1)*cos(b1)*cos(g1))/f - cos(a1)*sin(b1) + (y1*cos(a1)*cos(b1)*sin(g1))/f)* (cos(b2)*sin(a2) + (x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f + (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    (cos(b1)*sin(a1) + (x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (y1*(cos(a1)*cos(g1) +sin(a1)*sin(b1)*sin(g1)))/f)* ((x2*cos(a2)*cos(b2)*cos(g2))/f - cos(a2)*sin(b2) +  (y2*cos(a2)*cos(b2)*sin(g2))/f) - (cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) +sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1)))/f)* ((x2*cos(b2)*cos(g2)*sin(a2))/f -sin(a2)*sin(b2) +  (y2*cos(b2)*sin(a2)*sin(g2))/f),
    
    
    ((y1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (x1*(-(cos(a1)*cos(g1)) - sin(a1)*sin(b1)*sin(g1)))/f)* (cos(a2)*cos(b2) + (x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f +  (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f) - ((y1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (x1*(cos(g1)*sin(a1) - cos(a1)*sin(b1)*sin(g1)))/f)* (cos(b2)*sin(a2) +(x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    (cos(b1)*sin(a1) + (x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (y1*(cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1)))/f)* ((y2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f +(x2*(cos(g2)*sin(a2) - cos(a2)*sin(b2)*sin(g2)))/f) - (cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1)))/f)* ((y2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (x2*(-(cos(a2)*cos(g2)) - sin(a2)*sin(b2)*sin(g2)))/f),
    
    
    ((cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1))* (cos(a2)*cos(b2) + (x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f + (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f))/f -((cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1))* (cos(b2)*sin(a2) + (x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f))/f,
    
    
    ((cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1))* (cos(a2)*cos(b2) + (x2*(cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f + (y2*(-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f))/f - ((-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1))* (cos(b2)*sin(a2) + (x2*(cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f +  (y2*(cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f))/f,
    
    
    -(((cos(a1)*cos(b1) + (x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) + cos(a1)*sin(b1)*sin(g1)))/f)* (cos(g2)*sin(a2)*sin(b2) - cos(a2)*sin(g2)))/f) + ((cos(b1)*sin(a1) +(x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (y1*(cos(a1)*cos(g1) +sin(a1)*sin(b1)*sin(g1)))/f)* (cos(a2)*cos(g2)*sin(b2) + sin(a2)*sin(g2)))/f,
    
    
    ((cos(b1)*sin(a1) + (x1*(cos(g1)*sin(a1)*sin(b1) - cos(a1)*sin(g1)))/f +  (y1*(cos(a1)*cos(g1) + sin(a1)*sin(b1)*sin(g1)))/f)* (-(cos(g2)*sin(a2)) + cos(a2)*sin(b2)*sin(g2)))/f - ((cos(a1)*cos(b1) +(x1*(cos(a1)*cos(g1)*sin(b1) + sin(a1)*sin(g1)))/f +  (y1*(-(cos(g1)*sin(a1)) +cos(a1)*sin(b1)*sin(g1)))/f)* (cos(a2)*cos(g2) + sin(a2)*sin(b2)*sin(g2)))/f
    
    ])

    D=np.array([D3,-D2,D1])
    return(D)


def deriv_det(x, theta, C, j, k, l): 
 #x ensemble des coordonnées des projections: n=nombre de sous matrices, 1 matrice=1 projeté sur les 6 cameras, ligne i = coordonnées homogenes du projeté sur la caméra i
 #C ensemble des centres des caméras en coordonnées non hom --> matrice Kx3
 #theta ensemble des angles des caméras: 1 ligne = 1 caméra --> matrice Kx3
 #j indice du point projeté
 #k, l: numeros des caméras  on aura toujours k<l
 
 #retourne les derivées partielles en chacune des composantes (dans l'ordre (theta, X)) du determinant complet
    D=calcul_D(x, theta, j,k,l)
    
    centre = C[l]-C[k]   #matrice 1x3
    return(centre.dot(D))
    


def genere_liste_couples(K): # N= nombre de caméras
    liste_couples=[]
    for i in range(K):
        for h in range(i+1, K):
            liste_couples.append([i,h])
    return(liste_couples)
        

def matrice_Aj_Bj(x, theta, C, j):
    K, N = C.shape[0], x.shape[0]
    B_j=np.zeros((K*(K-1)/2,2*K))
    A_j = np.zeros((K*(K-1)/2, 3*K))
    liste_couples=genere_liste_couples(K)
    for (i, (k, l)) in enumerate(liste_couples):
        deriv=deriv_det(x, theta, C, j, k, l)
        A_j[i, 3*k] = deriv[0]
        A_j[i, 3*l+1] = deriv[1]
        A_j[i, 3*k+2] = deriv[2]
        A_j[i, 3*l] = deriv[3]
        A_j[i, 3*k+1] = deriv[4]
        A_j[i, 3*l+2] = deriv[5]
        B_j[i,2*k]=deriv[6] 
        B_j[i,2*l]=deriv[7]
        B_j[i,2*k+1]=deriv[8] 
        B_j[i,2*l+1]=deriv[9]
    
    
    return A_j, B_j
    
def matrice_A_B(x, theta, C):
    K, N = C.shape[0], x.shape[0]
    B = np.zeros((K*(K-1)*N/2, 2*K*N))
    A = np.zeros((K*(K-1)/2*N, 3*K))
    for j in range(N):
        B_j=matrice_Aj_Bj(x, theta, C, j)[1]
        A_j=matrice_Aj_Bj(x, theta, C, j)[0]
        B[K*(K-1)/2*j:K*(K-1)/2*(j+1), 2*K*j:2*K*(j+1)] = B_j
        A[K*(K-1)/2*j:K*(K-1)/2*(j+1)]=A_j
    return A, B
    

K, N, f = 4, 1, 10
C = (rand(K, 3) - 0.5) * 20   # tirage au sort des coordonnées entre -10 et 10
theta = rand(K, 3) * 2 * np.pi   # tirage au sort des coordonnées des angles entre 0 et 2pi
x = (rand(N, K, 2) - 0.5) * 10 # tirage au sort des coordonnées entre -5 et 5
 
A, B = matrice_A_B(x, theta, C)
print(A)