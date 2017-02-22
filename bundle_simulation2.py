import numpy as np
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy import cos, sin


def calcul_D(x, theta, j, k ,l):
    
    alpha_1=theta[k,0]
    alpha_2=theta[l,1]
    beta_1=theta[k,1]
    beta_2=theta[l,1]
    gamma_1=theta[k,2]
    gamma_2=theta[l,2]
    
    x_1=x[j,k,0]
    x_2=x[j,l,0]
    y_1=x[j,k,1]
    y_2=x[j,l,1]
    

    D1=np.array([
    
    -(cos(beta_2) *(cos(gamma_2) *x_2 + sin(gamma_2) *y_2) -  f*sin(beta_2))*(-sin(alpha_1)*(- sin(gamma_1)* x_1 + cos(gamma_1)* y_1) + cos(alpha_1) *(sin(beta_1) *(cos(gamma_1)* x_1 + sin(gamma_1) *y_1) +  f*cos(beta_1))),
    
    (cos(beta_1) *(cos(gamma_1)* x_1 + sin(gamma_1)* y_1) -  f*sin(beta_1))* (-sin(alpha_2) *(- sin(gamma_2)* x_2 + cos(gamma_2)* y_2) + cos(alpha_2) *(sin(beta_2)* (cos(gamma_2)* x_2 + sin(gamma_2)* y_2) +  f*cos(beta_2))),
    
    (-sin(beta_1) *(cos(gamma_1)* x_1 + sin(gamma_1) *y_1) -  f*cos(beta_1)) *(cos(alpha_2)* (- sin(gamma_2) *x_2 + cos(gamma_2)* y_2) + sin(alpha_2)* (sin(beta_2) *(cos(gamma_2) *x_2 + sin(gamma_2) *y_2) +  f*cos(beta_2)))-(cos(beta_2)*(cos(gamma_2) *x_2 + sin(gamma_2)* y_2) -  f*sin(beta_2))* sin(alpha_1) *(cos(beta_1) *(cos(gamma_1)* x_1 + sin(gamma_1)* y_1) -  f*sin(beta_1)),
    
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*sin(alpha_2)*(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))-(-sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*cos(beta_2))*(cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),  #eventuellement une erreur de parenthese...
    
    cos(beta_1)*(-sin(gamma_1)*x_1 + cos(gamma_1)*y_1)*(cos(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))-(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))*(cos(alpha_1)*(- cos(gamma_1)*x_1 -sin(gamma_1)*y_1) + sin(alpha_1)*sin(beta_1)*(-sin(gamma_1)*x_1 + cos(gamma_1)*y_1)),
        
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*(cos(alpha_2)*(- cos(gamma_2)*x_2 -sin(gamma_2)*y_2) + sin(alpha_2)*sin(beta_2)*(-sin(gamma_2)*x_2 + cos(gamma_2)*y_2))-cos(beta_2)*(-sin(gamma_2)*x_2 + cos(gamma_2)*y_2)*(cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
    
    cos(beta_1)*cos(gamma_1)*(cos(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))-(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))*(- cos(alpha_1)*sin(gamma_1) + sin(alpha_1)*sin(beta_1)*cos(gamma_1)),
    
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*(- cos(alpha_2)*sin(gamma_2) + sin(alpha_2)*sin(beta_2)*cos(gamma_2))-cos(beta_2)*cos(gamma_2)*(cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
    
    cos(beta_1)*sin(gamma_1)*(cos(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))-(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))*(cos(alpha_1)*cos(gamma_1) + sin(alpha_1)*sin(beta_1)*sin(gamma_1)),
    
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*(cos(alpha_2)*cos(gamma_2) + sin(alpha_2)*sin(beta_2)*sin(gamma_2))-cos(beta_2)*sin(gamma_2)*(cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))
    
    ])
    



    D3=np.array([
    
    (-sin(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))- (cos(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))*(cos(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) - sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1))),
    
   (cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))*(cos(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) -sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))- (-sin(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2)+f*cos(beta_2)))*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))) #j'ai rajouté un  +f*cos(beta_2)et parenthèse qui semblent manquer dans le latex
   ,   
    
    sin(alpha_1)*(cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))- (cos(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))*cos(alpha_1)*(cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1)),
        
    (cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))*cos(alpha_2)*(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))- (sin(alpha_2)*(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2)))*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
    
    (cos(alpha_1)*(- cos(gamma_1)*x_1 - sin(gamma_1)*y_1) + sin(alpha_1)*sin(beta_1)*(-sin(gamma_1)*x_1 + cos(gamma_1)*y_1))*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))- (cos(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))*(sin(alpha_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) + cos(alpha_1)*sin(beta_1)*(-sin(gamma_1)*x_1 + cos(gamma_1)*y_1)),
    
    (cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))*(sin(alpha_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) + cos(alpha_2)*sin(beta_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2))- (cos(alpha_2)*(- cos(gamma_2)*x_2 - sin(gamma_2)*y_2) + sin(alpha_2)*sin(beta_2)*(-sin(gamma_2)*x_2 + cos(gamma_2)*y_2))*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
        
    (-cos(alpha_1)*sin(gamma_1) + sin(alpha_1)*sin(beta_1)*cos(gamma_1))*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))- (cos(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))*(sin(alpha_1)*sin(gamma_1) + cos(alpha_1)*sin(beta_1)*cos(gamma_1)),
        
    (cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))*(sin(alpha_2)*sin(gamma_2) + cos(alpha_2)*sin(beta_2)*cos(gamma_2))- (-cos(alpha_2)*sin(gamma_2) + sin(alpha_2)*sin(beta_2)*cos(gamma_2))*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
    
    (cos(alpha_1)*cos(gamma_1) + sin(alpha_1)*sin(gamma_1))*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))- (cos(alpha_2)*(- sin(gamma_2)*x_2 + cos(gamma_2)*y_2) + sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))*(-sin(alpha_1)*cos(gamma_1) + cos(alpha_1)*sin(beta_1)*sin(gamma_1)),
        
    (cos(alpha_1)*(- sin(gamma_1)*x_1 + cos(gamma_1)*y_1) + sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))*(-sin(alpha_2)*cos(gamma_2) + cos(alpha_2)*sin(beta_2)*sin(gamma_2))- (cos(alpha_2)*cos(gamma_2)*y_2 + sin(alpha_2)*sin(beta_2)*sin(gamma_2))*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))
    
    ])
    
    
    
    
    
    D2=np.array([
    
    -(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))*(cos(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) - sin(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
    
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*(cos(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) - sin(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2))),
    
    (-sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*cos(beta_1))*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))-(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))*cos(alpha_1)*(cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1)),
    
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*cos(alpha_2)*(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))-(-sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*cos(beta_2))*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
    
    cos(beta_1)*(-sin(gamma_1)*x_1 + cos(gamma_1)*y_1)*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))-(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))*(sin(alpha_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(-sin(gamma_1)*x_1 + cos(gamma_1)*y_1))),
    
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*(sin(alpha_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(-sin(gamma_2)*x_2 + cos(gamma_2)*y_2)))-cos(beta_2)*(-sin(gamma_2)*x_2 + cos(gamma_2)*y_2)*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
    
    cos(beta_1)*cos(gamma_1)*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))-(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))*(sin(alpha_1)*sin(gamma_1) + cos(alpha_1)*sin(beta_1)*cos(gamma_1)),
        
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*(sin(alpha_2)*sin(gamma_2) + cos(alpha_2)*sin(beta_2)*cos(gamma_2) )-cos(beta_2)*cos(gamma_2)*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1))),
    
    cos(beta_1)*sin(gamma_1)*(sin(alpha_2)*(sin(gamma_2)*x_2 - cos(gamma_2)*y_2) + cos(alpha_2)*(sin(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) +  f*cos(beta_2)))-(cos(beta_2)*(cos(gamma_2)*x_2 + sin(gamma_2)*y_2) -  f*sin(beta_2))*(-sin(alpha_1)*cos(gamma_1) + cos(alpha_1)*sin(beta_1)*sin(gamma_1)),
    
    (cos(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) -  f*sin(beta_1))*(-sin(alpha_2)*cos(gamma_2) + cos(alpha_2)*sin(beta_2)*sin(gamma_2))-cos(beta_2)*sin(gamma_2)*(sin(alpha_1)*(sin(gamma_1)*x_1 - cos(gamma_1)*y_1) + cos(alpha_1)*(sin(beta_1)*(cos(gamma_1)*x_1 + sin(gamma_1)*y_1) +  f*cos(beta_1)))
    
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
    return (1/f**2)*A, (1/f**2)*B
    

K, N, f = 4, 1, 10
C = (rand(K, 3) - 0.5) * 20   # tirage au sort des coordonnées entre -10 et 10
theta = rand(K, 3) * 2 * np.pi   # tirage au sort des coordonnées des angles entre 0 et 2pi
x = (rand(N, K, 2) - 0.5) * 10 # tirage au sort des coordonnées entre -5 et 5
 
A, B = matrice_A_B(x, theta, C)
print(A)