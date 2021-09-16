# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:26:08 2019

@author: dufourp
"""


import matplotlib.pyplot as plt
import numpy as np
import random
import time


size = 50          #nombre de villes
nb = 30   #nombre individus
G =5000  #nombre de generations
taux_mutation = 0.65
p = 0.7        


""" descentes correspond au nombre de descentes locales qui seront effectuées dans les deux algorithmes de descentes locales. Plus ce nombre est grand, meilleure sera la 
    solution approchée des deux algorithmes. """

descentes = 2000

""" alpha est un paramètre de l'algorithme recuit, 1/alpha correspond au nombre d'itérations que va effectuer cet algorithme. """

alpha = 0.00001             # pour 20 villes =>0.0001 / pour 50 villes =>0.00001 dans le cas ou To = 1


#____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________Création des villes___________________________________________________________________________________

""" Les listes X et Y corespondent aux coordonnées des villes. La liste P correspond au couple de coordonnées. La distance entre chaque ville est stockées dans le tableau 
    villes, une matrice symétrique. """

X = list(np.random.randint(1,100,size))
Y = list(np.random.randint(1,100,size))
P = [X,Y]    # liste des positions en abscisse et ordonnée de chaque ville

n = list(range(0,size))   # correspond aux noms des villes (exemple: l'indice 0 correspond à la ville 0), l'indice maximal des villes est donc size-1
                          # on l'utilise cette liste pour préciser le numéro de chaque ville sur les graphes notamment avec la fonction ax.annotate()

def distance(a,b,P):    # renvoie la distance entre deux villes
    """ a et b correspondent à deux villes """
    return ((P[0][a] - P[0][b])**2 + (P[1][a] - P[1][b])**2)**(1/2)

def generevilles(size):                 # renvoie une matrice symétrique où figurent les distances entre les villes (de diagonale nulle)
    villes = np.zeros((size,size))      # np.zeros(): permet de créer une matrice où les coefficients valent tous 0
    
    for i in range(size):
        for k in range(0,i):    
                
            if i != k :           # la matrice renvoyée est symétrique 
                d = distance(i,k,P)
                villes[i,k] = d
                villes[k,i] = d
                
    return villes

villes = generevilles(size)     # la matrice qui sert de base d'étude pour les différentes méthodes algorithmiques qui suivent

#_____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________Calcul de la distance totale des chemins_______________________________________________________________

""" La fonction renvoie la distance totale d'un chemin, elle recalcule les distances entre les villes au lieu de les relever directement dans la matrice villes où elles ont
    déjà été calculé (légèrement inutile mais pas important car elle sert uniquement aux graphes). """

def longueurTotale(chemin):
    l = 0
    for i in range(0,size):
        l += distance(chemin[i],chemin[i+1],P)      # la fonction distance a été définie au début de la page
    return l


#_____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________Méthode du plus proche voisin__________________________________________________________________________

""" On part de la ville d'indice 0. Puis on sélectionne la prochaine ville telle que le poids entre la ville courante et la prochaine soit minimal. On répète cette opération
    jusqu'à avoir visité toutes les villes. """

EE = time.time()

def procheVoisin(villes):   
    indice = 0            # initialisé à la premiere ville prise qui est 0
    N = size
    villespossible = list(range(1,N))       # liste des villes qui restent à visiter
    villesprises=[0]                # liste du chemin emprunté pour relier les villes
                                    # la première ville est 0, ce qui correpond au premier élément de la liste

    for i in range(0,N-1):           # nombre d'étapes moins 1 car on part de la ville 0 et on y revient
        
        distance = [villes[villesprises[-1],i] for i in villespossible]          
                            # on se place sur la ligne de la ville courante dans la matrice villes qui correspond à l'élément: villesprises[-1]
                            # on créé alors la liste distance composée de la distance de toutes les villes à visiter par rapport à celle courante 

        indice = np.argmin(distance)         # np.argmin(): renvoie l'indice du plus petit élément de la liste distance
        
        villesprises.append(villespossible[indice])
        villespossible.pop(indice)           # supprime la ville que l'on vient de visiter
        
    villesprises.append(0)
    return villesprises



chemin1 = procheVoisin(villes)


"""


FF = time.time()

a1 = FF -EE
print()
print("duree plus proche voisin", a1)

"""


#_____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________Tracé du plus proche voisin____________________________________________________________________________

""" On affiche dans un premier graphe la position des villes puis dans le second, le chemin des villes reliées entre elles, obtenu par la méthode du plus proche voisin. """

def trace(X,Y,chemin):   # à partir du chemin rentré, la fonction renvoie deux listes qui correspondant aux coordonnées en abscisse et en ordonnée
    nX=[]
    nY=[]
    for i in range(size):
        nX.append(X[chemin[i]])
        nY.append(Y[chemin[i]])
    return nX,nY


"""


nX,nY = trace(X,Y,chemin1) 

plt.scatter(X,Y,s=100)      # on créé un premier graphe où figure l'emplacement des villes, le couple (X,Y) est la position de chaque ville sur le graphe
plt.title('POSITION DES VILLES -' + str(size) + " villes-")
plt.xlabel('axe x')
plt.ylabel('axe y')
plt.xlim(-20,120)
plt.ylim(-20,120)


fig, ax = plt.subplots()
ax.scatter(nX,nY,s=25)
for i in range(size):
        ax.annotate(n[i],(X[i],Y[i]))       # ax.annotate(): permet de donner à chaque point le numéro de la ville qui lui est associé sur le graphe
plt.title("PLUS PROCHE VOISIN")
plt.plot(nX,nY,"r",marker="*")        
plt.xlim(-20,120)
plt.ylim(-20,120)
       
plt.show()

print("La longueur totale du chemin pour la méthode du plus proche voisin est:",longueurTotale(chemin1),"km")


"""


#_____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________Méthode de l'insertion_________________________________________________________________________________

""" On part de la ville d'indice 0. La seconde ville correspond à la ville d'indice 1. On place la ville suivante en balayant le chemin obtenu pour que sa position augmente
    le moins possible la distance totale. On réitère ensuite l'opération jusqu'à placer toutes les villes. """

GG = time.time()

chemin2 = [0,1,0]      # les deux premières villes sont par défaut 0 et 1 
villespossibles = list(range(2,size))            # liste des villes qui restent à visiter 



def calculDistance(chemin):         # renvoie la distance du chemin sans prendre en compte la distance entre la première et la dernière ville du chemin
    l = 0
    for i in range(len(chemin)-1):                  # on prend les dimensions de la liste chemin2 et non size car la taille de chemin2 varie au cours de l'algorithme
        l += distance(chemin[i],chemin[i+1],P)      # la fonction distance a été définie au début de la page
    return l

for k in villespossibles: 
    best = [float('inf'), []]      # float('inf') correspond à coder l'infini, ainsi quelque soit le chemin, sa longueur totale est inférieure à l'infini
   
    for i in range(1,len(chemin2)):
        d = calculDistance(chemin2[:i]+[k]+chemin2[i:])     # on insère la ville à la i-eme position dans le chemin et calculDistance renvoie le poids total
        
        if d < best[0]:     # on regarde où la ville peut être placée pour avoir une distance minimale
            best[0] = d
            best[1] = chemin2[:i]+[k]+chemin2[i:]   # on insère la ville à la i-ème position dans le chemin

    chemin2 = best[1]



meilleurchemin = chemin2

"""

HH = time.time()

a2 = HH-GG
print()
print("duree insertion", a2)

"""


#_____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________Tracé de l'insertion___________________________________________________________________________________

""" On affiche avec un graphe le chemin des villes reliées entre elles, obtenu par la méthode de l'insertion. """

def trace2(X,Y,chemin):     # à partir du chemin rentré, la fonction renvoie deux listes qui correspondant aux coordonnées en abscisse et en ordonnée
    nX=[]
    nY=[]
    for i in range(size):
        nX.append(X[chemin[i]])
        nY.append(Y[chemin[i]])
    return nX,nY

"""


nX,nY = trace2(X,Y,chemin2) 

fig, ax = plt.subplots()
ax.scatter(nX,nY,s=25)
for i in range(size):
        ax.annotate(n[i],(X[i],Y[i]))       # ax.annotate(): permet de donner à chaque point le numéro de la ville qui lui est associé sur le graphe
plt.title("INSERTION")
plt.plot(nX,nY,"r",marker="*")        
plt.xlim(-20,120)
plt.ylim(-20,120)
        
plt.show()

print("La longueur totale du chemin pour la méthode de l'insertion est:",longueurTotale(chemin2),"km")

"""


#_____________________________________________________________________________________________________________________________________________________________________________
#_____________________________________________________Méthode de la descente locale pour le plus proche voisin________________________________________________________________

""" A partir du chemin renvoyé par la méthode du plus proche voisin, l'algorithme permutte localement et de manière aléatoire des villes entre elles. Tant que le nouveau 
    chemin obtenu est plus court que le précédent, il est modifié. Il est ensuite renvoyé. """

II = time.time()


Msol = chemin1      # calculé auparavant par la méthode du plus proche voisin

def calculdist(sol):     # renvoie la distance totale du chemin
    l = 0
    for i in range(0,size):
        l += villes[sol[i],sol[i+1]]    # on utilise les distances calculées au préalable dans le tableau villes
    return l + villes[sol[-1],0]

best = calculdist(Msol)         # distance totale du chemin renvoyé par la méthode du plus proche voisin

def permutation(chemin,i,r):    # renvoie le chemin avec les éléments d'indice i et r permutés
    nchemin = chemin[0:]        # on créé une nouvelle liste nchemin pour copier la liste chemin
    nchemin[r] = chemin[i]
    nchemin[i] = chemin[r]
    return nchemin

def descenteLocale(Msol,longueur):   # la fonction échange deux villes de manière aléatoire avec longueur = distance totale du chemin Msol
    nMsol = Msol[:]                 # on copie la liste de manière indépendante
    l = longueur 
    r = np.random.randint(1,size)      # np.random.randint(): renvoie un entier aléatoire, ici compris entre 1 et size 
    
    for i in range(1,size): 
        if i != r:                     # on ne veut pas intervertir une ville avec elle-même   
            nouvelleList = permutation(Msol,i,r)
            dist = calculdist(nouvelleList)         # on calcule la distance de nouvelleList obtenue
            
            if dist < l:            # on compare la distance totale de nouvelleList avec celle de nMsol2
                l = dist 
                nMsol = nouvelleList[:]                
    return nMsol,l

for i in range(0,descentes):        # descente est le nombre de descentes locales, donné en début de page
    Msol,best=descenteLocale(Msol,best)
    



chemin3 = Msol  

"""

JJ = time.time()
a3 = JJ-II
print()
print("duree  descente locale plus proche voisin", a3)

"""


#_____________________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________Tracé de la descente locale pour le plus proche voisin______________________________________________________________ 

""" On affiche avec un graphe le chemin des villes reliées entre elles, obtenu par la méthode de la descente locale avec le plus proche voisin. """ 
 
def trace3(X,Y,chemins):    # à partir du chemin rentré, la fonction renvoie deux listes qui correspondant aux coordonnées en abscisse et en ordonnée
    nX=[]
    nY=[]
    for i in range(size):
        nX.append(X[chemins[i]])
        nY.append(Y[chemins[i]])
    return nX,nY

"""

nX,nY = trace3(X,Y,chemin3) 

fig, ax = plt.subplots()
ax.scatter(nX,nY,s=25)
for i in range(size):
        ax.annotate(n[i],(X[i],Y[i]))            # ax.annotate(): permet de donner à chaque point le nom de la ville qui lui est associé sur le graphe
plt.title("DESCENTE LOCALE -plus proche voisin-")
plt.plot(nX,nY,"r",marker="*")        
plt.xlim(-20,120)
plt.ylim(-20,120)
       
plt.show()

print("La longueur totale du chemin pour la méthode de descente locale avec plus proche voisin est:",longueurTotale(chemin3),"km")

"""

#_____________________________________________________________________________________________________________________________________________________________________________
#___________________________________________________________Méthode de la descente locale pour l'insertion____________________________________________________________________

""" A partir du chemin renvoyé par la méthode de l'insertion, l'algorithme permutte localement et de manière aléatoire des villes entre elles. Tant que le nouveau chemin 
    obtenu est plus court que le précédent, il est modifié. Il est ensuite renvoyé. """

KK = time.time()


Msol2 = meilleurchemin      # calculé auparavant par la méthode de l'insertion

def calculdist2(chemin):    # renvoie la distance totale du chemin, on n'utilise pas la fonction longueurTotale car elle a un coût temporel plus élévé
    l = 0
    for i in range(0,size):
        l += villes[chemin[i],chemin[i+1]]   # on utilise les distances calculées au préalable dans la matrice villes
    return l + villes[chemin[-1],0]

best2 = calculdist2(Msol2)   # distance totale du chemin renvoyé par la méthode insertion

def permutation2(chemin,i,r):     # renvoie le chemin avec les éléments d'indice i et r permutés
    nchemin = chemin[0:]         # on créé une nouvelle liste nchemin pour copier la liste chemin
    nchemin[r]=chemin[i]
    nchemin[i]=chemin[r]
    return nchemin

def descenteLocale2(Msol2,longeur): # la fonction échange deux villes de manière aléatoire avec longueur = distance totale du chemin Msol2
    nMsol2 = Msol2[:]               # on copie la liste sa manière indépendante
    l = longeur 
    r = np.random.randint(1,size)        # np.random.randint(): renvoie un entier aléatoire, ici compris entre 1 et size
    
    for i in range(1,size):       
        if i != r:                       # on ne veut pas intervertir une ville avec elle-même
            nouvelleList = permutation2(Msol2,i,r)  
            dist = calculdist2(nouvelleList)          # on calcule la distance de nouvelleList obtenue
            
            if dist < l:            # on compare la distance totale de nouvelleList avec celle de nMsol2
                l = dist 
                nMsol2 = nouvelleList[:]                
    return nMsol2,l

for i in range(0,descentes):        # descente est le nombre de descentes locales, donné en début de page
    Msol2,best2 = descenteLocale2(Msol2,best2)
   
   


chemin4 = Msol2

"""

LL = time.time()

a4 = LL-KK
print()
print("duree descente locale insertion", a4)

"""


#_____________________________________________________________________________________________________________________________________________________________________________
#_____________________________________________________________Tracé de la descente locale pour l'insertion____________________________________________________________________ 
  
""" On affiche avec un graphe le chemin des villes reliées entre elles, obtenu par la méthode de la descente locale avec l'insertion. """  

def trace4(X,Y,chemin):     # à partir du chemin rentré, la fonction renvoie deux listes qui correspondant aux coordonnées en abscisse et en ordonnée
    nX=[]
    nY=[]
    for i in range(size):
        nX.append(X[chemin[i]])
        nY.append(Y[chemin[i]])
    return nX,nY

"""

nX,nY = trace4(X,Y,chemin4) 

fig, ax = plt.subplots()
ax.scatter(nX,nY,s=25)
for i in range(size):
        ax.annotate(n[i],(X[i],Y[i]))                # ax.annotate(): permet de donner à chaque point le numéro de la ville qui lui est associé sur le graphe
plt.title("DESCENTE LOCALE -insertion-")
plt.plot(nX,nY,"r",marker="*")        
plt.xlim(-20,120)
plt.ylim(-20,120)        

plt.show()

print("La longueur totale du chemin pour la méthode de descente locale avec insertion est:",longueurTotale(chemin4),"km")

"""


#_____________________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________Méthode du recuit simulé____________________________________________________________________________

""" Le recuit simulé est un algorithme itératif considérant les minimums locaux. On échange aléatoirement les positions des villes entre elles à partir d'une solution 
    initiale. Si le chemin obtenu est meilleur que l'ancien, on le garde et on recommence. Sinon, il est moins bon que celui actuel et peut être accepté ou non selon une
    probabilité. """


AA = time.time()

# Lo = list(range(0,size))
Lo = list(range(0,size)) + [0]
To = 0.01

""" On cherche à modifier légèrement le chemin considéré par la fonction recuit. """



def fluctuation(chemin):    # on prend aléatoirement deux villes dans chemin et on inverse l'ordre des villes comprises entre ces deux villes 
    M = chemin[:]
    n = len(chemin)
    i = np.random.randint(1,n)   # la fonction renvoie un entier aléatoire à partir de la distribution uniforme dans l'intervalle (bas inclusif, haut exclusif)
    j = np.random.randint(1,n)
    
    while i == j:       # on veut i différent de j, tant qu'ils sont égaux, on change leur valeur
        i = np.random.randint(1,n)   
        j = np.random.randint(1,n)
    
    Min = min(i,j)
    Max = max(i,j)
    M[Min:Max] = M[Min:Max][::-1]       # on inverse les éléments compris entre Min et Max dans la liste M
    return  M

def calcDist(chemin):        # cette fonction renvoie la distance totale du chemin à partir des coordonnées de la matrice villes
    N = len(chemin)
    d = 0       
    l = chemin[0]       
    for i in range(1,N):
        d += villes[l,chemin[i]]
        l = chemin[i]
    return d  



def recuit(L):     # L correspond à une liste, c'est-à-dire un chemin 
    T = To
    for k in range(1,int(1/alpha)):     
        No = calcDist(L)
        L1 = fluctuation(L)
        N1 = calcDist(L1)
            
        if No > N1:
            L = L1[:]
        elif np.exp(-(N1-No)/T) >= np.random.uniform():   # loi de Boltzmann, suit une loi uniforme
            L = L1[:]
            
        T -= alpha*T    # la température diminue au cours du temps
    return L




cheminRecuit = recuit(Lo)

"""

BB = time.time()

a5 = BB-AA
print()
print("duree recuit ",a5)

"""


#_____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________Tracé du recuit simulé_________________________________________________________________________________

""" On affiche avec un graphe le chemin des villes reliées entre elles, obtenu par la méthode du recuit simulé. """

def trace5(X,Y,chemin):     # à partir du chemin rentré, la fonction renvoie deux listes qui correspondant aux coordonnées en abscisse et en ordonnée
    nX=[]
    nY=[]
    for i in range(size):
        nX.append(X[chemin[i]])
        nY.append(Y[chemin[i]])
    return nX,nY

"""

nX,nY = trace5(X,Y,cheminRecuit) 

fig, ax = plt.subplots()
ax.scatter(nX,nY,s=25)
for i in range(size):
        ax.annotate(n[i],(X[i],Y[i]))       # ax.annotate(): permet de donner à chaque point le numéro de la ville qui lui est associé sur le graphe
plt.title("RECUIT SIMULE")
plt.plot(nX,nY,"r",marker="*")        
plt.xlim(-20,120)
plt.ylim(-20,120)         

plt.show()

print("La longueur totale du chemin pour la méthode du recuit simulé est:",longueurTotale(cheminRecuit),"km")  

"""



#____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________GENERATION DE LA POPULATION INITIALE___________________________________________________________________________________


#individu: liste du type [0,1,5,3,4,2,0] solution du pvc (liste des villes par lesquelles passent le voyageur)

#population: ensemble d'individus 


def individu_aleat(villes):               #generation aléatoire de solutions approchées (individus)
       L = [ i for i in range(1,size)]    
       random.shuffle(L)
       L = [0] + L +[0]
       return L
    

        
def population(villes,nb):                #on crée une liste de nb individus
    pop = []
    for i in range(nb):
        pop += [individu_aleat(villes)]
    return pop
    
#____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________FONCTION D'ADAPTATION ET EVALUATION___________________________________________________________________________________



def fitness(ind):                          #fonction d'adaptation: distance totale parcourue par le voyageur
    d = 0
    for i in range(1,len(ind)):            
         d += villes[ind[i-1],ind[i]]
    return d
    
    

def fittest(pop):                          # renvoie le meilleur individu dans la population (chemin le plus court)
    L = [ (fitness(pop[i]),pop[i]) for i in range(len(pop))]
    return min(L)[1]

          
    
#____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________SELECTION___________________________________________________________________________________
            


def selection_roulette(pop):                  
    L =  [fitness(ind) for ind in pop]                   
    somme_fitness =  sum(L)
    r = random.uniform(0,somme_fitness)
    s = L[0]
    i = 0
    while s < r:
        i += 1
        s += L[i]
    return pop[i]
     

    

def selection_tournoi(pop,p):
        m,n = random.randint(0,len(pop)-1),random.randint(0,len(pop)-1)     #les deux individus peuvent être les mêmes
        x = random.random()
        if fitness(pop[m]) < fitness(pop[n]):          
            if x < p:                                                       #probabilité p que le meilleur individu soit sélectionné
                return pop[m]
            else:
                return pop[n]
        else:
            if x < p:
                return pop[n]
            else:
                return pop[m]



    

#____________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________________________CROISEMENT/RECOMBINAISON_____________________________________________________________________________



def crossover_simple(pere,mere):                 #croisement simple où la cassure est situé à la moitié du parcours
    N = len(pere)//2
    fils = pere[:N]
    for k in range(N,len(mere)):
        a = mere[k]
        while a in fils and a !=0 :
              a += 1
              if a == size:       
                  a = 1  
        fils += [a]
    return fils
        
def crossover_double(pere,mere):               #croisement double où la portion du chemin insérée est comprise entre le premier et le deuxième tiers
    N = len(pere)//3
    fils = pere[:N]
    for k in range(N,2*N):
        a = mere[k]
        while a in fils and a != 0:
              a += 1
              if a == size:       
                  a = 1
        fils += [a]
    for k in range(2*N,len(pere)):
        b = pere[k]
        while b in fils and b != 0:
              b += 1
              if b == size:      
                  b = 1
        fils += [b]
    return fils    
    

    
#____________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________________________MUTATION_____________________________________________________________________________



def mutation_permutation(ind,taux_mutation):                  #l'individu a une probabilité égale à taux_mutation de permuter deux villes
    for pos1 in range(1,len(ind)-1):
          if random.random() < taux_mutation:         #taux_mutation = paramètre à définir
            pos2 = random.randrange(1,len(ind)-1)
            ind[pos1],ind[pos2] = ind[pos2],ind[pos1]    #permutation des deux villes
    return ind
    


def mutation_bridge(ind,taux_mutation):            #l'individu a une probabilité égale à taux_mutation de former des ponts séquentiels
    i = random.randrange(1,len(ind)-10)
    k = random.randrange(i+2,len(ind)-6)
    l = k + random.randrange(0,3)
    x = random.random()
    if x < taux_mutation:
        ind = ind[:i+1] + ind[k:l+1] + ind[i+1:k] + ind[l+1:]           #pont séquentiel
    return ind   
    


    
#____________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________________________OPTIMISATION_____________________________________________________________________________



def optimisation_inversion_parcours(ind):          #optimisation de l'individu: choix du meilleur individu entre celui initial et celui dont le parcours est inversé
    for i in range(1,len(ind)-3):
        for j in range(i+1,len(ind)-1):
            L = []
            S = []
            L = ind[i:j]
            L.reverse()
            S = ind[:i] + L + ind[j:]
            if fitness(S) < fitness(ind):
                ind = S            
    return ind


def optimisation_inversion_parcours_v2(ind):                   #même principe mais moins coûteux (on détermine seulement les variations de distance sur les arêtes modifiées)
    for i in range(1,len(ind)-3):
        j = random.randrange(i+1,len(ind)-2)   
        a = villes[ind[i-1],ind[j]] + villes[ind[i],ind[j+1]]
        b = villes[ind[i-1],ind[i]] + villes[ind[j],ind[j+1]]
        if a < b:
            L = ind[i:j+1]
            L.reverse()
            ind = ind[:i] + L + ind[j+1:]
    return ind
                          
def optimisation_inversion_parcours2(ind):      #même principe mais moins coûteux (on détermine seulement les variations de distance sur les arêtes modifiées)
    for i in range(1,(len(ind)-3)):
        for j in range(i+1,len(ind)-1):
            a = villes[ind[i-1],ind[j]] + villes[ind[i],ind[j+1]]
            b = villes[ind[i-1],ind[i]] + villes[ind[j],ind[j+1]]
            if a < b:
                L = ind[i:j+1]
                L.reverse()
                ind = ind[:i] + L + ind[j+1:]
    return ind



#____________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________________________REINSERTION_____________________________________________________________________________


def reinsertion(pop,fils):                          #insertion du fils dans la population et suppression du pire individu
    L = [ (fitness(pop[i]),pop[i]) for i in range(len(pop))] 
    L.append((fitness(fils),fils))
    L = [ x[1] for x in sorted(L)]
    del L[-1]  
    return L


#____________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________________________GRAPHES_____________________________________________________________________________




def coordonnees(X,Y,chemin):   # à partir du chemin rentré, la fonction renvoie deux listes qui correspondant aux coordonnées en abscisse et en ordonnée
    nX=[]
    nY=[]
    for i in range(size):
        nX.append(X[chemin[i]])      #abscisses des villes du chemin
        nY.append(Y[chemin[i]])      #ordonnées des villes du chemin
    return nX,nY



plt.scatter(X,Y,s=100)      # on créé un premier graphe où figure l'emplacement des villes, le couple (X,Y) est la position de chaque ville sur le graphe
plt.title('POSITION DES VILLES -' + str(size) + " villes-")
plt.xlabel('axe x')
plt.ylabel('axe y')
plt.xlim(-20,120)
plt.ylim(-20,120)


def graphe(ind):                                  #tracé du chemin généré par l'algorithme génétique
    nX,nY = coordonnees(X,Y,ind) 
    nville = list(range(0,size))                   # correspond aux noms des villes (exemple: l'indice 0 correspond à la ville 0), l'indice maximal des villes est donc size-1                       
    fig, ax = plt.subplots()
    ax.scatter(nX,nY,s=25)
    
    for i in range(size):
            ax.annotate(nville[i],(X[i],Y[i]))          # ax.annotate(): permet de donner à chaque point le numéro de la ville qui lui est associé sur le graphe
    plt.title("ALGORITHME GENETIQUE")
    plt.plot(nX,nY,"r",marker="*")        
    plt.xlim(-20,120)
    plt.ylim(-20,120)
    plt.show()
    
    print("La longueur du plus court chemin par l'algorithme génétique est:",fitness(ind),"km")
    print()
    
    

   
#____________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________________________EVOLUTION_____________________________________________________________________________




def evolution(pop):                                                            #Evolution de la population
        pere, mere = selection_tournoi(pop,p),selection_tournoi(pop,p)                   #SELECTION des parents                            
        fils = crossover_simple(pere,mere)                                        #CROISEMENT des parents
        fils_mute = mutation_bridge(fils,taux_mutation)                                  #MUTATION du fils
        fils_opti = optimisation_inversion_parcours2(fils_mute)                          #OPTIMISATION du fils
        pop = reinsertion(pop, fils_opti)                                                #INSERTION du fils
        return pop  


def generation(pop,G):                       #Amélioration de la population sur G générations et affichage de la solution finale
    for i in range(G):                                   #on teste sur G générations
        pop = evolution(pop)
    graphe(fittest(pop))
    return pop

    



#___________________________Tracé de la distance minimale en fonction du nombre de générations_______________________

def determination_coordonnees(pop,G):  #X liste des générations successives, Y liste des distances minimales successives  
    X = [0]
    Y = [fitness(fittest(pop))]
    for i in range(1,G):                                                
         pop = evolution(pop)
         X += [i]
         Y += [fitness(fittest(pop))]
    return np.array(X),np.array(Y), pop
    
def trace_generation(pop,G):          #tracé de la distance minimale en fonction du nombre de générations
    X,Y,pop = determination_coordonnees(pop,G)
    fig, ax = plt.subplots()
    plt.plot(X,Y,label = "Algorithme génétique", color = "blue")
    a = longueurTotale(chemin1)
    plt.plot(X, G*[a],label = "Plus proche voisin", color = "red")
    b = longueurTotale(chemin3)
    plt.plot(X,G*[b],label = "Descente locale plus proche voisin", color = "green")
    c = longueurTotale(chemin2)
    plt.plot(X,G*[c],label = "Insertion", color = "pink")
    d = longueurTotale(chemin4)
    plt.plot(X,G*[d],label = "Descente locale insertion", color = "grey")
    e = longueurTotale(cheminRecuit)
    plt.plot(X,G*[e],label = "Recuit simulé", color = "yellow")
    plt.title("Evolution de la distance minimale en fonction du nombre de générations ")
    plt.xlabel('Nombre de générations')
    plt.ylabel('Longueur du circuit')
    plt.ylim(500,700)
    plt.xlim(4000)
    plt.legend(fontsize = 8, loc = 'best')
    plt.show()
    return pop



"""

CC = time.time()

   
pop = population(villes,nb)
pop = generation(pop,G) 

DD = time.time()

a6 = DD-CC
print()
print("duree genetique: ",a6)

"""


pop = population(villes,nb)
pop = trace_generation(pop,G)



