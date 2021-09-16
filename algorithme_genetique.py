# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import random
import time

size = 50              #nombre de villes : villes allant de 0 à size-1 (pour 10 villes, 0 à 9 sont les noms des villes)
nb = 30                #nombre individus
G = 5000               #nombre de generations
taux_mutation = 0.65
p = 0.7                             #selection_tournoi: probabilité de choisir le meilleur individu parmi 2


#____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________CREATION DES VILLES ET DE LA MATRICE D'ADJACENCE___________________________________________________________________________________

""" Les listes X et Y correspondent aux coordonnées des villes. La liste P correspond au couple de coordonnées. La distance entre chaque ville est stockées dans le tableau 
    villes, une matrice symétrique. """

X = list(np.random.randint(1,100,size))
Y = list(np.random.randint(1,100,size))
P = [X,Y]                 # liste des positions en abscisse et ordonnée de chaque ville

                          # correspond aux noms des villes (exemple: l'indice 0 correspond à la ville 0), l'indice maximal des villes est donc size-1
                          # on l'utilise cette liste pour préciser le numéro de chaque ville sur les graphes notamment avec la fonction ax.annotate()


def distance(a,b,P):                    # renvoie la distance entre deux villes
    """ a et b correspondent à deux villes """
    return ((P[0][a] - P[0][b])**2 + (P[1][a] - P[1][b])**2)**(1/2)


def generevilles(size):                 # renvoie une matrice symétrique où figurent les distances entre les villes (de diagonale nulle)
    villes = np.zeros((size,size))      # np.zeros(): permet de créer une matrice où les coefficients valent tous 0
    
    for i in range(size):
        for k in range(0,i):    
    
            if i != k:                  # la matrice renvoyée est symétrique  et la distance d'une ville à elle-même est nulle
                d = distance(i,k,P)
                villes[i,k] = d
                villes[k,i] = d
                
    return villes



villes = generevilles(size)     # la matrice qui sert de base d'étude pour les différentes méthodes algorithmiques qui suivent




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

 
def selection_hasard(pop):                              #generation de p individus de manière aléatoire(ces individus peuvent être les mêmes)
    L= [ individu_aleat(pop) for i in range(0,len(pop)*2)]        
    return fittest(L)                      


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
     

def selection_roulette2(pop):                                              #codage plus concis et moins coûteux de la sélection par roulette
    L = [ (fitness(pop[i])*random.random(),pop[i]) for i in range(len(pop))] 
    L = [ x[1] for x in sorted(L)] 
    return L[0]
    

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



def crossover_simple_moitie(pere,mere):                 #croisement simple où la cassure est situé à la moitié du parcours
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
        
def crossover_double_tiers(pere,mere):               #croisement double où la portion du chemin insérée est comprise entre le premier et le deuxième tiers
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
    
  

def crossover_simple_duo(pere,mere):                  #on génère deux individus fils en inversant les rôles du père et de la mère
    N = random.randint(2,len(pere)-3)   #bon nombre d'iterations: liste de taille 2 [0,5] au minimum et égale à la liste pere sans le 0 moins 1 element : [0,5,..,4] + [7]+[0]
    fils1 = pere[:N]
    fils2 = mere[:N]
    
    for k in range(N,len(pere)):
        a = mere[k]
        while a in fils1 and a != 0:
              a += 1
              if a == size:
                  a = 1
        fils1 += [a]         

    for k in range(N,len(pere)):
        b = pere[k]
        while b in fils2 and b != 0:
              b += 1
              if b == size:
                  b = 1
        fils2 += [b]
    
    return (fils1,fils2)


     
def crossover_double_duo(pere,mere):             #on génère deux individus fils en inversant les rôles du père et de la mère
    N = random.randint(2,len(pere)-4)
    M = random.randint(N+1,len(pere)-3)
    
    copie_mere = mere.copy()
    copie_pere = pere.copy()
    fils1 = pere[:N]
    fils2 = mere[:N]
    
    for k in range(N,M):
        a = copie_mere[k]
        while a in fils1 and a != 0:
              a += 1
              if a == size:           
                 a = 1
        fils1 += [a]
    for k in range(M,len(pere)):
        b= copie_pere[k]
        while b in fils1 and b != 0:
              b += 1
              if b == size:          
                  b = 1
        fils1 += [b]
   
    for k in range(N,M):
        c= pere[k]
        while c in fils2 and c != 0:
              c += 1
              if c== size:         
                  c = 1
        fils2 += [c]
    for k in range(M,len(mere)):
        d = mere[k]
        while d in fils2 and d != 0:
              d += 1
              if d == size:         
                  d = 1
        fils2 += [d]
        
    return (fils1,fils2)
    
#____________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________________________MUTATION_____________________________________________________________________________


def mutation_permutation(ind,taux_mutation):                  #l'individu a une probabilité égale à taux_mutation de permuter deux villes
    for pos1 in range(1,len(ind)-1):
          if random.random() < taux_mutation:      #taux_mutation = paramètre à définir
            pos2 = random.randrange(1,len(ind)-1)
            ind[pos1],ind[pos2] = ind[pos2],ind[pos1]    #permutation des deux villes
    return ind
    
    
def mutation_inversion_parcours(ind,taux_mutation):       #l'individu a une probabilité égale à taux_mutation d'inverser le parcours dans une portion de chemin choisie aléatoirement
    if random.random() < taux_mutation:      #taux_mutation = paramètre à définir
        pos2 = random.randrange(1,len(ind)-1)
        pos1 = random.randint(1,len(ind)-1)
        m = min(pos1,pos2)
        M = max(pos1, pos2)
        L = ind[m:M]
        L.reverse()                    #inversion du parcours de m inclus à M exclus
        ind = ind[:m] + L + ind[M:]
    return ind


def mutation_bridge(ind,taux_mutation):            #l'individu a une probabilité égale à taux_mutation de former des ponts séquentiels
    i = random.randrange(1,len(ind)-10)
    k = random.randrange(i+2,len(ind)-6)
    l = k + random.randrange(0,3)
    x = random.random()
    if x < taux_mutation:
        ind = ind[:i+1] + ind[k:l+1] + ind[i+1:k] + ind[l+1:]
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


def optimisation_inversion_parcours3(ind):         
    for i in range(1,len(ind)-3):
        j = random.randrange(i+1,len(ind)-2)   
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


def reinsertion_duo(pop,fils1,fils2):               #insertion des deux fils dans la population et suppression des deux pires individus
    L = [ (fitness(pop[i]),pop[i]) for i in range(len(pop))] 
    L += [(fitness(fils1),fils1)] + [(fitness(fils2),fils2)]
    L = [ x[1] for x in sorted(L)]
    del L[len(L)-2:]  
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
    print("La solution générée par l'algorithme génétique est: ")
    print(ind)
    

   
#____________________________________________________________________________________________________________________________________________________________________________
#____________________________________________________________________________EVOLUTION_____________________________________________________________________________




def evolution(pop):                                                            #Evolution de la population
        pere, mere = selection_tournoi(pop,p),selection_tournoi(pop,p)                   #SELECTION des parents                            
        fils = crossover_simple_moitie(pere,mere)                                        #CROISEMENT des parents
        fils_mute = mutation_bridge(fils,taux_mutation)                                  #MUTATION du fils
        fils_opti = optimisation_inversion_parcours3(fils_mute)                          #OPTIMISATION du fils
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        pop = reinsertion(pop, fils_opti)                                                #INSERTION du fils
        return pop  

def generation(pop,G):                       #Amélioration de la population sur G générations et affichage de la solution finale
    for i in range(G):                                   #on teste sur G générations
        pop = evolution(pop)
    graphe(fittest(pop))
    return pop

   
def generation_affichage(pop,G):             #Amélioration de la population sur G générations et affichage de la meilleure solution pour chaque génération              
    for i in range(G):                                      #on teste sur G générations
        print("GENERATION",i+1)               
        graphe(fittest(pop))                                                 
        pop = evolution(G) 
    return pop
    

#_____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________COURBES____________________________________________________________________________     


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
    plt.plot(X,Y)
    plt.title("Evolution de la distance minimale en fonction du nombre de générations ")
    plt.xlabel('Nombre de générations')
    plt.ylabel('Longueur du circuit')
    plt.ylim(0)
    plt.show()
    return pop




#___________________________Comparaison opérateur sélection_______________________




def evolution2(pop):                                                    #Evolution avec selection_roulette
        pere, mere = selection_roulette(pop),selection_roulette(pop)                                    
        fils = crossover_simple_moitie(pere,mere)                             
        fils_mute = mutation_bridge(fils,taux_mutation)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        pop = reinsertion(pop, fils_opti)                             
        return pop


def determination_coordonnees2(pop,G):  
    X = [0]
    Y = [fitness(fittest(pop))]
    for i in range(1,G):                                                
         pop = evolution2(pop)
         X += [i]
         Y += [fitness(fittest(pop))]
    return np.array(X),np.array(Y), pop

def trace_selection(pop,G):                               #Comparaison des distances minimales renvoyées sur G générations avec sélection_roulette et sélection_tournoi
    X,Y, pop1 = determination_coordonnees(pop,G)
    X2,Y2, pop2 = determination_coordonnees2(pop,G)
    fig, ax = plt.subplots()
    plt.plot(X,Y,label= "selection_tournoi")
    plt.plot(X2,Y2,label= "selection_roulette") 
    plt.title("Evolution de la distance minimale en fonction du nombre de générations ")
    plt.xlabel('Nombre de générations')
    plt.ylabel('Longueur du circuit') 
    plt.legend(loc = 'best')
    plt.ylim(0)
    plt.show()
    print("Longueur finale selection_tournoi: ",fitness(fittest(pop1)))
    print()
    print("Longueur finale selection_roulette: ", fitness(fittest(pop2)))




#___________________________Comparaison opérateur croisement_______________________




def evolution3(pop):                                                   #Evolution avec croisement double
        pere, mere = selection_tournoi(pop,p),selection_tournoi(pop,p)                                 
        fils = crossover_double_tiers(pere,mere)                             
        fils_mute = mutation_bridge(fils,taux_mutation)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        pop = reinsertion(pop, fils_opti)                             
        return pop


def determination_coordonnees3(pop,G):  #X liste des generations successives, Y liste des distances minimales successives  
    X = [0]
    Y = [fitness(fittest(pop))]
    for i in range(1,G):                                                
         pop = evolution3(pop)
         X += [i]
         Y += [fitness(fittest(pop))]
    return np.array(X),np.array(Y), pop

def trace_croisement(pop,G):                                  #Comparaison des distances minimales renvoyées sur G générations avec croisement simple et croisement double
    X,Y, pop1 = determination_coordonnees(pop,G)
    X2,Y2, pop2 = determination_coordonnees3(pop,G)
    fig, ax = plt.subplots()
    plt.plot(X,Y,label= "crossover_simple")
    plt.plot(X2,Y2,label= "crossover_double") 
    plt.title("Evolution de la distance minimale en fonction du nombre de générations ")
    plt.xlabel('Nombre de générations')
    plt.ylabel('Longueur du circuit') 
    plt.legend(loc = 'best')
    plt.ylim(0)
    plt.show()
    print("Longueur finale crossover_simple: ",fitness(fittest(pop1)))
    print()
    print("Longueur finale crossover_double: ", fitness(fittest(pop2)))



#___________________________Comparaison mutation_______________________



 
def evolution4(pop):                                       #Evolution avec mutation_permutation
        pere, mere = selection_tournoi(pop,p),selection_tournoi(pop,p)                                    
        fils = crossover_simple_moitie(pere,mere)                            
        fils_mute = mutation_permutation(fils,taux_mutation)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        pop = reinsertion(pop, fils_opti)                             
        return pop


def determination_coordonnees4(pop,G):  #X liste des generations successives, Y liste des distances minimales successives  
    X = [0]
    Y = [fitness(fittest(pop))]
    for i in range(1,G):                                                
         pop = evolution4(pop)
         X += [i]
         Y += [fitness(fittest(pop))]
    return np.array(X),np.array(Y), pop
    


def evolution5(pop):                                     #Evolution avec mutation_inversion_parcours
        pere, mere = selection_tournoi(pop,p),selection_tournoi(pop,p)                                   
        fils = crossover_simple_moitie(pere,mere)                            
        fils_mute = mutation_inversion_parcours(fils,taux_mutation)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        fils_opti = optimisation_inversion_parcours3(fils_mute)
        pop = reinsertion(pop, fils_opti)                             
        return pop


def determination_coordonnees5(pop,G):  #X liste des generations successives, Y liste des distances minimales successives  
    X = [0]
    Y = [fitness(fittest(pop))]
    for i in range(1,G):                                                
         pop = evolution4(pop)
         X += [i]
         Y += [fitness(fittest(pop))]
    return np.array(X),np.array(Y), pop
    


def trace_mutation(pop,G):                                     #Comparaison des distances minimales renvoyées sur G générations avec mutation_permutation, mutation_bridge et mutation_inversion_parcours
    X,Y, pop1 = determination_coordonnees(pop,G)
    X2,Y2, pop2 = determination_coordonnees4(pop,G)
    X3,Y3, pop3 = determination_coordonnees5(pop,G)
    fig, ax = plt.subplots()
    plt.plot(X,Y,label= "mutation bridge")
    plt.plot(X2,Y2,label= "mutation_permutation") 
    plt.plot(X3,Y3,label= "mutation_inversion_parcours") 
    plt.title("Evolution de la distance minimale en fonction du nombre de générations ")
    plt.xlabel('Nombre de générations')
    plt.ylabel('Longueur du circuit') 
    plt.legend(loc = 'best')
    plt.ylim(0)
    plt.show()
    print("Longueur finale mutation bridge: ",fitness(fittest(pop1)))
    print()
    print("Longueur finale mutation_permutation: ", fitness(fittest(pop2)))
    print()
    print("Longueur finale mutation_inversion_parcours: ", fitness(fittest(pop3)))




#___________________________Comparaison taux_mutation_______________________




    
def trace_taux_mutation(pop,G):                            #Comparaison des distances minimales renvoyées sur G générations avec différentes valeurs de taux_mutation
    taux_mutation1 = 0.5
    X,Y, pop1 = determination_coordonnees(pop,G)
    taux_mutation2 = 0.6
    X2,Y2, pop2 = determination_coordonnees(pop,G)
    taux_mutation3 = 0.7
    X3,Y3, pop3 = determination_coordonnees(pop,G)
    fig, ax = plt.subplots()
    plt.plot(X,Y,label= taux_mutation1)
    plt.plot(X2,Y2,label=taux_mutation2)
    plt.plot(X3,Y3,taux_mutation3)  
    plt.title("Evolution de la distance minimale en fonction du nombre de générations ")
    plt.xlabel('Nombre de générations')
    plt.ylabel('Longueur du circuit') 
    plt.legend(loc = 'best')
    plt.ylim(0)
    plt.show()
    print("Longueur finale taux_mutation =",taux_mutation1,": ", fitness(fittest(pop1)))
    print()
    print("Longueur finale taux_mutation =",taux_mutation2,": ", fitness(fittest(pop2)))
    print()
    print("Longueur finale taux_mutation =",taux_mutation3,": ", fitness(fittest(pop3)))




#___________________________Comparaison probabilité_______________________


def trace_probabilite(pop,G):                         #Comparaison des distances minimales renvoyées sur G générations avec différentes valeurs de probabilité_sélection_tournoi
    p1 = 0.6
    X,Y, pop1 = determination_coordonnees(pop,G)
    p2 = 0.7
    X2,Y2, pop2 = determination_coordonnees(pop,G)
    p3 = 0.8
    X3,Y3, pop3 = determination_coordonnees(pop,G)
    fig, ax = plt.subplots()
    plt.plot(X,Y,label= p1)
    plt.plot(X2,Y2,label= p2)
    plt.plot(X3,Y3,label= p3)  
    plt.title("Evolution de la distance minimale en fonction du nombre de générations ")
    plt.xlabel('Nombre de générations')
    plt.ylabel('Longueur du circuit') 
    plt.legend(loc = 'best')
    plt.ylim(0)
    plt.show()
    print("Longueur finale p =",p1,": ", fitness(fittest(pop1)))
    print()
    print("Longueur finale p =",p2,": ", fitness(fittest(pop2)))
    print()
    print("Longueur finale p =",p3,": ", fitness(fittest(pop3)))



#_____________________________________________________________________________________________________________________________________________________________________________
#______________________________________________________________________APPLICATION____________________________________________________________________________     





pop = population(villes,nb)




L = pop.copy()

AA =time.time()
L = trace_generation(pop,G)
BB = time.time()

graphe(fittest(L))
print("Durée d'exécution",BB-AA)





trace_selection(pop,G)
trace_croisement(pop,G)
trace_mutation(pop,G)
trace_taux_mutation(pop,G)
trace_probabilite(pop,G)




