# Genetics-algorithms-for-TSP-
We solve the Traveling Salesman Problem using Genetics algorithms.


Le problème du voyageur de commerce a été posé pour la première fois sous la forme d’un jeu par William Rowan Hamilton en 1859. Il le formalise de la manière suivante : « Un voyageur de commerce doit faire sa tournée en visitant une seule fois un nombre fini de villes avant de revenir à son point d’origine. »                          
Ce problème d’optimisation appartient à la classe des problèmes NP-Complets. Cela signifie qu’on ne connaît pas d’algorithme trouvant la solution en temps polynomial, même si l’on peut vérifier toute solution rapidement. Ainsi les algorithmes permettant de trouver la solution exacte au problème ont un coût temporel très élevé et on leur préfère les algorithmes d’optimisation, plus rapides, qui trouvent une solution approchée.
Le problème du voyageur de commerce peut être modélisé à l’aide d’un graphe constitué d’un ensemble de sommets et d’un ensemble d’arêtes. Chaque sommet représente une ville, une arête symbolise le passage d’une ville à une autre, et on lui associe un poids qui représente la distance entre celles-ci.
 Résoudre le problème du voyageur de commerce revient à trouver dans ce graphe un cycle passant par tous les sommets une unique fois (un tel cycle est dit hamiltonien) et qui soit de longueur minimale.


Les algorithmes génétiques, initiés par John Holland en 1970, permettent de trouver efficacement une solution en un temps raisonnable. Ils reposent sur les principes du néodarwinisme : la sélection naturelle et la recombinaison génétique. D’une part, la sélection naturelle, selon Charles Darwin, énonce que les individus les mieux adaptés sont plus aptes à survivre et à devenir parents de la prochaine génération. D’autre part, Gregor Mendel, père de la génétique, explique que des mutations au niveau des gènes permettent l’évolution des espèces. Ainsi, les mutations les plus adaptées à l’environnement sont les plus transmises et grâce à cet héritage génétique, les espèces actuelles deviennent des “versions optimisées” de leurs ancêtres. 
Ils imitent ainsi au sein d’un programme les mécanismes d’évolution dans la nature : croisement, mutation, sélection. En effet, dans un algorithme génétique, on part d’une population de solutions potentielles au problème, initialement choisies aléatoirement. Les solutions sont ensuite évaluées grâce à la « fonction d’adaptation » ou fitness qui leurs assigne une valeur, afin de déterminer leur capacité de survie et de désigner celles qui transmettront leurs gènes à la génération suivante. Les solutions changent donc constamment, les plus adaptées survivent et se reproduisent, les autres disparaissent. On recommence ce cycle jusqu’à obtenir une solution satisfaisante au problème.

Le langage de programmation employé est Python. Ici la fonction d'adaptation représente la distance totale parcourue par le voyageur. Les solutions, ou individus, sont représentés en tant que liste des villes par lesquelles passent le voyageur. La population est alors une liste de liste (ensemble d’individus). Après sélection, les individus sont croisés de manière à former un individu fils, qui peut subir des mutations et est ensuite réintroduit au sein de la population.  Au cours des générations créées, de nouveaux individus apparaissent et remplacent les moins bons individus. Ainsi la qualité des individus s’améliore (la fonction d’adaptation est minimisée) et des chemins de plus en plus courts sont déterminés. La force des algorithmes génétiques est qu’ils font appel au hasard :  tous les individus peuvent être modifiés au cours du temps et être sélectionnés. 



![Figure 2021-09-16 151511](https://user-images.githubusercontent.com/90830443/133620341-04333adb-89dd-41cf-95da-7f7faade7496.png)
![Figure 2021-09-16 151500](https://user-images.githubusercontent.com/90830443/133619609-52b56e0d-75dc-4e1a-9c94-305c2f56d6f0.png)
![Figure 2021-09-16 151505](https://user-images.githubusercontent.com/90830443/133619621-f64d31c6-c2a7-4f30-8ec6-0603adfa189a.png)
![Figure 2021-09-16 151523](https://user-images.githubusercontent.com/90830443/133619653-a59dca6f-e7b3-4e56-b459-a8cbe89daf56.png)
![Untitled](https://user-images.githubusercontent.com/90830443/133620770-a15a6265-92ba-42bc-997a-ad392010e300.png)

