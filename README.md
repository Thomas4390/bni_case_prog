# Cas de programmation | BNI

# Description de la structure du projet : 

- **converted_data** (dossier) : Contient les données de départ converties au format .parquet afin de rendre l'ouverture et le téléchargement des données plus rapide. 
- **data** (dossier) : Contient les données de départ auxquelles la 2ème ligne a manuellement été retirée afin de rendre la lecture et la conversion en .parquet plus facile. Ces données n'ont pas été mise en ligne sur GitHub à cause de leur taille trop volumineuse. 
- **documentation** (dossier) : Contient la documentation utilisée pour le cas. 
- **figures** (dossier) : Contient les graphiques utilisés dans le projet. Note : La majeure partie des figures se trouvent sous la forme d'un fichier html dans le dossier **results_data**. 
- **filtered_data** (dossier) : Contient les données après filtration par le filtre de liquidité. Les extensions sont en .parquet. 
- **results_data** (dossier) : Contient les résultats les plus importants du projet. On y retrouve les métriques de performance (avec de nombreux graphiques) sous la forme d'une page hmtl pour les 3 stratégies testées (base_strategy, new_strategy et other_strategy). On y retrouve aussi la répartition des poids annuels concernant les secteurs GICS, les poids de rééquilibrage annuels ainsi que les "daily drifted weights" pour chacune des stratégies. 
- **backtesting.py** (fichier) : Script qui contient le calcul des "daily drifted weights" ainsi que la comparaison des performances des différentes stratégies par rapport au benchmark. Les outputs sont ensuite sauvergardés dans le dossier results_data. 
- **base_strategy.py** (fichier) : Ce script exécute la stratégie de la volatilité inverse de base, en prenant en compte des contraintes sur les poids de manière individuel et des contraintes sectorielles. Le script contient aussi des fonctions qui permettent de vérifier que les critères ont bien été respectés, comme par exemple le fait que les poids doivent sommer à 1, ou encore que les contraintes d'inégalités sont respectées. Si les contraintes ne sont pas respectées, un message d'erreur apparaît dans la console (l'allocation est quand même obtenue). Pour certains paramètres initiaux, il peut arriver que les contraintes ne soient légèrement pas respectées. La performance de cette stratégie est comparé au benchmark dans le dossier **results_data** avec le fichier html : `base_strategy_metrics.html`. 
- **new_strategy.py** (fichier) : Ce script exécute la stratégie de volatilité inverse avec skewness. Cette stratégie d'investissement est basée sur l'inverse de la volatilité et la skewness négative des actifs du portefeuille. Elle vise à attribuer des poids aux actifs en tenant compte de ces deux mesures. Voici un aperçu du fonctionnement de la stratégie:

1. Calculer les rendements quotidiens des actifs à partir des prix.
Pour chaque date de rééquilibrage :
a. Sélectionner les rendements de l'année écoulée jusqu'à la date de rééquilibrage.
b. Calculer la volatilité annuelle de chaque actif.
c. Calculer la skewness négative de chaque actif.
d. Calculer l'inverse de la volatilité et normaliser les valeurs.
e. Normaliser les valeurs de skewness négative.
f. Créer une combinaison pondérée de l'inverse de la volatilité et de la skewness négative en utilisant les poids vol_weight et skew_weight.
2. Appliquer les contraintes de poids individuelles (min_weight, max_weight) et sectorielles (sector_max_weight) aux poids combinés.
3. Redistribuer les poids pour s'assurer que leur somme est égale à 1.
4. Stocker les poids du portefeuille rééquilibré pour chaque date de rééquilibrage dans un DataFrame.

La stratégie alloue des poids plus importants aux actifs ayant une faible volatilité et une skewness négative importante. L'objectif est de diversifier le portefeuille en tenant compte de ces deux caractéristiques, afin de minimiser les risques et de potentiellement améliorer les rendements.
