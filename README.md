# Cas de programmation | BNI

*disclaimer* : Au cours du projet et de l'implémentation des différentes stratégies, j'ai fait de mon mieux afin de minimiser les erreurs de backtestings courantes dans le temps qui m'étais imparti. La principale erreur que l'on peut commetre en backtesting est d'utiliser des données futures pour prédire le passé. Avec du temps supplémentaire, j'aurais pu tester plus rigoureusement chacune des fonctions avant de passer à la suite du projet et ainsi être certain d'éviter ce genre d'erreur. J'aurais aussi aimé pouvoir tester des modèles plus complexes afin d'améliorer la stratégie initiale. Je reste tout de même satisfait du travail effectué dans ce cours laps de temps.

# Description de la structure du projet : 

- **converted_data** (dossier) : Contient les données de départ converties au format .parquet afin de rendre l'ouverture et le téléchargement des données plus rapide. 
- **data** (dossier) : Contient les données de départ auxquelles la 2ème ligne a manuellement été retirée afin de rendre la lecture et la conversion en .parquet plus facile. Ces données n'ont pas été mise en ligne sur GitHub à cause de leur taille trop volumineuse. 
- **documentation** (dossier) : Contient le rapport écrit du cas au format Word. Contient aussi la documentation utilisée pour le cas. 
- **figures** (dossier) : Contient les graphiques utilisés dans le projet. Note : La majeure partie des figures se trouvent sous la forme d'un fichier html dans le dossier **results_data**. 
- **filtered_data** (dossier) : Contient les données après filtration par le filtre de liquidité. Les extensions sont en .parquet. 
- **results_data** (dossier) : Contient les résultats les plus importants du projet. On y retrouve les métriques de performance (avec de nombreux graphiques) sous la forme d'une page hmtl pour les 3 stratégies testées (base_strategy, new_strategy et other_strategy). Les fichiers html sont censés pouvoir être ouverts à partir de l'éditeur de code ou avec un utilitaire de fichier. On y retrouve aussi la répartition des poids annuels concernant les secteurs GICS, les poids de rééquilibrage annuels ainsi que les "daily drifted weights" pour chacune des stratégies. Les plus gros fichiers sont en `parquet` afin de faciliter le téléchargement et la lecture des fichiers. Si vous souhaitez accéder à ces fichiers, vous pouvez utiliser la librairie `pandas` ou encore les convertir sur votre machine locale en `csv`. 
- **backtesting.py** (fichier) : Script qui contient le calcul des "daily drifted weights" ainsi que la comparaison des performances des différentes stratégies par rapport au benchmark. Les outputs sont ensuite sauvergardés dans le dossier results_data. 
- **base_strategy.py** (fichier) : Ce script exécute la stratégie de la volatilité inverse de base, en prenant en compte des contraintes sur les poids de manière individuel et des contraintes sectorielles. Le détail de la stratégie se trouve en haut du script. Le script contient aussi des fonctions qui permettent de vérifier que les critères ont bien été respectés, comme par exemple le fait que les poids doivent sommer à 1, ou encore que les contraintes d'inégalités sont respectées. Si les contraintes ne sont pas respectées, un message d'erreur apparaît dans la console (l'allocation est quand même obtenue). Pour certains paramètres initiaux, il peut arriver que les contraintes ne soient légèrement pas respectées. La performance de cette stratégie est comparé au benchmark dans le dossier **results_data** avec le fichier html : `base_strategy_metrics.html`. 
- **eda.ipynb** (fichier) : Fichier Jupyter Notebook qui permet de faire de l'analyse exploratoire des données (visualisation des DataFrames + vérification de certains résultats). 
- **plot.py** (fichier) : Ce script permet de générer les graphiques de répartition sectorielle des poids pour les trois différentes stratégies. Les figures sont ensuite sauvergardés dans le dossier `figures`. 
- **preprocessing.py** (fichier) : Ce script m'a été utile pour convertir les fichiers au format `xlsx` au format `parquet` et faire en sorte que les index soient au format DateTime pour tous les fichiers de données. Pour rappel, la deuxième ligne des fichiers `xlsx` a manuellement été retirée afin de rendre l'ouverture et la conversion plus facile. Il est conseillé d'accéder directement aux données via le dossier `converted_data`au format `parquet` pour tester les scripts. 
- **filter.py** (fichier) : Ce script contient le filtre de liquidité discuté dans le rapport de la stratégie. Pour rappel, ce script ne conserve que les actions qui se trouvent dans le quantile de liquidité annuel supérieur, avec des NaN values pour les autres titres dans l'univers du S&P/TSX. Après avoir obtenu les titres les plus liquides, j'applique le filtre de liquidité aux autres jeux de données. 
- **new_strategy.py** (fichier) : Ce script exécute la stratégie de volatilité inverse avec skewness. Le détail de la stratégie se trouve en haut du script en docstring. Cette stratégie d'investissement est basée sur l'inverse de la volatilité et la skewness positive des actifs du portefeuille. Pour résumer, cette stratégie vise à attribuer des poids aux actifs en tenant compte de ces deux mesures. La performance de cette stratégie est comparé au benchmark dans le dossier **results_data** avec le fichier html : `new_strategy_metrics.html`. 
- **other_strategy.py** (fichier) : Ce script exécute la stratégie du portefeuille à volatilité minimale de Markovitz avec bootstraping. Le détail de la stratégie se trouve en haut du script. Après une longue période d'optimisation, j'ai pu observer que les contraintes sectorielles n'avaient pas été respectées pour le secteur `#NA NA` que j'ai renommé en `Not Attributed` dans le script **preprocessing**. Par manque de temps, je n'ai pas pu corriger cette erreur sur les contraintes sectorielles. Cependant, les contraintes sur les poids individuels ont bien été respectées et les autres secteurs sont bornés par la contrainte.   
