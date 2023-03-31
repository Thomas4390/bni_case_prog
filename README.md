# Cas de programmation | BNI

# Description de la structure du projet : 

- **converted_data** (dossier) : Contient les données de départ converties au format .parquet afin de rendre l'ouverture et le téléchargement des données plus rapide. 
- **data** (dossier) : Contient les données de départ auxquelles la 2ème ligne a manuellement été retirée afin de rendre la lecture et la conversion en .parquet plus facile. Ces données n'ont pas été mise en ligne sur GitHub à cause de leur taille trop volumineuse. 
- **documentation** (dossier) : Contient la documentation utilisée pour le cas. 
- **figures** (dossier) : Contient les graphiques utilisés dans le projet. Note : La majeure partie des figures se trouvent sous la forme d'un fichier html dans le dossier **results_data**. 
- **filtered_data** (dossier) : Contient les données après filtration par le filtre de liquidité. Les extensions sont en .parquet. 
- **results_data** (dossier) : Contient les résultats les plus importants du projet. On y retrouve les métriques de performance (avec de nombreux graphiques) sous la forme d'une page hmtl pour les 3 stratégies testées (base_strategy, new_strategy et other_strategy). On y retrouve aussi la répartition des poids annuels concernant les secteurs GICS, les poids de rééquilibrage annuels ainsi que les "daily drifted weights" pour chacune des stratégies. 
- backtesting.py (fichier) : Script qui contient le calcul des "daily drifted weights" ainsi que la comparaison des performances des différentes stratégies par rapport au benchmark. Les outputs sont ensuite sauvergardés dans le dossier results_data. 
