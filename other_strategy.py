from typing import Tuple

import pandas as pd
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from base_strategy import (
    calculate_sector_weights,
    calculate_returns,
    check_sector_constraints,
    check_weights_sum_to_one,
)
from filter import get_rebalance_dates
from concurrent.futures import ThreadPoolExecutor

"""La stratégie présentée ici est basée sur la méthode de l'échantillonnage bootstrap 
pour générer un portefeuille optimal. Cette méthode vise à minimiser la variance 
du portefeuille sous contraintes sectorielles. Voici un aperçu du fonctionnement de cette stratégie:

1. Pour chaque période de rééquilibrage :
a. Extraire les rendements des actifs pour la période courante.
b. Supprimer les colonnes contenant des NaN.
c. Optimiser le portefeuille en utilisant la méthode de l'échantillonnage bootstrap.
d. Réintégrer les colonnes supprimées avec des poids de 0.
e. Ajouter les poids pour la période courante à la liste des poids de rééquilibrage.

2. Convertir la liste des poids de rééquilibrage en DataFrame.
La méthode de l'échantillonnage bootstrap est utilisée pour estimer la distribution 
des rendements en générant des échantillons à partir des données historiques. 
Cette approche permet de prendre en compte l'incertitude liée à l'estimation 
des paramètres de la distribution.

La fonction resampled_ef_portfolio effectue les étapes suivantes pour générer 
un portefeuille optimal :

1. Générer n_bootstrap échantillons bootstrap de la distribution des rendements 
en utilisant une normale multivariée.
Pour chaque échantillon bootstrap :
a. Calculer la matrice de covariance des rendements de l'échantillon.
b. Trouver les poids du portefeuille avec la variance minimale en utilisant la 
matrice de covariance et les contraintes sectorielles.

2. Moyenner les poids optimaux sur les échantillons bootstrap.

En résumé, cette stratégie utilise la méthode de l'échantillonnage bootstrap 
pour estimer la distribution des rendements et générer un portefeuille optimal 
en minimisant la variance sous contraintes sectorielles. Le but est de construire 
un portefeuille diversifié qui minimise le risque.

Attention: cette stratégie prend beaucoup de temps à tourner."""

def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> ndarray:
    """
    Calcule la variance d'un portefeuille à partir des poids et de la matrice de covariance.

    Parameters
    ----------
    weights : numpy ndarray
        Vecteur de poids du portefeuille.

    cov_matrix : numpy ndarray
        Matrice de covariance des rendements.

    Returns
    -------
    numpy ndarray
        La variance du portefeuille.

    """

    return np.dot(weights.T, np.dot(cov_matrix, weights))


def get_bootstrap_samples(returns: pd.DataFrame, n_samples: int = 252) -> np.ndarray:
    """
    Génère des échantillons bootstrap de la distribution des rendements.
    On utilise une normale multivariée pour générer les échantillons.

    Parameters
    ----------
    returns : pandas DataFrame
        DataFrame contenant les rendements des actifs.

    n_samples : int, optional
        Nombre d'échantillons à générer, par défaut 252.

    Returns
    -------
    numpy ndarray
        Les échantillons bootstrap des rendements.

    """

    cov_matrix = returns.cov().to_numpy() + np.eye(returns.shape[1]) * 1e-8
    mean_returns = returns.mean().to_numpy()
    return np.random.multivariate_normal(mean_returns, cov_matrix, n_samples)


def min_variance_portfolio(
    cov_matrix: np.ndarray,
    gics_sectors: pd.DataFrame,
    sector_constraints: Tuple[float, float],
    period_returns_clean: pd.DataFrame,
) -> np.ndarray:
    """
    Trouve les poids du portefeuille avec la variance minimale sous contraintes.

    Parameters
    ----------
    cov_matrix : numpy ndarray
        Matrice de covariance des rendements.

    gics_sectors : pandas DataFrame
        DataFrame contenant les classifications sectorielles GICS des actifs.

    sector_constraints : tuple of two floats
        Limites de pondération sectorielle pour les poids des actions.

    period_returns_clean : pandas DataFrame
        DataFrame contenant les rendements des actifs.

    Returns
    -------
    numpy ndarray
        Les poids du portefeuille avec la variance minimale.

    """

    n_assets = cov_matrix.shape[0]

    # Initialiser les poids de manière égale
    initial_weights = np.ones(n_assets) / n_assets

    # Bornes pour les poids
    bounds = [(0, 1) for _ in range(n_assets)]

    # Contrainte globale pour la somme des poids égale à 1
    total_weight_constraint = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # Contraintes sectorielles pour les poids des actions
    sector_constraint = []
    unique_sectors = gics_sectors["GICS Sector"].unique()
    for sector in unique_sectors:
        sector_indices = (
            gics_sectors.loc[gics_sectors["GICS Sector"] == sector]
            .index.intersection(period_returns_clean.columns)
            .tolist()
        )
        sector_constraint.append(
            {
                "type": "ineq",
                "fun": lambda x, s=sector_indices: np.sum(x[s]) - sector_constraints[0],
            }
        )
        sector_constraint.append(
            {
                "type": "ineq",
                "fun": lambda x, s=sector_indices: sector_constraints[1] - np.sum(x[s]),
            }
        )

    # Contraintes totales
    constraints = [total_weight_constraint] + sector_constraint

    # Minimiser la variance
    result = minimize(
        portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        bounds=bounds,
        constraints=constraints,
    )

    return result.x


def resampled_ef_portfolio(
    returns: pd.DataFrame,
    gics_sectors: pd.DataFrame,
    sector_constraints: Tuple[float, float],
    n_bootstrap: int = 3,
) -> np.ndarray:
    """
    Génère un portefeuille optimal en utilisant la méthode de l'échantillonnage bootstrap.

    Parameters
    ----------
    returns : pandas DataFrame
        DataFrame contenant les rendements des actifs.

    gics_sectors : pandas DataFrame
        DataFrame contenant les classifications sectorielles GICS des actifs.

    sector_constraints : tuple of two floats
        Limites de pondération sectorielle pour les poids des actions.

    n_bootstrap : int, optional
        Nombre d'échantillons bootstrap à générer, par défaut 3.

    Returns
    -------
    numpy ndarray
        Les poids du portefeuille optimal.

    """

    def single_bootstrap_iteration(_, returns: pd.DataFrame):
        # Générer un échantillon bootstrap de la distribution des rendements
        bootstrap_returns = get_bootstrap_samples(returns)

        # Calculer la matrice de covariance pour l'échantillon bootstrap
        bootstrap_cov_matrix = np.cov(bootstrap_returns.T)

        # Trouver les poids du portefeuille avec la variance minimale pour l'échantillon bootstrap
        return min_variance_portfolio(
            bootstrap_cov_matrix, gics_sectors, sector_constraints, returns
        )

    # Utiliser ThreadPoolExecutor pour exécuter plusieurs itérations bootstrap en parallèle
    with ThreadPoolExecutor() as executor:
        optimal_weights_array = list(
            executor.map(
                single_bootstrap_iteration, range(n_bootstrap), [returns] * n_bootstrap
            )
        )

    # Moyenner les poids optimaux sur les itérations bootstrap
    optimal_weights = np.sum(optimal_weights_array, axis=0) / n_bootstrap

    return optimal_weights


def resampled_ef_weights_by_rebalance_dates(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    gics_sectors: pd.DataFrame,
    sector_constraints: Tuple[float, float],
) -> pd.DataFrame:
    """
    Calcule les poids du portefeuille optimal pour chaque période de
    rééquilibrage à l'aide de la méthode de l'échantillonnage bootstrap.

    Parameters
    ----------
    returns : pandas DataFrame
        DataFrame contenant les rendements des actifs.

    rebalance_dates : pandas DatetimeIndex
        Dates de rééquilibrage du portefeuille.

    gics_sectors : pandas DataFrame
        DataFrame contenant les classifications sectorielles GICS des actifs.

    sector_constraints : tuple of two floats
        Limites de pondération sectorielle pour les poids des actions.

    Returns
    -------
    pandas DataFrame
        Les poids du portefeuille optimal pour chaque période de rééquilibrage.

    """

    # Liste des poids du portefeuille pour chaque période de rééquilibrage
    rebalance_weights = []

    # Itérer sur chaque période de rééquilibrage et calculer les poids optimaux
    for start_date, end_date in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        print(f"Calculating weights for period {start_date} to {end_date}...")

        # Extraire les rendements pour la période courante
        period_returns = returns.loc[start_date:end_date]

        # Supprimer les colonnes avec des NaN values
        period_returns_clean = period_returns.dropna(axis=1)

        # Optimiser le portefeuille avec les colonnes restantes
        optimal_weights = resampled_ef_portfolio(
            period_returns_clean, gics_sectors, sector_constraints
        )

        # Réintégrer les colonnes supprimées avec des poids de 0
        full_weights = pd.Series(0, index=returns.columns)
        full_weights[period_returns_clean.columns] = optimal_weights

        # Ajouter les poids pour la période courante à la liste
        rebalance_weights.append(full_weights.values)
        print(f"Optimal weights: {full_weights}")

    # Convertir la liste de poids en DataFrame
    rebalance_weights_df = pd.DataFrame(
        rebalance_weights, index=rebalance_dates[:-1], columns=returns.columns
    )

    return rebalance_weights_df


if __name__ == "__main__":
    # Paramètres
    sector_max_weight = 0.4

    df_total_ret_filtered = pd.read_parquet("filtered_data/total_ret_data.parquet")
    rendements = calculate_returns(df_total_ret_filtered)
    rebalance_dates = get_rebalance_dates(rendements)
    gics_sectors = pd.read_parquet("converted_data/Constituents GICS sectors.parquet")

    weights = resampled_ef_weights_by_rebalance_dates(
        rendements,
        gics_sectors=gics_sectors,
        sector_constraints=(0.05, 0.4),
        rebalance_dates=rebalance_dates,
    )
    weights.to_parquet("results_data/other_strategy_weights.parquet")

    weights = pd.read_parquet("results_data/other_strategy_weights.parquet")
    sector_weights = calculate_sector_weights(weights=weights, df_sectors=gics_sectors)

    sector_weights.to_parquet("results_data/other_strategy_sector_weights.parquet")

    are_sectors_constraints_respected = check_sector_constraints(
        weights=weights,
        df_sectors=gics_sectors,
        verbose=True,
        sector_max_weight=sector_max_weight,
    )

    if are_sectors_constraints_respected:
        print("\nToutes les contraintes sectorielles sont respectées.")
    else:
        print("\nDes erreurs ont été trouvées dans les contraintes sectorielles.")

    sum_to_one = check_weights_sum_to_one(weights)

    if sum_to_one:
        print("\nLes poids somment à 1 pour chaque date de rééquilibrage.")
    else:
        print("\nLes poids ne somment pas à 1 pour certaines dates de rééquilibrage.")
