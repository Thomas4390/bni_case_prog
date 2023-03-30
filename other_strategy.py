import pandas as pd
import numpy as np
from scipy.optimize import minimize
from base_strategy import calculate_returns, get_rebalance_dates, calculate_sector_weights


def create_factor_matrix(gics_sectors: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une matrice de facteurs à partir des secteurs GICS.

    Parameters
    ----------
    gics_sectors : pandas.DataFrame
        Un DataFrame contenant les secteurs GICS pour chaque actif.

    Returns
    -------
    pandas.DataFrame
        Une matrice de facteurs pour chaque actif, où chaque colonne représente un secteur GICS.
    """
    # Convertit les secteurs GICS en dummies pour créer une matrice de facteurs
    return pd.get_dummies(gics_sectors.set_index('Ticker')['GICS Sector'])


def calculate_factor_returns(rendements: pd.DataFrame, factor_matrix: pd.DataFrame) -> pd.Series:
    """
    Calcule les rendements des facteurs à partir des rendements des actifs et de la matrice de facteurs.

    Parameters
    ----------
    rendements : pandas.DataFrame
        Un DataFrame contenant les rendements pour chaque actif.
    factor_matrix : pandas.DataFrame
        Une matrice de facteurs pour chaque actif.

    Returns
    -------
    pandas.Series
        Une série contenant les rendements des facteurs pour chaque date.
    """
    # Multiplie les rendements des actifs par la matrice de facteurs et divise par la somme des facteurs pour chaque date
    return rendements.dot(factor_matrix).div(factor_matrix.sum())


def calculate_asset_covariance(factor_returns: pd.Series, factor_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la matrice de covariance des actifs à partir des rendements des facteurs et de la matrice de facteurs.

    Parameters
    ----------
    factor_returns : pandas.Series
        Une série contenant les rendements des facteurs pour chaque date.
    factor_matrix : pandas.DataFrame
        Une matrice de facteurs pour chaque actif.

    Returns
    -------
    pandas.DataFrame
        Une matrice de covariance des actifs.
    """
    # Calcule la covariance des facteurs
    factor_covariance = factor_returns.cov()
    # Transpose la matrice de facteurs
    asset_exposures = factor_matrix.T
    # Calcule la matrice de covariance des actifs en multipliant les expositions des actifs par la covariance des facteurs
    return asset_exposures.T.dot(factor_covariance).dot(asset_exposures)


def portfolio_variance(weights: np.ndarray, asset_covariance: pd.DataFrame) -> float:
    """
    Calcule la variance du portefeuille à partir des poids des actifs et de la matrice de covariance des actifs.

    Parameters
    ----------
    weights : numpy.ndarray
        Un tableau numpy contenant les poids pour chaque actif.
    asset_covariance : pandas.DataFrame
        Une matrice de covariance des actifs.

    Returns
    -------
    float
        La variance du portefeuille.
    """
    # Calcule la variance du portefeuille en multipliant les poids des actifs par la matrice de covariance des actifs
    return weights.T.dot(asset_covariance).dot(weights)


def portfolio_return(weights: np.ndarray, mean_returns: pd.Series) -> float:
    """
    Calcule le rendement du portefeuille à partir des poids des actifs et des rendements moyens des actifs.

    Parameters
    ----------
    weights : numpy.ndarray
        Un tableau numpy contenant les poids pour chaque actif.
    mean_returns : pandas.Series
        Une série contenant les rendements moyens pour chaque actif.

    Returns
    -------
    float
        Le rendement du portefeuille.
    """
    # Calcule le rendement du portefeuille en multipliant les poids des actifs par les rendements moyens des actifs
    return weights.T.dot(mean_returns)


def negative_sharpe_ratio(weights: np.ndarray, mean_returns: pd.Series, asset_covariance: pd.DataFrame,
                          risk_free_rate: float = 0.02) -> float:
    """
    Calcule le ratio de Sharpe négatif pour un portefeuille à partir des poids des actifs, des rendements moyens des actifs, de la matrice de covariance des actifs et du taux sans risque.

    Parameters
    ----------
    weights : numpy.ndarray
        Un tableau numpy contenant les poids pour chaque actif.
    mean_returns : pandas.Series
        Une série contenant les rendements moyens pour chaque actif.
    asset_covariance : pandas.DataFrame
        Une matrice de covariance des actifs.
    risk_free_rate : float, optional
        Le taux sans risque. Default is 0.02.

    Returns
    -------
    float
        Le ratio de Sharpe négatif pour le portefeuille.
    """
    # Calcule le rendement et la variance du portefeuille
    port_return = portfolio_return(weights, mean_returns)
    port_variance = portfolio_variance(weights, asset_covariance)
    # Calcule le ratio de Sharpe négatif en utilisant le rendement et la variance du portefeuille ainsi que le taux sans risque
    return -(port_return - risk_free_rate) / np.sqrt(port_variance)


def sum_weights_constraint(weights: np.ndarray) -> float:
    """
    Contrainte pour s'assurer que les poids des actifs somment à 1.

    Parameters
    ----------
    weights : numpy.ndarray
        Un tableau numpy contenant les poids pour chaque actif.

    Returns
    -------
    float
        La somme des poids des actifs, qui doit être égale à 1.
    """
    # Retourne la somme des poids des actifs, qui doit être égale à 1
    return np.sum(weights) - 1


def optimize_portfolio(rendements: pd.DataFrame, asset_covariance: pd.DataFrame, sector_constraints: list,
                       individual_constraints: float) -> np.ndarray:
    """
    Optimise les poids des actifs pour maximiser le ratio de Sharpe en utilisant
    l'optimiseur de SciPy.

    Parameters
    ----------
    rendements : pandas.DataFrame
        Un DataFrame contenant les rendements des actifs.
    asset_covariance : pandas.DataFrame
        Une matrice de covariance des actifs.
    sector_constraints : list
        Une liste de contraintes de secteur qui doivent être respectées.
    individual_constraints : float
        La limite individuelle pour chaque poids d'actif.

    Returns
    -------
    numpy.ndarray
        Les poids des actifs optimisés.
    """
    # Initialise les poids des actifs à des valeurs égales pour tous les actifs
    initial_weights = np.ones(len(rendements.columns)) / len(rendements.columns)
    # Définit les bornes pour les poids des actifs
    bounds = [(0, individual_constraints) for _ in range(len(rendements.columns))]
    # Définit les contraintes pour s'assurer que les poids des actifs respectent les limites et les contraintes de secteur
    constraints = [{'type': 'eq', 'fun': sum_weights_constraint}] + sector_constraints
    # Applique la méthode de minimisation de SciPy pour obtenir les poids des actifs optimisés
    result = minimize(negative_sharpe_ratio, initial_weights,
                      args=(rendements.mean(), asset_covariance),
                      bounds=bounds, constraints=constraints, method='SLSQP')
    return result.x


def rebalance_portfolio(rendements: pd.DataFrame, gics_sectors: pd.DataFrame, rebalance_dates: pd.DatetimeIndex,
                        sector_limit: float = 0.4, individual_limit: float = 0.05) -> pd.DataFrame:
    """
    Rééquilibre un portefeuille en utilisant la méthode de l'optimisation des poids des actifs pour maximiser le ratio de Sharpe.

    Parameters
    ----------
    rendements : pandas.DataFrame
        Un DataFrame contenant les rendements des actifs.
    gics_sectors : pandas.DataFrame
        Un DataFrame contenant les secteurs GICS pour chaque actif.
    rebalance_dates : pandas.DatetimeIndex
        Une DatetimeIndex contenant les dates de rééquilibrage.
    sector_limit : float, optional
        La limite de poids de secteur pour chaque secteur. Default is 0.4.
    individual_limit : float, optional
        La limite de poids d'actif pour chaque actif. Default is 0.05.

    Returns
    -------
    pandas.DataFrame
        Un DataFrame contenant les poids des actifs rééquilibrés pour chaque date de rééquilibrage.
    """
    # Crée une matrice de facteurs pour les secteurs GICS
    factor_matrix = create_factor_matrix(gics_sectors)
    asset_weights = []

    for idx, rebalance_date in enumerate(rebalance_dates):
        # Définit la période pour le calcul des rendements et des facteurs
        end_date = rebalance_date - pd.DateOffset(days=1)
        start_date = end_date - pd.DateOffset(years=1) if idx == 0 else rebalance_dates[idx - 1]
        # Sélectionne les colonnes avec des valeurs non nulles
        rendements_period = rendements.loc[start_date:end_date].dropna(axis=1)
        # Sélectionne les facteurs correspondant aux actifs sélectionnés
        factor_matrix_period = factor_matrix.loc[rendements_period.columns]
        # Calcule les rendements des facteurs
        factor_returns = calculate_factor_returns(rendements_period,
                                                  factor_matrix_period)
        # Calcule la matrice de covariance des actifs en fonction des rendements des facteurs
        asset_covariance = calculate_asset_covariance(factor_returns,
                                                      factor_matrix_period)

        # Définit les contraintes de secteur pour chaque secteur
        sector_constraints = []
        for sector in gics_sectors['GICS Sector'].unique():
            index = (gics_sectors['GICS Sector'] == sector) & gics_sectors[
                'Ticker'].isin(rendements_period.columns)
            constraint = {'type': 'ineq', 'fun': lambda w, i=index.loc[
                rendements_period.columns]: sector_limit - np.sum(w[i])}
            sector_constraints.append(constraint)

        # Définit les bornes pour les poids des actifs
        bounds = [(0, individual_limit) for _ in
                  range(len(rendements_period.columns))]
        # Applique l'optimisation pour obtenir les poids des actifs rééquilibrés
        optimal_weights = optimize_portfolio(rendements_period,
                                             asset_covariance,
                                             sector_constraints,
                                             individual_limit)

        # Crée une série de poids pour chaque actif
        asset_weights_period = pd.Series(np.zeros(len(rendements.columns)),
                                         index=rendements.columns)
        asset_weights_period[rendements_period.columns] = optimal_weights
        # Ajoute les poids des actifs rééquilibrés pour la période actuelle
        asset_weights.append(asset_weights_period.values)

    # Crée un DataFrame des poids des actifs rééquilibrés pour chaque date de rééquilibrage
    return pd.DataFrame(asset_weights, index=rebalance_dates,
                        columns=rendements.columns)


df_total_ret_filtered = pd.read_parquet("filtered_data/total_ret_data.parquet")
print(df_total_ret_filtered)
rendements = calculate_returns(df_total_ret_filtered)
rebalance_dates = get_rebalance_dates(rendements)
gics_sectors = pd.read_parquet("converted_data/Constituents GICS sectors.parquet")

# Exécuter la fonction de rééquilibrage du portefeuille avec les rendements, les secteurs GICS et les dates de rééquilibrage
optimized_weights = rebalance_portfolio(rendements, gics_sectors, rebalance_dates)

print(optimized_weights)
print(calculate_sector_weights(optimized_weights, gics_sectors))

