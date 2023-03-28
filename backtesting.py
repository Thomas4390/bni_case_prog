import pandas as pd
import numpy as np
from preprocessing import read_data


def get_rebalance_dates(weights: pd.DataFrame) -> pd.DatetimeIndex:
    """Retourne les dates de rééquilibrage à partir d'un DataFrame de poids."""
    return weights.index


def compute_daily_portfolio_returns(prices: pd.DataFrame,
                                      weights: pd.DataFrame) -> pd.Series:
    """
    Calcule les rendements quotidiens du portefeuille en utilisant les prix et les poids.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame contenant les prix ajustés quotidiens pour chaque action.
        Les colonnes sont les noms des actions et l'index est la date en jours.
    weights : pd.DataFrame
        DataFrame contenant les poids annuels pour chaque action. Les colonnes
        sont les noms des actions et l'index est la date en années.

    Returns
    -------
    pd.Series
        Série contenant les rendements quotidiens du portefeuille.
    """
    # Calcule les rendements quotidiens
    daily_returns = prices.pct_change()
    # Ne garder que les rendements à partir de la première date de rééquilibrage
    daily_returns = daily_returns.loc[get_rebalance_dates(weights)[0]:]

    # Remplir les poids pour chaque jour de trading
    daily_weights = weights.reindex(daily_returns.index, method="ffill")

    # Calculer les rendements quotidiens du portefeuille
    portfolio_daily_returns = (daily_weights * daily_returns).sum(axis=1)

    return portfolio_daily_returns


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calcule les rendements cumulés sur toute la période pour une série de rendements.

    Parameters
    ----------
    returns : pd.Series
        Série contenant les rendements de l'actif ou du portefeuille pour chaque période.

    Returns
    -------
    pd.Series
        Série contenant les rendements cumulés pour chaque période.
    """
    # Calcule les rendements cumulés
    cumulative_returns = (1 + returns).cumprod() - 1
    return cumulative_returns


def compute_benchmark_returns(benchmark: pd.Series, weights: pd.DataFrame) -> pd.Series:
    """
    Calcule la série de rendements du benchmark à partir de la première date de rééquilibrage.

    Parameters
    ----------
    benchmark : pd.Series
        Série contenant les prix ajustés du benchmark.
    weights : pd.DataFrame
        DataFrame contenant les poids du portefeuille à chaque date de rééquilibrage.

    Returns
    -------
    pd.Series
        Série contenant la série de rendements du benchmark à partir de la première date de rééquilibrage.
    """

    # Récupérer la première date de rééquilibrage
    rebalance_dates = get_rebalance_dates(weights)
    first_rebalance_date = rebalance_dates[0]

    # Convertir l'index en DatetimeIndex s'il ne l'est pas déjà
    if not isinstance(benchmark.index, pd.DatetimeIndex):
        benchmark.index = pd.to_datetime(benchmark.index)

    # Convertir la date de rééquilibrage en Timestamp s'il ne l'est pas déjà
    if not isinstance(first_rebalance_date, pd.Timestamp):
        first_rebalance_date = pd.Timestamp(first_rebalance_date)

    # Calculer les rendements quotidiens du benchmark
    benchmark_returns = benchmark.pct_change().dropna()
    # Ne garder que les rendements à partir de la première date de rééquilibrage
    benchmark_returns = benchmark_returns.loc[first_rebalance_date:]

    return benchmark_returns


if __name__ == "__main__":

    # Lecture des données
    df_total_ret = pd.read_parquet("filtered_data/total_ret_data.parquet")

    df_weights = pd.read_parquet("results_data/base_strategy_weights.parquet")
    benchmark_prices = read_data("Constituents TOT_RET_INDEX data").iloc[:, -1]
    print(benchmark_prices)
    print(type(benchmark_prices))

    # Calcul des rendements quotidiens du portefeuille
    portfolio_daily_returns = compute_daily_portfolio_returns(df_total_ret,
                                                              df_weights)
    portfolio_cumul_returns = compute_cumulative_returns(
        portfolio_daily_returns)
    benchmark_daily_returns = compute_benchmark_returns(benchmark_prices, weights=df_weights)

    # Afficher les rendements du portefeuille
    print("Rendements quotidiens du portefeuille")
    print(portfolio_daily_returns)
    print("Rendements cumulés du portefeuille")
    print(portfolio_cumul_returns)
    print("Rendements quotidiens du benchmark")
    print(benchmark_daily_returns)