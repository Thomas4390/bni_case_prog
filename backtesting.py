import pandas as pd
import numpy as np
from preprocessing import read_data, move_file_to_directory
import quantstats as qs
import warnings
# supress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)


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

    # Calcul des rendements quotidiens du portefeuille
    portfolio_daily_returns = compute_daily_portfolio_returns(df_total_ret,
                                                              df_weights)

    benchmark_daily_returns = compute_benchmark_returns(benchmark_prices,
                                                        weights=df_weights)

    # Calcul des métriques de performance en utilisant le package quantstats
    qs.extend_pandas()
    # output sous la forme d'un fichier html à ouvrir sur un web browser
    print("Début de la génération du rapport de backtesting...")
    backtesting_metrics = qs.reports.html(portfolio_daily_returns,
                                          benchmark_daily_returns,
                                          rf=0.01,
                                          mode="full",
                                          title="Backtesting Base Strategy",
                                          output=True,
                                          download_filename="base_strategy_metrics.html",
                                          match_dates=True)
    print("Rapport de backtesting généré avec succès!")

    move_file_to_directory("base_strategy_metrics.html", "results_data")



