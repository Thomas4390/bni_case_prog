import pandas as pd
import numpy as np
from preprocessing import read_data
import quantstats as qs


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


# def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
#     """
#     Calcule les rendements cumulés sur toute la période pour une série de rendements.
#
#     Parameters
#     ----------
#     returns : pd.Series
#         Série contenant les rendements de l'actif ou du portefeuille pour chaque période.
#
#     Returns
#     -------
#     pd.Series
#         Série contenant les rendements cumulés pour chaque période.
#     """
#     # Calcule les rendements cumulés
#     cumulative_returns = (1 + returns).cumprod()
#     return cumulative_returns
#
#
# def compute_performance_metrics(returns: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
#     """
#     Calcule les métriques de performance pour différentes périodes.
#
#     Parameters
#     ----------
#     returns : pd.Series
#         Série contenant les rendements quotidiens.
#     alpha : float, optional
#         Niveau de risque à utiliser pour le calcul de VaR et CVaR.
#
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame contenant les métriques de performance pour différentes périodes.
#
#     """
#
#     # Initialiser le DataFrame pour stocker les résultats
#     index = ["1 mois", "3 mois", "6 mois", "1 an", "2 ans", "3 ans", "5 ans",
#              "10 ans", "20 ans"]
#     columns = ["Rendement", "Volatilité", "Ratio de Sharpe",
#                f"VaR à {alpha * 100:.0f}%", f"CVaR à {alpha * 100:.0f}%"]
#     metrics = pd.DataFrame(index=index, columns=columns)
#
#     # Périodes en jours pour les rendements
#     periods = [21, 63, 126, 252, 504, 756, 1260, 2520, 5040]
#
#     # Calculer les métriques pour chaque période
#     for i, period in enumerate(periods):
#         # Vérifier si la période dépasse la taille de la série de rendements
#         if period > len(returns):
#             break
#
#         # Calculer la volatilité
#         period_returns = returns[-period:]
#         compounded_return = (1 + returns.tail(period)).prod() - 1
#         metrics.loc[index[i], "Rendement"] = compounded_return
#
#         volatility = np.std(period_returns) * np.sqrt(period)
#         metrics.loc[index[i], "Volatilité"] = volatility
#
#         # Calculer le ratio de Sharpe
#         sharpe_ratio = np.mean(period_returns) / np.std(
#             period_returns) * np.sqrt(period)
#         metrics.loc[index[i], "Ratio de Sharpe"] = sharpe_ratio
#
#         # Calculer la VaR à 5%
#         var_5 = -np.percentile(period_returns, 100 * alpha)
#         metrics.loc[index[i], f"VaR à {alpha * 100:.0f}%"] = var_5
#
#         # Calculer la CVaR à 5%
#         cvar_5 = -np.mean(period_returns[period_returns <= var_5])
#         metrics.loc[index[i], f"CVaR à {alpha * 100:.0f}%"] = cvar_5
#
#     return metrics
#
#


if __name__ == "__main__":
    # Lecture des données
    df_total_ret = pd.read_parquet("filtered_data/total_ret_data.parquet")
    print(df_total_ret.head())

    df_weights = pd.read_parquet("results_data/base_strategy_weights.parquet")
    benchmark_prices = read_data("Constituents TOT_RET_INDEX data").iloc[:, -1]

    # Calcul des rendements quotidiens du portefeuille
    portfolio_daily_returns = compute_daily_portfolio_returns(df_total_ret,
                                                              df_weights)

    benchmark_daily_returns = compute_benchmark_returns(benchmark_prices, weights=df_weights)

    # Calcul des métriques de performance en utilisant le package quantstats
    qs.extend_pandas()
    # output sous la forme d'un fichier html à ouvrir sur un web browser
    backtesting_metrics = qs.reports.html(portfolio_daily_returns,
                                          benchmark_daily_returns,
                                          mode="full",
                                          title="Backtesting Base Strategy",
                                          output=True,
                                          download_filename="base_strategy_metrics.html")



