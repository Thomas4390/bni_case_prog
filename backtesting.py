import pandas as pd
from preprocessing import read_data, move_file_to_directory
import quantstats as qs
import warnings

# supress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def get_rebalance_dates(weights: pd.DataFrame) -> pd.DatetimeIndex:
    """Retourne les dates de rééquilibrage à partir d'un DataFrame de poids."""
    return weights.index


def calculate_daily_drifted_weights(
    weights: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcule les poids quotidiens ajustés en fonction des rendements quotidiens
    et des dates de rééquilibrage du portefeuille.

    Parameters
    ----------
    weights : pandas DataFrame
        DataFrame contenant les poids du portefeuille à chaque date de rééquilibrage.

    prices : pandas DataFrame
        DataFrame contenant les prix quotidiens des actifs.

    Returns
    -------
    pandas DataFrame
        DataFrame contenant les poids ajustés pour chaque jour et chaque actif.

    """

    # Calcule les rendements quotidiens
    daily_returns = prices.pct_change()

    # Trouve la première date de rééquilibrage
    first_rebalance_date = weights.index[0]

    # Sélectionne les rendements quotidiens à partir de la première date de rééquilibrage
    daily_returns = daily_returns.loc[first_rebalance_date:]

    # Initialise le DataFrame drifted_weights avec la même forme que daily_returns
    drifted_weights = pd.DataFrame(index=daily_returns.index, columns=weights.columns)

    # Initialise les poids du portefeuille avec les poids de la première date de rééquilibrage
    current_weights = weights.loc[first_rebalance_date]

    # Remplit les valeurs initiales de drifted_weights avec les poids du premier jour
    drifted_weights.loc[first_rebalance_date] = current_weights

    # Calcule les poids quotidiens ajustés pour chaque jour et chaque actif
    for i in range(1, len(daily_returns)):
        # Si la date courante est une date de rééquilibrage
        if daily_returns.index[i] in weights.index:
            # Mettre à jour les poids du portefeuille avec les poids de rééquilibrage correspondants
            current_weights = weights.loc[daily_returns.index[i]]

        # Sinon, calculer les poids ajustés à partir des poids courants et des rendements quotidiens
        else:
            daily_change = current_weights * (1 + daily_returns.iloc[i])
            current_weights = daily_change / daily_change.sum()

        # Ajouter les poids courants à drifted_weights
        drifted_weights.iloc[i] = current_weights

    return drifted_weights


def compute_daily_portfolio_returns(
    prices: pd.DataFrame, weights: pd.DataFrame
) -> pd.Series:
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
    daily_returns = daily_returns.loc[get_rebalance_dates(weights)[0] :]

    # Remplir les poids pour chaque jour de trading avec les daily drifted weights
    daily_weights = calculate_daily_drifted_weights(weights, prices)

    # Calculer les rendements quotidiens du portefeuille
    portfolio_daily_returns = (daily_weights.shift(1) * daily_returns).sum(axis=1)

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
    df_weights_bs = pd.read_parquet("results_data/base_strategy_weights.parquet")
    df_weights_bs_nc = pd.read_parquet("results_data/base_strategy_weights_nc.parquet")
    df_weights_ns = pd.read_parquet("results_data/new_strategy_weights.parquet")
    df_weights_os = pd.read_parquet("results_data/other_strategy_weights.parquet")

    benchmark_prices = read_data("Constituents TOT_RET_INDEX data").iloc[:, -1]

    # Calcul des rendements quotidiens du portefeuille
    portfolio_daily_returns_bs = compute_daily_portfolio_returns(
        df_total_ret, df_weights_bs
    )
    portfolio_daily_returns_bs_nc = compute_daily_portfolio_returns(
        df_total_ret, df_weights_bs_nc
    )
    portfolio_daily_returns_ns = compute_daily_portfolio_returns(
        df_total_ret, df_weights_ns
    )
    portfolio_daily_returns_os = compute_daily_portfolio_returns(
        df_total_ret, df_weights_os
    )

    # sauvegarde des rendements quotidiens dans results data
    portfolio_daily_returns_bs.to_csv("results_data/base_strategy_daily_returns.csv")
    portfolio_daily_returns_bs_nc.to_csv("results_data/base_strategy_daily_returns_nc.csv")
    portfolio_daily_returns_ns.to_csv("results_data/new_strategy_daily_returns.csv")
    portfolio_daily_returns_os.to_csv("results_data/other_strategy_daily_returns.csv")


    daily_drifted_weights_bs = calculate_daily_drifted_weights(df_weights_bs, df_total_ret)
    daily_drifted_weights_bs_nc = calculate_daily_drifted_weights(df_weights_bs_nc, df_total_ret)
    daily_drifted_weights_ns = calculate_daily_drifted_weights(df_weights_ns, df_total_ret)
    daily_drifted_weights_os = calculate_daily_drifted_weights(df_weights_os, df_total_ret)

    # sauvegarde des daily drifted weights dans results data
    daily_drifted_weights_bs.to_parquet("results_data/base_strategy_ddw.parquet")
    daily_drifted_weights_bs_nc.to_parquet("results_data/base_strategy_ddw_nc.parquet")
    daily_drifted_weights_ns.to_parquet("results_data/new_strategy_ddw.parquet")
    daily_drifted_weights_os.to_parquet("results_data/other_strategy_ddw.parquet")

    benchmark_daily_returns = compute_benchmark_returns(
        benchmark_prices, weights=df_weights_bs
    )

    portfolio_daily_returns_bs_nc.columns = ["BSNC"]



    # Calcul des métriques de performance en utilisant le package quantstats
    qs.extend_pandas()
    # output sous la forme d'un fichier html à ouvrir sur un web browser
    print("Début de la génération du rapport de backtesting...")

    backtesting_metrics_bs = qs.reports.html(
        portfolio_daily_returns_bs,
        benchmark_daily_returns,
        rf=0.01,
        mode="full",
        title="Backtesting Base Strategy",
        output=True,
        download_filename="base_strategy_metrics.html",
        match_dates=True,
    )

    backtesting_metrics_bs_nc = qs.reports.html(
        portfolio_daily_returns_bs_nc,
        benchmark_daily_returns,
        rf=0.01,
        mode="full",
        title="Backtesting Base Strategy with No Constraints",
        output=True,
        download_filename="base_strategy_nc_metrics.html",
        match_dates=True,
    )

    backtesting_metrics_ns = qs.reports.html(
        portfolio_daily_returns_ns,
        benchmark_daily_returns,
        rf=0.01,
        mode="full",
        title="Backtesting New Strategy",
        output=True,
        download_filename="new_strategy_metrics.html",
        match_dates=True,
    )

    backtesting_metrics_os = qs.reports.html(
        portfolio_daily_returns_os,
        benchmark_daily_returns,
        rf=0.01,
        mode="full",
        title="Backtesting Other Strategy",
        output=True,
        download_filename="other_strategy_metrics.html",
        match_dates=True,
    )


    print("Rapport de backtesting généré avec succès!")

    move_file_to_directory("base_strategy_metrics.html", "results_data")
    move_file_to_directory("base_strategy_nc_metrics.html", "results_data")
    move_file_to_directory("new_strategy_metrics.html", "results_data")
    move_file_to_directory("other_strategy_metrics.html", "results_data")
    move_file_to_directory("base_strategy_vs_nc_metrics.html", "results_data")

