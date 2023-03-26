import pandas as pd
from datetime import timedelta
from preprocessing import read_data

def get_rebalance_dates(df_volume: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Récupère toutes les dates de rebalancement annuelles (31 décembre) dans le DataFrame.

    Parameters
    ----------
    df_volume : pd.DataFrame
        DataFrame contenant les volumes quotidiens des actifs.

    Returns
    -------
    pd.DatetimeIndex
        Index contenant les dates de rebalancement annuelles (31 décembre).

    """
    # Convertir l'index en DatetimeIndex s'il ne l'est pas déjà
    if not isinstance(df_volume.index, pd.DatetimeIndex):
        df_volume.index = pd.to_datetime(df_volume.index)
    return df_volume.index[df_volume.index.strftime('%m-%d') == '12-31']

def filter_positive_volume_year(df_volume: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Conserve les actions ayant un volume supérieur à 0 sur l'année précédant chaque rebalancement.

    Parameters
    ----------
    df_volume : pd.DataFrame
        DataFrame contenant les volumes quotidiens des actifs.
    rebalance_dates : pd.DatetimeIndex
        Index contenant les dates de rebalancement annuelles (31 décembre).

    Returns
    -------
    pd.DataFrame
        DataFrame filtré contenant uniquement les actions avec des volumes positifs.

    """
    valid_assets = pd.Series(True, index=df_volume.columns)
    for rebalance_date in rebalance_dates:
        one_year_ago = rebalance_date - timedelta(days=252)
        available_history = df_volume.loc[one_year_ago:rebalance_date]
        valid_assets &= available_history.gt(0).all()
    return df_volume.loc[:, valid_assets]

def filter_six_months_history(df_volume: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Exclut les actions sans historique de volume 6 mois avant chaque date de rebalancement.

    Parameters
    ----------
    df_volume : pd.DataFrame
        DataFrame contenant les volumes quotidiens des actifs.
    rebalance_dates : pd.DatetimeIndex
        Index contenant les dates de rebalancement annuelles (31 décembre).

    Returns
    -------
    pd.DataFrame
        DataFrame filtré contenant uniquement les actions avec un historique de volume suffisant.

    """
    valid_assets = pd.Series(True, index=df_volume.columns)
    for rebalance_date in rebalance_dates:
        six_months_ago = rebalance_date - timedelta(days=6*20)
        available_history = df_volume.loc[six_months_ago:rebalance_date]
        valid_assets &= available_history.notna().all()
    return df_volume.loc[:, valid_assets]

def filter_out_lowest_quantile(df_volume: pd.DataFrame,
                               rebalance_dates: pd.DatetimeIndex,
                               quantile: float = 0.2) -> pd.DataFrame:
    """
    Exclut le quantile inférieur de l'univers d'investissement en fonction des volumes les plus récents.

    Parameters
    ----------
    df_volume : pd.DataFrame
        DataFrame contenant les volumes quotidiens des actifs.
    rebalance_dates : pd.DatetimeIndex
        Index contenant les dates de rebalancement annuelles (31 décembre).

    Returns
    -------
    pd.DataFrame
        DataFrame filtré contenant uniquement les actions du quantile supérieur.

    """
    valid_assets = pd.Series(True, index=df_volume.columns)

    for rebalance_date in rebalance_dates:
        latest_volumes = df_volume.loc[rebalance_date]
        quantile_cutoff = latest_volumes.quantile(quantile) # filtre du quantile
        valid_assets &= latest_volumes.gt(quantile_cutoff)
    return df_volume.loc[:, valid_assets]

def filter_universe(df_volume: pd.DataFrame, quantile: float = 0.2) -> pd.DataFrame:
    """
       Filtre un dataframe de volumes de trading pour créer un nouveau dataframe
       avec seulement les actions qui répondent à des critères spécifiques.

       Parameters
       ----------
       df_volume : pandas.DataFrame
           Un dataframe de volumes de trading pour différentes actions.

       Returns
       -------
       pandas.DataFrame
           Un nouveau dataframe avec seulement les actions qui répondent aux critères spécifiés.

       Notes
       -----
       Cette fonction effectue les étapes de filtrage suivantes :

       1. Obtient une liste des dates de rééquilibrage à partir du dataframe de volumes de trading.
       2. Filtre les actions avec un volume de trading négatif dans l'année en cours.
       3. Filtre les actions qui n'ont pas au moins six mois d'historique de trading.
       4. Filtre les actions dans le plus bas quantile de volume de trading pour chaque date de rééquilibrage.

       """
    rebalance_dates = get_rebalance_dates(df_volume)
    filtered_volume = filter_positive_volume_year(df_volume, rebalance_dates)
    filtered_volume = filter_six_months_history(filtered_volume, rebalance_dates)
    filtered_volume = filter_out_lowest_quantile(filtered_volume,
                                                 rebalance_dates,
                                                 quantile=quantile)
    return filtered_volume


if "__main__" == __name__:
    df_volume = read_data("Constituents PX_VOLUME data")
    filtered_volume = filter_universe(df_volume)
    print(df_volume.shape)
    print(filtered_volume.shape)
    print(filtered_volume.head())
