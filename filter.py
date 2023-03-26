import pandas as pd
from datetime import timedelta
from preprocessing import read_data

import numpy as np


def get_rebalance_dates(df_volume: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Récupère toutes les dates de rebalancement annuelles
    (dernier jour avant le changement d'année) dans le DataFrame.

    Parameters
    ----------
    df_volume : pd.DataFrame
        DataFrame contenant les volumes quotidiens des actifs.

    Returns
    -------
    pd.DatetimeIndex
        Index contenant les dates de rebalancement annuelles
        (dernier jour avant le changement d'année).

    """
    # Convertir l'index en DatetimeIndex s'il ne l'est pas déjà
    if not isinstance(df_volume.index, pd.DatetimeIndex):
        df_volume.index = pd.to_datetime(df_volume.index)

    # Identifier les changements d'année dans l'index
    years = df_volume.index.year
    year_changes = np.where(years[:-1] != years[1:])[0]

    # Sélectionner les dates de rebalancement comme étant les dernières dates avant les changements d'année
    rebalance_dates = df_volume.index[year_changes]

    return pd.DatetimeIndex(rebalance_dates)


def filter_top_quantile(
    df_volume: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, quantile: float
) -> pd.DataFrame:
    """
    Filtrer les actions en ne conservant que celles dont la moyenne annuelle
    de volume est dans le quantile supérieur sur l'année précédant chaque date
    de rééquilibrage.

    Parameters
    ----------
    df_volume : pd.DataFrame
        DataFrame contenant les données de volume à filtrer.
    rebalance_dates : pd.DatetimeIndex
        Index contenant les dates de rééquilibrage.
    quantile : float
        Quantile à utiliser pour le filtrage.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les données filtrées.
    """
    df_filtered = df_volume.copy()
    for idx in range(len(rebalance_dates)):
        if idx == 0:
            one_year_start_date = df_volume.index[0]
        else:
            one_year_start_date = rebalance_dates[idx - 1] + pd.DateOffset(days=1)

        one_year_data = df_volume.loc[one_year_start_date : rebalance_dates[idx]]

        # Calculer le seuil unique pour chaque colonne
        threshold = np.nanquantile(one_year_data, quantile)

        # Conserver les actions qui satisfont le critère ou remplacer par des NaN pour chaque colonne
        for col in one_year_data:
            if one_year_data[col].mean() < threshold:
                df_filtered.loc[
                    one_year_start_date : rebalance_dates[idx], col
                ] = np.nan

    # Supprimer les lignes dont la date est supérieure à la dernière date de rééquilibrage
    last_rebalance_date = rebalance_dates[-1]
    df_filtered = df_filtered.loc[:last_rebalance_date]

    return df_filtered


if "__main__" == __name__:
    df_volume = read_data("Constituents PX_VOLUME data")
    # On ne prend pas en compte la dernière colonne qui
    # correspond au volume total de l'indice
    df_volume = df_volume.iloc[:, :-1]
    rebalance_dates = get_rebalance_dates(df_volume)
    df_filtred = filter_top_quantile(
        df_volume=df_volume, rebalance_dates=rebalance_dates, quantile=0.2
    )
    # save dataframe to parquet in converted_data folder
    df_filtred.to_parquet("converted_data/filtered_data.parquet")
