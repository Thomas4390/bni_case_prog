import pandas as pd
from datetime import timedelta
from preprocessing import read_data

import numpy as np

def get_rebalance_dates(df_volume: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Récupère toutes les dates de rebalancement annuelles (dernier jour avant le changement d'année) dans le DataFrame.

    Parameters
    ----------
    df_volume : pd.DataFrame
        DataFrame contenant les volumes quotidiens des actifs.

    Returns
    -------
    pd.DatetimeIndex
        Index contenant les dates de rebalancement annuelles (dernier jour avant le changement d'année).

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


def filter_top_quantile(df_volume: pd.DataFrame, rebalance_dates: pd.DatetimeIndex, quantile: float) -> pd.DataFrame:
    """
    Filtrer les actions faisant partie du quantile supérieur sur
    l'année précédant chaque date de rebalancement.

    Parameters
    ----------
    df_volume : pd.DataFrame
        DataFrame contenant les données à filtrer.
    rebalance_dates : pd.DatetimeIndex
        Index contenant les dates de rebalancement.
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
            one_year_start_date = df_volume.index[
                0]  # Utiliser la première date du DataFrame
        else:
            one_year_start_date = rebalance_dates[idx - 1] + pd.DateOffset(
                days=1)

        one_year_data = df_volume.loc[one_year_start_date:rebalance_dates[idx]]

        # Calculer le seuil unique pour chaque colonne
        threshold = np.nanquantile(one_year_data, quantile)
        print(threshold)

        # Conserver les actions qui satisfont le critère ou remplacer par des NaN pour chaque colonne
        for col in one_year_data:
            if one_year_data[col].mean() < threshold:
                df_filtered.loc[one_year_start_date:rebalance_dates[idx],col] = np.nan

    # Supprimer les lignes dont la date est supérieure à la dernière date de rééquilibrage
    last_rebalance_date = rebalance_dates[-1]
    df_filtered = df_filtered.loc[:last_rebalance_date]

    return df_filtered


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
    filtered_volume = filter_top_quantile(df_volume=df_volume,
                                             rebalance_dates=rebalance_dates,
                                             quantile=quantile)
    return filtered_volume




if "__main__" == __name__:
    df_volume = read_data("Constituents PX_VOLUME data")
    # drop la dernière colonne
    df_volume = df_volume.iloc[:, :-1]

    print(df_volume.shape)
    print(get_rebalance_dates(df_volume))
    filtered_volume = filter_universe(df_volume, quantile=0.2)
    print(filtered_volume.isna().sum().sum())
    # save dataframe to excel
    filtered_volume.to_excel("filtered_volume.xlsx")

