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



def apply_nan_mask(df_source: pd.DataFrame,
                   df_target: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le masque des valeurs NaN d'un DataFrame source sur un DataFrame cible.
    Il est possible que le DataFrame cible ait plus de NaN values que le DataFrame source.
    Cela peut s'expliquer par des valeurs NaN deja présentes dans le DataFrame cible.

    Parameters
    ----------
    df_source : pd.DataFrame
        DataFrame source dont le masque NaN sera utilisé.
    df_target : pd.DataFrame
        DataFrame cible sur lequel le masque NaN sera appliqué.

    Returns
    -------
    pd.DataFrame
        DataFrame cible avec le masque NaN appliqué.
    """
    # Trouver l'intersection des dates et des colonnes
    common_dates = df_source.index.intersection(df_target.index)
    common_columns = df_source.columns.intersection(df_target.columns)

    # Créer un masque NaN à partir du DataFrame source
    nan_mask = df_source.loc[common_dates, common_columns].isna()

    # Appliquer le masque NaN sur le DataFrame cible
    df_target_masked = df_target.loc[common_dates, common_columns].where(
        ~nan_mask, np.nan)

    return df_target_masked


if "__main__" == __name__:
    df_volume = read_data("Constituents PX_VOLUME data")
    df_px = read_data("Constituents PX_LAST data")
    df_total_ret = read_data("Constituents TOT_RET_INDEX data")
    # On ne prend pas en compte la dernière colonne qui
    # correspond au volume total de l'indice
    df_volume = df_volume.iloc[:, :-1]
    rebalance_dates = get_rebalance_dates(df_volume)

    df_volume_filtered = filter_top_quantile(
        df_volume=df_volume, rebalance_dates=rebalance_dates, quantile=0.2)

    df_px_filtered = apply_nan_mask(df_source=df_volume_filtered, df_target=df_px)

    df_total_ret_filtered = apply_nan_mask(
        df_source=df_volume_filtered, df_target=df_total_ret)

    # save all dataframe to parquet in the filtered_data folder
    df_volume_filtered.to_parquet("filtered_data/volume_data.parquet")
    df_px_filtered.to_parquet("filtered_data/PX_LAST_data.parquet")
    df_total_ret_filtered.to_parquet("filtered_data/total_ret_data.parquet")
