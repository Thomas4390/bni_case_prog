from typing import Iterable
import numpy as np
import pandas as pd
from numpy import ndarray
from filter import get_rebalance_dates

"""Cette stratégie d'investissement est basée sur l'inverse de la volatilité 
des actifs du portefeuille. Elle vise à attribuer des poids aux actifs 
en tenant compte de leur volatilité. Voici un aperçu du fonctionnement de la stratégie:

1. Calculer les rendements quotidiens des actifs à partir des prix.

2. Pour chaque date de rééquilibrage :
a. Sélectionner les rendements de l'année écoulée jusqu'à la date de rééquilibrage.
b. Calculer la volatilité annuelle de chaque actif.
c. Calculer l'inverse de la volatilité et remplacer les valeurs infinies par NaN.
d. Normaliser les valeurs de l'inverse de la volatilité (diviser chaque valeur par la somme de toutes les valeurs).
e. Stocker les poids normalisés dans le DataFrame weights.

3. Appliquer les contraintes de poids individuelles (min_weight, max_weight) 
et sectorielles (sector_max_weight) aux poids normalisés.

4. Redistribuer les poids pour s'assurer que leur somme est égale à 1.

5. Stocker les poids du portefeuille rééquilibré pour chaque date de rééquilibrage dans un DataFrame.

La stratégie alloue des poids plus importants aux actifs ayant une faible volatilité. 
L'objectif est de diversifier le portefeuille en tenant compte de la volatilité 
des actifs, afin de minimiser les risques."""

def calculate_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les rendements quotidiens pour chaque action.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame contenant les prix des actions.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les rendements quotidiens pour chaque action.
    """
    df_returns = df_prices.pct_change()
    # Supprimer la première ligne qui contient des NaN
    df_returns = df_returns.iloc[1:]
    return df_returns


def calculate_volatility(df_returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Calcule la volatilité (écart type) des rendements pour chaque action sur une fenêtre donnée.

    Parameters
    ----------
    df_returns : pd.DataFrame
        DataFrame contenant les rendements des actions.
    window : int
        Fenêtre sur laquelle calculer la volatilité (en jours).

    Returns
    -------
    pd.DataFrame
        DataFrame contenant la volatilité des rendements pour chaque action.
    """
    return df_returns.rolling(window=window).std()


def redistribute_weights(
    weights: pd.Series, min_weight: float, max_weight: float
) -> pd.Series:
    """
    Redistribue les poids excédentaires d'une série de poids en conservant
    les poids restants entre `min_weight` et `max_weight`.

    Parameters :
    ------------
    weights : pandas.Series
        La série de poids à redistribuer.
    min_weight : float
        Le poids minimal autorisé pour chaque élément de la série de poids.
    max_weight : float
        Le poids maximal autorisé pour chaque élément de la série de poids.

    Returns :
    ---------
    redistributed_weights : pandas.Series
        La série de poids après redistribution.

    """
    excess_weights = weights[weights > max_weight] - max_weight
    total_excess_weight = excess_weights.sum()

    # Limiter les poids excédentaires à max_weight
    weights[weights > max_weight] = max_weight

    # Calculer les indices des poids restants qui sont supérieurs ou égaux à min_weight
    remaining_indices = weights >= min_weight

    # Calculer la somme des poids restants pour éviter de redistribuer les poids excédentaires aux poids inférieurs à min_weight
    remaining_weights_sum = weights[remaining_indices].sum()

    # Redistribuer le poids excédentaire proportionnellement aux poids restants
    weights[remaining_indices] += (total_excess_weight / remaining_weights_sum) * (
        weights[remaining_indices] / remaining_weights_sum
    )

    return weights


def apply_sector_constraints(
    weights: pd.Series,
    gics_sectors: pd.DataFrame,
    min_weight: float,
    max_weight: float,
    sector_max_weight: float,
) -> pd.Series:
    """
    Applique les contraintes de poids sectorielles et individuelles à une série de poids.

    Parameters
    ----------
    weights : pandas Series
        La série de poids à laquelle appliquer les contraintes.

    gics_sectors : pandas DataFrame
        DataFrame contenant les classifications sectorielles GICS des actifs.

    min_weight : float
        Le poids minimum autorisé pour chaque action.

    max_weight : float
        Le poids maximum autorisé pour chaque action.

    sector_max_weight : float
        Le poids maximum autorisé pour chaque secteur.

    Returns
    -------
    pandas Series
        La série de poids ajustée avec les contraintes de poids sectorielles et individuelles appliquées.

    """
    # Appliquer les contraintes de poids individuelles aux actions de tous les secteurs
    weights = weights.clip(lower=min_weight, upper=max_weight)

    sector_excess_weights = {}

    # Boucle pour vérifier et ajuster les poids en fonction des contraintes
    while True:
        sector_violations = False

        for sector in gics_sectors["GICS Sector"].unique():
            sector_tickers = gics_sectors.loc[
                gics_sectors["GICS Sector"] == sector, "Ticker"
            ]

            # Filtrer les tickers de secteur pour ne conserver que ceux présents dans l'index de la série 'weights'
            sector_tickers = sector_tickers[sector_tickers.isin(weights.index)]

            sector_weights = weights[sector_tickers]
            sector_allocation = sector_weights.sum()

            if sector_allocation > sector_max_weight:
                sector_violations = True
                excess_weight = sector_allocation - sector_max_weight

                # Redistribuer l'excédent de poids proportionnellement aux actions du secteur
                weights[sector_tickers] *= sector_max_weight / sector_allocation

                # Appliquer les contraintes de poids individuelles aux actions du secteur
                weights[sector_tickers] = weights[sector_tickers].clip(
                    lower=min_weight, upper=max_weight
                )

                sector_excess_weights[sector] = excess_weight

        if not sector_violations:
            break

    # Redistribuer l'excédent de poids des secteurs proportionnellement aux actions des autres secteurs
    for sector, excess_weight in sector_excess_weights.items():
        non_sector_tickers = weights.index.difference(
            gics_sectors.loc[gics_sectors["GICS Sector"] == sector, "Ticker"]
        )
        weights[non_sector_tickers] += (
            excess_weight / weights.loc[non_sector_tickers].sum()
        ) * weights.loc[non_sector_tickers]

    # Appliquer les contraintes de poids individuelles aux actions de tous les secteurs après redistribution
    weights = weights.clip(lower=min_weight, upper=max_weight)

    # Redistribuer les poids pour s'assurer que leur somme est égale à 1
    weights /= weights.sum()

    return weights


def inverse_volatility_strategy(
    df_prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    gics_sectors: pd.DataFrame,
    max_weight: float = 0.05,
    min_weight: float = 0.0005,
    sector_max_weight: float = 0.4,
) -> pd.DataFrame:
    """
    Implémente la stratégie de volatilité inverse avec des contraintes
    de poids individuelles et sectorielles.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame contenant les prix des actions.
    rebalance_dates : pd.DatetimeIndex
        Index contenant les dates de rééquilibrage.
    gics_sectors : pd.DataFrame
        DataFrame contenant les informations sur les secteurs GICS des actions.
    max_weight : float, optional
        Poids maximum autorisé pour chaque action, par défaut 0.05 (5%).
    min_weight : float, optional
        Poids minimum autorisé pour chaque action, par défaut 0.0005 (0.05%).
    sector_max_weight : float, optional
        Poids maximum autorisé pour chaque secteur GICS, par défaut 0.4 (40%).

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les poids associés à chaque action pour chaque date de rééquilibrage.
    """
    df_returns = calculate_returns(df_prices)
    weights = pd.DataFrame(index=rebalance_dates, columns=df_prices.columns)

    for idx, rebalance_date in enumerate(rebalance_dates):
        if idx == 0:
            one_year_start_date = df_prices.index[0]
        else:
            one_year_start_date = rebalance_dates[idx - 1] + pd.DateOffset(days=1)

        one_year_returns = df_returns.loc[one_year_start_date:rebalance_date]
        one_year_volatility = calculate_volatility(
            one_year_returns, window=len(one_year_returns)
        ).iloc[-1]

        inverse_volatility = 1 / one_year_volatility
        inverse_volatility.replace([np.inf, -np.inf], np.nan, inplace=True)
        sum_inverse_volatility = np.nansum(inverse_volatility)

        raw_weights = inverse_volatility / sum_inverse_volatility
        weights.loc[rebalance_date] = raw_weights

        # Appliquer les contraintes de poids individuelles et sectorielles
        rebalance_weights = weights.loc[rebalance_date]
        rebalance_weights = rebalance_weights.clip(lower=min_weight, upper=max_weight)
        rebalance_weights = apply_sector_constraints(
            rebalance_weights, gics_sectors, min_weight, max_weight, sector_max_weight
        )

        # Appliquer à nouveau les contraintes de poids individuelles
        rebalance_weights = rebalance_weights.clip(lower=min_weight, upper=max_weight)

        # Redistribuer les poids pour s'assurer que leur somme est égale à 1
        rebalance_weights /= rebalance_weights.sum()
        weights.loc[rebalance_date] = rebalance_weights

    # Remplace tous les None par des NaN
    weights.replace([None], np.nan, inplace=True)

    return weights


def check_weight_constraints(
    weights: pd.DataFrame,
    min_weight: float = 0.0005,
    max_weight: float = 0.05,
    verbose: bool = True,
) -> bool:
    """
    Vérifie si les poids de chaque titre respectent les contraintes de poids individuelles.

    Parameters
    ----------
    weights : pd.DataFrame
        DataFrame contenant les poids de chaque titre.
    min_weight : float
        Pondération minimale autorisée pour chaque titre.
    max_weight : float
        Pondération maximale autorisée pour chaque titre.
    verbose : bool, optional
        Afficher ou non le résultat de la vérification, par défaut True.

    Returns
    -------
    bool
        True si toutes les contraintes de poids individuelles sont respectées, False sinon.
    """
    rebalance_dates = weights.index
    constraints_respected = True

    for date in rebalance_dates:
        weights_on_date = (
            weights.loc[date]
            .dropna()
            .apply(lambda x: round(x, 4) if not np.isnan(x) else x)
        )

        for ticker, weight in weights_on_date.items():
            if weight < min_weight or weight > max_weight:
                constraints_respected = False
                if verbose:
                    print(
                        f"[ERREUR] La contrainte de poids individuelle n'est pas respectée pour le titre {ticker} à la date {date}: {weight:.2%}"
                    )

    return constraints_respected


def calculate_sector_weights(
    weights: pd.DataFrame, df_sectors: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcule les pondérations sectorielles après chaque date de rééquilibrage.

    Parameters
    ----------
    weights : pd.DataFrame
        DataFrame contenant les poids de chaque titre.
    df_sectors : pd.DataFrame
        DataFrame contenant la classification GICS de chaque titre.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les pondérations sectorielles pour chaque secteur
        GICS et chaque date de rééquilibrage.
    """
    rebalance_dates = weights.index
    sector_weights_dict = {}

    for date in rebalance_dates:
        weights_on_date = (
            weights.loc[date]
            .dropna()
            .apply(lambda x: round(x, 4) if not np.isnan(x) else x)
        )

        combined = pd.concat(
            [weights_on_date, df_sectors.set_index("Ticker")], axis=1, join="inner"
        )
        sector_weights = combined.groupby("GICS Sector").sum()
        sector_weights_dict[date] = sector_weights.T.squeeze()

    sector_weights_df = pd.DataFrame.from_dict(sector_weights_dict).T
    return sector_weights_df


def check_sector_constraints(
    weights: pd.DataFrame,
    df_sectors: pd.DataFrame,
    sector_max_weight: float = 0.4,
    verbose: bool = True,
) -> bool:
    """
    Vérifie si les poids de chaque secteur respectent les contraintes de poids sectorielles.

    Parameters
    ----------
    weights : pd.DataFrame
        DataFrame contenant les poids de chaque titre.
    df_sectors : pd.DataFrame
        DataFrame contenant la classification GICS de chaque titre.
    sector_max_weight : float, optional
        Pondération maximale autorisée pour chaque secteur GICS, par défaut 0.4.
    verbose : bool, optional
        Si True, affiche les messages d'erreur ou de réussite pour chaque contrainte, par défaut True.

    Returns
    -------
    bool
        True si toutes les contraintes de poids sectorielles sont respectées, False sinon.
    """
    sector_weights_df = calculate_sector_weights(weights, df_sectors)
    rebalance_dates = sector_weights_df.index
    constraints_respected = True

    for date in rebalance_dates:
        sector_weights_on_date = sector_weights_df.loc[date]

        for sector, weight in sector_weights_on_date.items():
            if weight > sector_max_weight:
                constraints_respected = False
                if verbose:
                    print(
                        f"   [ERREUR] La contrainte de poids sectorielle "
                        f"n'est pas respectée pour le secteur {sector} "
                        f"({100 * weight:.2f}%) à la date {date}"
                    )

    return constraints_respected


def check_weights_sum_to_one(weights: pd.DataFrame, tolerance: float = 1e-6) -> bool:
    """
    Vérifie que les poids somment à 1 à chaque date de rééquilibrage.

    Parameters
    ----------
    weights : pd.DataFrame
        DataFrame contenant les poids associés à chaque action pour chaque date de rééquilibrage.
    tolerance : float, optional
        Tolérance pour les erreurs numériques, par défaut 1e-6.

    Returns
    -------
    bool
        Retourne True si les poids somment à 1 pour chaque date de rééquilibrage, sinon False.
    """

    def is_close_to_one(x: pd.Series) -> ndarray | Iterable | int | float:
        return np.isclose(x.sum(), 1, rtol=tolerance, atol=tolerance)

    result = weights.apply(is_close_to_one, axis=1)
    return result.all()


if __name__ == "__main__":
    # Paramètres
    min_weight = 0.0005
    max_weight = 0.05
    sector_max_weight = 0.4
    # Charger les données
    df_total_ret_filtered = pd.read_parquet("filtered_data/total_ret_data.parquet")
    df_sectors = pd.read_parquet("converted_data/Constituents GICS sectors.parquet")

    rebalance_dates = get_rebalance_dates(df_total_ret_filtered)

    print("Calcul des poids de l'investissement...")

    weights = inverse_volatility_strategy(
        df_prices=df_total_ret_filtered,
        rebalance_dates=rebalance_dates,
        gics_sectors=df_sectors,
        min_weight=min_weight,
        max_weight=max_weight,
        sector_max_weight=sector_max_weight,
    )

    sector_weights = calculate_sector_weights(weights=weights, df_sectors=df_sectors)

    # save weights to parquet in converted_data folder
    weights.to_parquet("results_data/base_strategy_weights.parquet")
    sector_weights.to_parquet("results_data/base_strategy_sector_weights.parquet")

    are_weight_constraints_respected = check_weight_constraints(
        weights=weights, min_weight=min_weight, max_weight=max_weight, verbose=True
    )
    if are_weight_constraints_respected:
        print("\nToutes les contraintes de poids individuelles sont respectées.")
    else:
        print(
            "\nDes erreurs ont été trouvées dans les contraintes de poids individuelles."
        )

    are_sectors_constraints_respected = check_sector_constraints(
        weights=weights,
        df_sectors=df_sectors,
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
