import numpy as np
import pandas as pd
from filter import get_rebalance_dates
from base_strategy import (
    calculate_returns,
    calculate_volatility,
    apply_sector_constraints,
    calculate_sector_weights,
)
from base_strategy import (
    check_sector_constraints,
    check_weight_constraints,
    check_weights_sum_to_one,
)

"""Cette stratégie d'investissement est basée sur l'inverse de la volatilité et 
la skewness positive des actifs du portefeuille. Elle vise à attribuer des poids 
aux actifs en tenant compte de ces deux mesures. Voici un aperçu du fonctionnement de la stratégie:

1. Calculer les rendements quotidiens des actifs à partir des prix.

2. Pour chaque date de rééquilibrage :
a. Sélectionner les rendements de l'année écoulée jusqu'à la date de rééquilibrage.
b. Calculer la volatilité annuelle de chaque actif.
c. Calculer la skewness positive de chaque actif.
d. Calculer l'inverse de la volatilité et normaliser les valeurs.
e. Normaliser les valeurs de skewness positive.
f. Créer une combinaison pondérée de l'inverse de la volatilité et de la skewness positive en utilisant les poids vol_weight et skew_weight.

3. Appliquer les contraintes de poids individuelles (min_weight, max_weight) et sectorielles (sector_max_weight) aux poids combinés.

4. Redistribuer les poids pour s'assurer que leur somme est égale à 1.

5. Stocker les poids du portefeuille rééquilibré pour chaque date de rééquilibrage dans un DataFrame.

La stratégie alloue des poids plus importants aux actifs ayant une faible volatilité 
et une skewness positive importante. L'objectif est de diversifier le portefeuille en 
tenant compte de ces deux caractéristiques, afin de minimiser les risques et de 
potentiellement améliorer les rendements.

Le paramètre vol_weight détermine l'importance de l'inverse de la volatilité dans la combinaison pondérée, 
tandis que le paramètre skew_weight détermine l'importance de la skewness positive. 
En ajustant ces paramètres, vous pouvez modifier la manière dont la stratégie 
attribue des poids aux actifs en fonction de leur volatilité et de leur skewness."""


def inverse_volatility_and_skewness_strategy(
    df_prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    gics_sectors: pd.DataFrame,
    max_weight: float = 0.05,
    min_weight: float = 0.0005,
    sector_max_weight: float = 0.4,
    vol_weight: float = 0.7,  # Poids de l'inverse de la volatilité
    skew_weight: float = 0.3,  # Poids de la skewness positive
) -> pd.DataFrame:
    """
    Applique une stratégie de poids de portefeuille basée sur l'inverse de
    la volatilité et la skewness positive.

    Parameters
    ----------
    df_prices : pandas DataFrame
        DataFrame contenant les prix quotidiens des actifs.

    rebalance_dates : pandas DatetimeIndex
        Liste de dates de rééquilibrage du portefeuille.

    gics_sectors : pandas DataFrame
        DataFrame contenant les classifications sectorielles GICS des actifs.

    max_weight : float, optional
        Poids maximum autorisé pour un actif individuel, par défaut 0.05.

    min_weight : float, optional
        Poids minimum autorisé pour un actif individuel, par défaut 0.0005.

    sector_max_weight : float, optional
        Poids maximum autorisé pour un secteur, par défaut 0.4.

    vol_weight : float, optional
        Poids de l'inverse de la volatilité dans la combinaison pondérée, par défaut 0.7.

    skew_weight : float, optional
        Poids de la skewness positive dans la combinaison pondérée, par défaut 0.3.

    Returns
    -------
    pandas DataFrame
        DataFrame contenant les poids du portefeuille à chaque date de rééquilibrage.

    """

    # Calculer les rendements quotidiens
    df_returns = calculate_returns(df_prices)

    # DataFrame contenant les poids du portefeuille à chaque date de rééquilibrage
    weights = pd.DataFrame(index=rebalance_dates, columns=df_prices.columns)

    for idx, rebalance_date in enumerate(rebalance_dates):
        if idx == 0:
            one_year_start_date = df_prices.index[0]
        else:
            one_year_start_date = rebalance_dates[idx - 1] + pd.DateOffset(days=1)

        # Sélectionner les rendements de l'année écoulée jusqu'à la date de rééquilibrage
        one_year_returns = df_returns.loc[one_year_start_date:rebalance_date]

        # Calculer la volatilité sur l'année écoulée jusqu'à la date de rééquilibrage
        one_year_volatility = calculate_volatility(
            one_year_returns, window=len(one_year_returns)
        ).iloc[-1]

        # Calculer la skewness négative sur l'année écoulée jusqu'à la date de rééquilibrage
        one_year_skewness = one_year_returns.skew()

        # Calculer l'inverse de la volatilité et remplacer les valeurs infinies par NaN
        inverse_volatility = 1 / one_year_volatility
        inverse_volatility.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Normaliser l'inverse de la volatilité et la skewness négative
        normalized_inverse_volatility = inverse_volatility / inverse_volatility.sum()
        positive_skewness = one_year_skewness
        normalized_positive_skewness = positive_skewness / positive_skewness.sum()

        # Combinaison pondérée de l'inverse de la volatilité et de la skewness négative
        combined_weights = (
            vol_weight * normalized_inverse_volatility
            + skew_weight * normalized_positive_skewness
        )
        weights.loc[rebalance_date] = combined_weights

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


if __name__ == "__main__":
    # Paramètres
    min_weight = 0.0005
    max_weight = 0.05
    sector_max_weight = 0.4
    # Chargement des données
    df_prices = pd.read_parquet("filtered_data/total_ret_data.parquet")
    rebalance_dates = get_rebalance_dates(df_prices)
    gics_sectors = pd.read_parquet("converted_data/Constituents GICS sectors.parquet")

    # Calculer les poids du portefeuille
    weights = inverse_volatility_and_skewness_strategy(
        df_prices, rebalance_dates, gics_sectors
    )

    sector_weights = calculate_sector_weights(weights=weights, df_sectors=gics_sectors)

    # save weights to parquet in converted_data folder
    weights.to_parquet("results_data/new_strategy_weights.parquet")
    sector_weights.to_parquet("results_data/new_strategy_sector_weights.parquet")

    are_weight_constraints_respected = check_weight_constraints(
        weights=weights,
        min_weight=min_weight,
        max_weight=max_weight,
        verbose=True,
    )

    if are_weight_constraints_respected:
        print("\nToutes les contraintes de poids individuelles sont respectées.")
    else:
        print(
            "\nDes erreurs ont été trouvées dans les contraintes de poids individuelles."
        )

    are_sectors_constraints_respected = check_sector_constraints(
        weights=weights,
        df_sectors=gics_sectors,
        sector_max_weight=sector_max_weight,
        verbose=False,
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
