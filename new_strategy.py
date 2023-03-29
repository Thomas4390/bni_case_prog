import numpy as np
import pandas as pd
from filter import get_rebalance_dates
from base_strategy import (
    inverse_volatility_strategy,
    calculate_returns,
    calculate_volatility,
    apply_sector_constraints,
    redistribute_weights,
    calculate_sector_weights,
)
from base_strategy import (
    check_sector_constraints,
    check_weight_constraints,
    check_weights_sum_to_one,
)


def inverse_volatility_and_skewness_strategy(
    df_prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    gics_sectors: pd.DataFrame,
    max_weight: float = 0.05,
    min_weight: float = 0.0005,
    sector_max_weight: float = 0.4,
    vol_weight: float = 0.7,  # Poids de l'inverse de la volatilité
    skew_weight: float = 0.3,  # Poids de la skewness négative
) -> pd.DataFrame:
    # Calculer les rendements quotidiens
    df_returns = calculate_returns(df_prices)
    # DataFrame contenant les poids du portefeuille à chaque date de rééquilibrage
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
        one_year_skewness = one_year_returns.skew()

        inverse_volatility = 1 / one_year_volatility
        inverse_volatility.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Normaliser l'inverse de la volatilité et la skewness négative
        normalized_inverse_volatility = inverse_volatility / inverse_volatility.sum()
        negative_skewness = -one_year_skewness
        normalized_negative_skewness = negative_skewness / negative_skewness.sum()

        # Combinaison pondérée de l'inverse de la volatilité et de la skewness négative
        combined_weights = (
            vol_weight * normalized_inverse_volatility
            + skew_weight * normalized_negative_skewness
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

    sector_weights = calculate_sector_weights(weights=weights,
                                              df_sectors=gics_sectors)
    print(sector_weights)
    print(sector_weights.max(axis=0))
    # save weights to parquet in converted_data folder
    weights.to_parquet("results_data/new_strategy_weights.parquet")
    sector_weights.to_parquet(
        "results_data/new_strategy_sector_weights.parquet")
    print(weights.iloc[:10, :10])

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
        weights=weights, df_sectors=gics_sectors, verbose=False
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
