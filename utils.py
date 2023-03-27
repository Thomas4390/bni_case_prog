import numpy as np
import pandas as pd
from filter import get_rebalance_dates


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
    excess_weights = weights[weights > max_weight] - max_weight
    total_excess_weight = excess_weights.sum()

    weights[weights > max_weight] = max_weight
    remaining_weights = weights[weights >= min_weight]
    remaining_weights += total_excess_weight / len(remaining_weights)
    weights[weights >= min_weight] = remaining_weights

    return weights


def apply_sector_constraints(
    weights: pd.Series,
    gics_sectors: pd.DataFrame,
    min_weight: float,
    max_weight: float,
    sector_max_weight: float,
) -> pd.Series:
    for sector in gics_sectors["GICS Sector"].unique():
        sector_tickers = gics_sectors.loc[
            gics_sectors["GICS Sector"] == sector, "Ticker"
        ]
        sector_weights = weights[sector_tickers]
        sector_allocation = sector_weights.sum()

        if sector_allocation > sector_max_weight:
            redistributed_weights = redistribute_weights(
                sector_weights, min_weight, max_weight
            )
            weights[sector_tickers] = redistributed_weights

    return weights


def inverse_volatility_strategy(
    df_prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    gics_sectors: pd.DataFrame,
    max_weight: float = 0.05,
    min_weight: float = 0.0005,
    sector_max_weight: float = 0.4,
) -> pd.DataFrame:
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

        # Redistribuer les poids pour s'assurer que leur somme est égale à 1
        rebalance_weights /= rebalance_weights.sum()
        weights.loc[rebalance_date] = rebalance_weights

    return weights


def check_weight_constraints(
        weights: pd.DataFrame,
        min_weight: float = 0.0005,
        max_weight: float = 0.05,
        verbose: bool = True) -> bool:
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
            elif verbose:
                print(
                    f"[OK] La contrainte de poids individuelle est respectée pour le titre {ticker} à la date {date}: {weight:.2%}"
                )

    return constraints_respected


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
    rebalance_dates = weights.index
    constraints_respected = True

    for date in rebalance_dates:
        if verbose:
            print(f"\nVérification des contraintes sectorielles pour la date {date}:")

        # On drop les poids qui sont NaN et on calcule l'arrondi à 4 décimales
        weights_on_date = (
            weights.loc[date]
            .dropna()
            .apply(lambda x: round(x, 4) if not np.isnan(x) else x)
        )

        combined = pd.concat(
            [weights_on_date, df_sectors.set_index("Ticker")], axis=1, join="inner"
        )
        sector_weights = combined.groupby("GICS Sector").sum()

        for sector, weight in sector_weights.iterrows():
            weight_value = weight[0]
            if verbose:
                print(f" - {sector}: {weight_value:.2%}")
            if weight_value > sector_max_weight:
                constraints_respected = False
                if verbose:
                    print(
                        f"   [ERREUR] La contrainte de poids sectorielle n'est pas respectée pour le secteur {sector} à la date {date}"
                    )
            elif verbose:
                print(
                    f"   [OK] La contrainte de poids sectorielle est respectée pour le secteur {sector} à la date {date}"
                )

    return constraints_respected


if __name__ == "__main__":
    # Paramètres
    min_weight = 0.0005
    max_weight = 0.05
    sector_max_weight = 0.4
    # Charger les données
    df_filtered = pd.read_parquet("converted_data/filtered_data.parquet")
    df_sectors = pd.read_parquet("converted_data/Constituents GICS sectors.parquet")

    print("Calcul des poids de l'investissement...")

    rebalance_dates = get_rebalance_dates(df_filtered)
    weights = inverse_volatility_strategy(df_prices=df_filtered,
                                          rebalance_dates=rebalance_dates,
                                          min_weight=min_weight,
                                          max_weight=max_weight,
                                          sector_max_weight=sector_max_weight)


    are_weight_constraints_respected = check_weight_constraints(weights=weights,
                                                                min_weight=min_weight,
                                                                max_weight=max_weight,
                                                                verbose=False)
    if are_weight_constraints_respected:
        print("\nToutes les contraintes de poids individuelles sont respectées.")
    else:
        print(
            "\nDes erreurs ont été trouvées dans les contraintes de poids individuelles."
        )

    are_sectors_constraints_respected = check_sector_constraints(weights=weights,
                                                                 df_sectors=df_sectors,
                                                                 verbose=False)

    if are_sectors_constraints_respected:
        print("\nToutes les contraintes sectorielles sont respectées.")
    else:
        print("\nDes erreurs ont été trouvées dans les contraintes sectorielles.")
