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


def inverse_volatility_strategy(df_prices: pd.DataFrame, rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Implémente la stratégie de volatilité inverse.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame contenant les prix des actions.
    rebalance_dates : pd.DatetimeIndex
        Index contenant les dates de rééquilibrage.

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
        one_year_volatility = calculate_volatility(one_year_returns, window=len(one_year_returns)).iloc[-1]

        inverse_volatility = 1 / one_year_volatility
        inverse_volatility.replace([np.inf, -np.inf], np.nan, inplace=True)  # Remplacer les infinis par NaN
        sum_inverse_volatility = np.nansum(inverse_volatility)

        # Calculer les poids et les stocker dans le DataFrame weights
        weights.loc[rebalance_date] = inverse_volatility / sum_inverse_volatility

    return weights


if __name__ == "__main__":
    # Charger les données
    df_filtered = pd.read_parquet("converted_data/filtered_data.parquet")
    rebalance_dates = get_rebalance_dates(df_filtered)

    # Calculer les poids
    weights = inverse_volatility_strategy(df_filtered, rebalance_dates)

    # Afficher les poids
    print(weights)