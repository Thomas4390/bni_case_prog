import pandas as pd
import matplotlib.pyplot as plt


def plot_sector_weights(weights: pd.DataFrame) -> None:
    """
    Affiche les poids des secteurs en fonction du temps.

    Parameters
    ----------
    weights : pd.DataFrame
        DataFrame contenant les poids des secteurs en fonction du temps.

    Returns
    -------
    None
    """
    # Afficher les poids des secteurs
    weights.plot.area(figsize=(15, 8))
    plt.title("Poids des secteurs")
    plt.ylabel("Poids")
    plt.xlabel("Date")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.show()

    return None


if __name__ == "__main__":
    # Lecture des données
    df_weights_sectors = pd.read_parquet("results_data/base_strategy_sector_weights.parquet")
    print(df_weights_sectors.iloc[:5, :5])
    plot_sector_weights(df_weights_sectors)