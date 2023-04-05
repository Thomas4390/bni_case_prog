import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_sector_weights(weights: pd.DataFrame, strategy_name: str) -> None:
    """
    Affiche les poids des secteurs en fonction du temps.

    Parameters
    ----------
    weights : pd.DataFrame
        DataFrame contenant les poids des secteurs en fonction du temps.
    strategy_name : str
        Nom de la stratégie à ajouter au nom de la figure.

    Returns
    -------
    None
    """
    # Afficher les poids des secteurs
    fig, ax = plt.subplots(figsize=(15, 8))
    weights.plot.area(ax=ax)
    ax.set_title(f"Évolution du poids des secteurs GICS au cours du temps en aire | {strategy_name}")
    ax.set_ylabel("Poids")
    ax.set_xlabel("Date")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    # Ajuster la taille de la zone de tracé pour laisser de la place à la légende
    plt.subplots_adjust(right=0.8)

    # Sauvegarder la figure dans le sous-répertoire "figures"
    if not os.path.exists("figures"):
        os.mkdir("figures")
    fig.savefig(f"figures/sector_weights_{strategy_name}.png")

    plt.show()

    return None


if __name__ == "__main__":
    # Lecture des données
    df_weights_sectors_bs = pd.read_parquet(
        "results_data/base_strategy_sector_weights.parquet"
    )
    df_weights_sectors_ns = pd.read_parquet(
        "results_data/new_strategy_sector_weights.parquet"
    )
    df_weights_sectors_os = pd.read_parquet(
        "results_data/other_strategy_sector_weights.parquet"
    )
    df_weights_sectors_bs_nc = pd.read_parquet(
        "results_data/base_strategy_sector_weights_nc.parquet"
    )

    # Afficher les poids des secteurs
    plot_sector_weights(df_weights_sectors_bs, strategy_name="base_strategy")
    plot_sector_weights(df_weights_sectors_bs_nc, strategy_name="base_strategy_nc")
    plot_sector_weights(df_weights_sectors_ns, strategy_name="new_strategy")
    plot_sector_weights(df_weights_sectors_os, strategy_name="other_strategy")
