from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Union
import pandas as pd


def xlsx_to_parquet(xlsx_file: str, parquet_file: str) -> None:
    """
    Convertit un fichier Excel en fichier Parquet.

    Parameters
    ----------
    xlsx_file : str
        Chemin du fichier Excel à convertir.
    parquet_file : str
        Chemin du fichier Parquet de sortie.

    Returns
    -------
    None
    """
    df = pd.read_excel(xlsx_file)
    df.to_parquet(parquet_file)

    return None


def convert_xlsx_files_to_parquet(
    file_list: List[str], output_folder: str = "converted_data"
) -> None:
    """
    Convertit une liste de fichiers Excel en fichiers Parquet.

    Parameters
    ----------
    file_list : List[str]
        Liste des fichiers Excel à convertir.
    output_folder : str, optional
        Chemin du dossier de sortie pour les fichiers Parquet, par défaut
        "converted_data".

    Returns
    -------
    None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for xlsx_file in file_list:
        file_name, _ = os.path.splitext(os.path.basename(xlsx_file))
        parquet_file = os.path.join(output_folder, f"{file_name}.parquet")
        xlsx_to_parquet(xlsx_file, parquet_file)
        print(f"Converted {xlsx_file} to {parquet_file}")

    return None


def get_xlsx_files_in_folder(folder: str) -> List[str]:
    """
    Retourne une liste contenant tous les fichiers Excel (.xlsx) dans le dossier spécifié.

    Parameters
    ----------
    folder : str
        Le chemin du dossier.

    Returns
    -------
    List[str]
        Une liste contenant tous les fichiers Excel (.xlsx) dans le dossier spécifié.
    """
    return [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.endswith(".xlsx")
    ]


def convert_dates(df: pd.DataFrame, date_column: str = "Index Date") -> pd.DataFrame:
    """
    Convertit la colonne des dates dans le format souhaité.

    Parameters
    ----------
    df : pd.DataFrame
        Le DataFrame contenant la colonne des dates.
    date_column : str, optional
        Le nom de la colonne des dates, par défaut "Index Date".

    Returns
    -------
    pd.DataFrame
        Le DataFrame avec la colonne des dates dans le format souhaité.
    """
    # Convertir les dates en chaînes de caractères
    df[date_column] = df[date_column].astype(str)
    # Convertir les chaînes de caractères en objets datetime avec le format souhaité
    df[date_column] = pd.to_datetime(df[date_column], format="%Y%m%d")
    # Convertir les objets datetime en chaînes de caractères avec le format souhaité (AAAA-MM-JJ)
    df[date_column] = df[date_column].dt.strftime("%Y-%m-%d")

    return df


def read_reference_index_holdings() -> pd.DataFrame:
    """
    Charge le fichier de données des pondérations des titres dans l'indice de référence.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant les pondérations des titres dans l'indice de référence.
    """
    df = pd.read_parquet("converted_data/Reference Index Holdings.parquet")
    # Convert the dates to the desired format
    df = convert_dates(df)
    # Convert Index Date to datetime object
    df["Index Date"] = pd.to_datetime(df["Index Date"])
    # Set Index Date as index
    df.set_index("Index Date", inplace=True)
    # Sort the index
    df.sort_index(inplace=True)
    return df


def imput_missing_values_gics_sectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes dans le DataFrame contenant la
    classification GICS des titres.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant la classification GICS des titres.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant la classification GICS des titres avec les valeurs
        manquantes imputées.
    """
    # Impute missing values in GICS Sector
    df["GICS Sector"].fillna("Not Attributed", inplace=True)
    # Save dataframe in parquet
    df.to_parquet("converted_data/Constituents GICS sectors.parquet")
    return df


def read_data(file_name: str) -> pd.DataFrame:
    """
    Lit un fichier de données en format parquet et le transforme en un DataFrame.
    La première colonne du DataFrame doit
    contenir des dates et est renommée en "Index Date" avant d'être définie
    comme index du DataFrame.

    Parameters
    ----------
    file_name : str
        Le nom du fichier de données à lire.

    Returns
    -------
    pd.DataFrame
        Le DataFrame correspondant aux données lues.
    """
    # read data in converted_data folder
    df = pd.read_parquet(f"converted_data/{file_name}.parquet")
    # convert the first column to datetime object
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    # set the first column as index and rename it Index Date
    df.rename(columns={df.columns[0]: "Index Date"}, inplace=True)
    df.set_index("Index Date", inplace=True)
    # sort the index
    df.sort_index(inplace=True)

    return df


import os


def move_file_to_directory(file_path: str, dest_directory: str) -> None:
    """
    Move a file to a specified destination directory.

    Parameters
    ----------
    file_path : str
        The path to the file to be moved.
    dest_directory : str
        The path to the directory where the file should be moved.

    Raises
    ------
    ValueError
        If either `file_path` or `dest_directory` is not a valid directory path.
    OSError
        If an error occurs while moving the file.
    """
    # Check if the destination directory exists, and create it if it doesn't
    if not os.path.isdir(dest_directory):
        raise ValueError(f"{dest_directory} is not a valid directory path.")

    # Get the base name of the file
    file_name = os.path.basename(file_path)

    # Create the destination path by joining the destination directory and the file name
    dest_path = os.path.join(dest_directory, file_name)

    try:
        # Move the file to the destination directory
        os.replace(file_path, dest_path)
        print(f"File '{file_name}' moved to '{dest_directory}' successfully!")
    except OSError as e:
        print(f"An error occurred while moving the file: {e}")
        raise e


if "__main__" == __name__:
    # Convertit les fichiers Excel en fichiers Parquet.
    # Fonctionne seulement si aucun Excel n'est ouvert en simultané.
    convert_xlsx_files_to_parquet(get_xlsx_files_in_folder("data"))

    # Chargement des données pour vérifier que tout s'affiche correctement
    df_px = read_data("Constituents PX_LAST data")
    df_volume = read_data("Constituents PX_VOLUME data")
    df_total_ret = read_data("Constituents TOT_RET_INDEX data")
    df_gics = pd.read_parquet("converted_data/Constituents GICS sectors.parquet")
    df_gics = imput_missing_values_gics_sectors(df_gics)
    df_ref = read_reference_index_holdings()

    print(df_ref.head())
    print(df_gics.head())
    print(df_total_ret.head())
    print(df_volume.head())
    print(df_px.head())
