from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Union
import pandas as pd


def xlsx_to_parquet(xlsx_file: str, parquet_file: str) -> None:
    df = pd.read_excel(xlsx_file)
    df.to_parquet(parquet_file)


def convert_xlsx_files_to_parquet(
    file_list: List[str], output_folder: str = "data"
) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for xlsx_file in file_list:
        file_name, _ = os.path.splitext(os.path.basename(xlsx_file))
        parquet_file = os.path.join(output_folder, f"{file_name}.parquet")
        xlsx_to_parquet(xlsx_file, parquet_file)
        print(f"Converted {xlsx_file} to {parquet_file}")


def get_xlsx_files_in_folder(folder: str) -> List[str]:
    return [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.endswith(".xlsx")
    ]


def convert_dates(df: pd.DataFrame, date_column: str = "Index Date"):
    # Convertir les dates en chaînes de caractères
    df[date_column] = df[date_column].astype(str)
    # Convertir les chaînes de caractères en objets datetime avec le format souhaité
    df[date_column] = pd.to_datetime(df[date_column], format="%Y%m%d")
    # Convertir les objets datetime en chaînes de caractères avec le format souhaité (AAAA-MM-JJ)
    df[date_column] = df[date_column].dt.strftime("%Y-%m-%d")

    return df


def read_reference_index_holdings() -> pd.DataFrame:
    # Read the reference index holdings in excel file and start from second row
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


def read_gics_sectors() -> pd.DataFrame:
    df = pd.read_parquet("converted_data/Constituents GICS sectors.parquet")

    return df


def read_data(file_name: str) -> pd.DataFrame:
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


if "__main__" == __name__:
    df_px = read_data("Constituents PX_LAST data")
    df_volume = read_data("Constituents PX_VOLUME data")
    df_total_ret = read_data("Constituents TOT_RET_INDEX data")
    df_gics = read_gics_sectors()
    df_ref = read_reference_index_holdings()
    print(df_ref.head())
    print(df_gics.head())
    print(df_total_ret.head())
    print(df_volume.head())
    print(df_px.head())
