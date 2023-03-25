from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


def read_reference_index_holdings() -> pd.DataFrame:
    # Read the reference index holdings in excel file and start from second row
    df = pd.read_excel("data/Reference Index Holdings.xlsx", skiprows=1)

    #df.index = pd.to_datetime(df.index)
    return df


if "__main__" == __name__:
    df = read_reference_index_holdings()
    print(df.head())
    print(df.tail())






