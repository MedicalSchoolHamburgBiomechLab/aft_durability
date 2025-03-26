import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def get_path_data_root() -> Path:
    load_dotenv()
    path_env = os.getenv("PATH_DATA_ROOT")
    if path_env is None:
        raise ValueError("The environment variable PATH_DATA_ROOT is not set.")
    path = Path(path_env)
    assert path.exists()
    return path


def get_plot_path():
    root = get_path_data_root()
    path = root.joinpath("plots")
    path.mkdir(exist_ok=True)
    return path


def get_demographics() -> pd.DataFrame:
    path = get_path_data_root()
    filename = "Durability.xlsx"
    path = path.joinpath(filename)
    assert path.exists()
    return pd.read_excel(path)


def get_merged_dataframe_path() -> Path:
    return get_path_data_root().joinpath("merged_data.xlsx")


def save_merged_dataframe(df: pd.DataFrame):
    path = get_merged_dataframe_path()
    # add datetime stamp to the file name
    # path = path.parent.joinpath(pd.Timestamp.now().strftime("%Y%m%d_%H%M%S") + "_" + path.stem + path.suffix)
    df.to_excel(path, index=False)


def load_merged_dataframe() -> pd.DataFrame:
    path = get_merged_dataframe_path()
    return pd.read_excel(path)


# todo: add the following function to the labtools repo
def find_k_largest_elements(arr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the k largest elements in the array arr and return their values and indices.
    :param arr:
    :param k:
    :return: tuple of two numpy arrays: the first array contains the k largest elements, the second array contains the
    indices of the k largest elements in the original array.
    """
    if k > len(arr):
        raise ValueError("k must be less than or equal to the length of the array")
    elif k == len(arr):
        return arr, np.arange(len(arr))

    indices = np.argpartition(arr, -k)[-k:]
    indices.sort()
    values = arr[indices]
    return values, indices
