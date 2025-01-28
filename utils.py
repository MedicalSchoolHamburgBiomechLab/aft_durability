import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def get_path_data_root():
    load_dotenv()
    path = Path(os.getenv("PATH_DATA_ROOT"))
    assert path.exists()
    return path


def get_demographics() -> pd.DataFrame:
    path = get_path_data_root()
    filename = "Durability .xlsx"
    path = path.joinpath(filename)
    assert path.exists()
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
