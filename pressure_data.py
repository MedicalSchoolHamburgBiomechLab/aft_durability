import multiprocessing as mp
from pathlib import Path

import pandas as pd
from labtools.systems.zebris.spatio_temnporal_parameters import analyze

from utils import get_path_data_root


def get_pressure_path():
    return get_path_data_root().joinpath("pressure")


def get_c3d_path():
    return get_pressure_path().joinpath("c3d")


def get_processed_path():
    return get_pressure_path().joinpath("processed")


def time_from_filepath(file: Path) -> str:
    return file.stem.split("_")[-1]


def get_data_frame() -> pd.DataFrame | None:
    path = get_pressure_path().joinpath("spatio_temporal_results.xlsx")
    if not path.exists():
        return None
    return pd.read_excel(path)


def save_data_frame(df: pd.DataFrame):
    path = get_pressure_path().joinpath("spatio_temporal_results.xlsx")
    df.to_excel(path, index=False)


def process_c3d(task):
    participant_id, shoe_condition, c3d_file = task
    time_condition = time_from_filepath(c3d_file)
    try:
        time_min = int(time_condition[1:])
    except (IndexError, ValueError):
        time_min = None
    results = analyze(c3d_file)
    return {
        "participant_id": participant_id,
        "shoe_condition": shoe_condition,
        "time_condition": time_condition,
        "time_min": time_min,
        "steps_per_minute": results.get('steps_per_minute'),
        "contact_time_ms": results.get('contact_time_ms'),
        "flight_time_ms": results.get('flight_time_ms'),
        "normalized_ground_contact_time": results.get('normalized_ground_contact_time'),
    }


def main(multiprocess: bool = True):
    c3d_root = get_c3d_path()
    processed_root = get_processed_path()
    processed_root.mkdir(exist_ok=True)

    cols = ["participant_id", "shoe_condition", "time_condition", "time_min",
            "steps_per_minute", "contact_time_ms", "flight_time_ms", "normalized_ground_contact_time"]
    df = pd.DataFrame(columns=cols)

    tasks = []
    # Loop over participant directories that contain "DUR"
    for entry in c3d_root.glob("*DUR*"):
        participant_id = entry.stem
        print(f"Processing participant: {participant_id}")
        for subfolder in entry.glob("*AFT*"):
            shoe_condition = subfolder.stem
            print(f"  Shoe condition: {shoe_condition}")
            for c3d_file in subfolder.glob("*.c3d"):
                tasks.append((participant_id, shoe_condition, c3d_file))

    if multiprocess:
        with mp.Pool() as pool:
            results_list = pool.map(process_c3d, tasks)
    else:
        results_list = [process_c3d(task) for task in tasks]

    df = pd.DataFrame(results_list)
    save_data_frame(df)


if __name__ == '__main__':
    main(multiprocess=True)
