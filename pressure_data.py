from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from labtools.systems.zebris.spatio_temnporal_parameters import analyze

from utils import get_path_data_root


def get_pressure_path():
    path = get_path_data_root()
    path = path.joinpath("pressure")
    return path


def get_c3d_path():
    path = get_pressure_path()
    path = path.joinpath("c3d")
    return path


def get_processed_path():
    path = get_pressure_path()
    path = path.joinpath("processed")
    return path


def time_from_filepath(file: Path) -> str:
    return file.stem.split("_")[-1]


def get_data_frame() -> pd.DataFrame | None:
    path = get_pressure_path()
    file_path = path.joinpath("spatio_temporal_results.xlsx")
    if not file_path.exists():
        return None
    return pd.read_excel(file_path)


def save_data_frame(df: pd.DataFrame):
    path = get_pressure_path()
    path = path.joinpath("spatio_temporal_results.xlsx")
    df.to_excel(path, index=False)


def main():
    path = get_c3d_path()
    output_path = get_processed_path()
    df = get_data_frame()
    if df is None:
        cols = ["participant_id", "shoe_condition", "time_condition", "time_min", "steps_per_minute", "contact_time_ms",
                "flight_time_ms"]
        df = pd.DataFrame(columns=cols)
    for entry in path.glob("*DUR*"):  # only loop over directories that contain DUR in their name
        participant_id = entry.stem
        print(participant_id)
        for file in entry.glob("*AFT*"):
            shoe_condition = file.stem
            print(shoe_condition)
            for c3d_file in file.glob("*.c3d"):
                time_condition = time_from_filepath(c3d_file)
                time_min = int(time_condition[1:])
                print(time_condition)
                results = analyze(c3d_file)
                print(results)
                # concatenate results to existing data frame
                new_row = pd.DataFrame({
                    "participant_id": [participant_id],
                    "shoe_condition": [shoe_condition],
                    "time_condition": [time_condition],
                    "time_min": [time_min],
                    "steps_per_minute": [results['steps_per_minute']],
                    "contact_time_ms": [results['contact_time_ms']],
                    "flight_time_ms": [results['flight_time_ms']]
                })
                df = pd.concat([df, new_row], axis=0, ignore_index=True)
    save_data_frame(df)
    #
    # results = analyze(path)
    # print(results)


def analysis():
    df = get_data_frame()
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    sns.lineplot(data=df,
                 x="time_min",
                 y="steps_per_minute",
                 hue="participant_id",
                 ax=ax, markers=True,
                 style="shoe_condition",
                 dashes=False,
                 markersize=10,
                 units="shoe_condition",
                 estimator=None)
    sns.lineplot(data=df,
                 x="time_min",
                 y="steps_per_minute",
                 hue="shoe_condition",
                 ax=ax,
                 markers=True,
                 dashes=False,
                 markersize=10)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    plt.show()


if __name__ == '__main__':
    # main()
    analysis()
