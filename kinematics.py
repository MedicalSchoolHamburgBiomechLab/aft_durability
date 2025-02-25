import warnings
from pathlib import Path

import matplotlib.pyplot as plt  # noqa
import numpy as np
import pandas as pd
from labtools.analyses.kinetics.event_detection import get_force_events_treadmill
from labtools.systems.zebris.utils import get_force
from labtools.utils.c3d import load_c3d
from scipy.io import loadmat

from pressure_data import get_c3d_path
from utils import get_path_data_root


def get_kinematics_path():
    path = get_path_data_root()
    path = path.joinpath("kinematics")
    return path


def get_raw_path():
    path = get_kinematics_path()
    path = path.joinpath("raw")
    return path


def get_processed_path():
    path = get_kinematics_path()
    path = path.joinpath("processed")
    return path


def get_data_frame() -> pd.DataFrame | None:
    path = get_kinematics_path()
    file_path = path.joinpath("kinematics_results.xlsx")
    if not file_path.exists():
        cols = ["participant_id", "shoe_condition", "time_condition"]
        return pd.DataFrame(columns=cols)
    return pd.read_excel(file_path)


def save_data_frame(df: pd.DataFrame):
    path = get_kinematics_path()
    path = path.joinpath("kinematics_results.xlsx")
    df.to_excel(path, index=False)


def get_matching_pressure_file(participant_id: str, shoe_condition: str, time_condition: str) -> Path:
    path = get_c3d_path()

    for p in Path(path).rglob('*'):
        if participant_id not in str(p):
            continue
        if not p.is_file():
            continue
        if p.parent.stem != shoe_condition:
            continue
        if time_condition not in str(p):
            continue
        return p


def get_time_condition_from_filename(mat_file_path: Path, shoe_condition: str, particpant_id: str) -> str | None:
    if "Markerless" not in mat_file_path.stem:
        return None
    num_trial = int(mat_file_path.stem.split("Markerless ")[1])
    time_conditions_list = [
        "T01",
        "T03",
        "T05",
        "T07",
        "T10",
        "T15",
        "T30",
        "T45",
        "T60",
        "T75",
        "T90"
    ]
    # DUR02     NonAFT      T45                 missing
    if (particpant_id == "DUR02") & (shoe_condition == "NonAFT"):
        time_conditions_list.remove("T45")
    return time_conditions_list[num_trial - 1]


def resample_signal(signal: np.ndarray, f_in: int, f_out: int) -> np.ndarray:
    return np.interp(np.arange(0, len(signal), f_in / f_out), np.arange(0, len(signal)), signal)


def get_get_vertical_pelvis_displacement(
        pelvis_trajectory_z: np.ndarray,
        events: dict,
) -> float:
    # find the peak to peak value of the vertical pelvis displacement during the stride
    ic_events = events['ic']
    if len(pelvis_trajectory_z) < ic_events[-1]:
        raise ValueError("The last IC event is before the end of the signal. Did you forget to resample the signal?")
    ptps = np.zeros(len(ic_events) - 1)
    for stride, (ic, next_ic) in enumerate(zip(ic_events, ic_events[1:])):
        ptps[stride] = np.ptp(pelvis_trajectory_z[ic:next_ic])
    return np.mean(ptps, dtype=float)


def get_value_at_initial_contact(signal: np.ndarray, ic_events: np.ndarray) -> float:
    return np.mean([signal[ic] for ic in ic_events], dtype=float)


def get_range_during_stance(signal: np.ndarray, ic_events: np.ndarray, tc_events: np.ndarray) -> float:
    ranges = np.zeros(len(ic_events))
    for i, (ic, tc) in enumerate(zip(ic_events, tc_events)):
        ranges[i] = np.ptp(signal[ic:tc])
    return np.mean(ranges, dtype=float)


def get_standing_trial_index(filename_array: np.ndarray) -> int:
    flat = [a[0] for a in filename_array.ravel()]
    index = next(i for i, s in enumerate(flat) if "Stand.c3d" in s)
    return index


def get_params(data: dict,
               file_index: int,
               events: dict) -> dict:
    vertical_pelvis_movement = get_get_vertical_pelvis_displacement(
        resample_signal(signal=data['Pelvis_COG'][file_index][0][:, 2],
                        f_in=85,
                        f_out=300),
        events=events['right'],
    )
    # sided parameters
    knee_flexion_angle_at_initial_contact = dict()
    knee_flexion_angle_rom = dict()
    for side in ["Left", "Right"]:
        knee_flexion_angle_at_initial_contact[side] = get_value_at_initial_contact(
            signal=resample_signal(signal=data[f'{side}_Knee_Angles'][file_index][0][:, 0],
                                   f_in=85,
                                   f_out=300),
            ic_events=events[side.lower()]['ic'],
        )
        knee_flexion_angle_rom[side] = get_range_during_stance(
            signal=resample_signal(signal=data[f'{side}_Knee_Angles'][file_index][0][:, 0],
                                   f_in=85,
                                   f_out=300),
            ic_events=events[side.lower()]['ic'],
            tc_events=events[side.lower()]['tc'],
        )

    return {
        "vertical_pelvis_movement_m": vertical_pelvis_movement,
        "knee_flexion_angle_at_initial_contact_deg": np.mean(list(knee_flexion_angle_at_initial_contact.values())),
        "knee_flexion_angle_rom_deg": np.mean(list(knee_flexion_angle_rom.values())),
    }


def get_pressure_events(pressure_file: Path) -> dict:
    pressure_data, meta = load_c3d(pressure_file)
    fz_r, fz_l = get_force(pressure_data, separate=True)
    events = dict()
    sample_rate = pressure_data['analog_rate']
    events['left'] = get_force_events_treadmill(f_z=fz_l, sample_rate=sample_rate)
    events['right'] = get_force_events_treadmill(f_z=fz_r, sample_rate=sample_rate)
    return events


def analyze(mat_file: Path, shoe_condition: str, participant_id: str) -> list[dict]:
    out = list()
    kinematic_data = loadmat(mat_file)
    filenames = kinematic_data.get("FILE_NAME")
    i_standing = get_standing_trial_index(filenames)
    # get data from standing trial

    # get data from dynamic trials
    for f, fn in enumerate(filenames):
        if f == i_standing:
            continue
        filename = Path(fn[0][0])
        time_condition = get_time_condition_from_filename(filename, shoe_condition, participant_id)
        if time_condition is None:
            raise ValueError(f"The time condition could not be determined from the filename. Filename: {filename}")
        # get the matching pressure file
        pressure_file = get_matching_pressure_file(participant_id, shoe_condition, time_condition)
        if pressure_file is None:
            warnings.warn(
                f"No matching pressure file found for participant {participant_id}, shoe condition {shoe_condition}, time condition {time_condition}")
            continue
            # raise ValueError(f"No matching pressure file found for participant {participant_id}, shoe condition {shoe_condition}, time condition {time_condition}")
        # get the events from the pressure file
        events = get_pressure_events(pressure_file)

        params = get_params(data=kinematic_data, file_index=f, events=events)
        result = {
            "participant_id": participant_id,
            "shoe_condition": shoe_condition,
            "time_condition": time_condition,
            **params
        }
        out.append(result)
    return out


def main():
    path = get_raw_path()
    df = pd.DataFrame()

    for participant_dir in path.glob("*DUR*"):  # only loop over directories that contain DUR in their name
        participant_id = participant_dir.stem
        print(participant_id)
        for shoe_dir in participant_dir.glob("*AFT*"):
            shoe_condition = shoe_dir.stem
            print(shoe_condition)
            mat_files = [file for file in shoe_dir.glob("*.mat")]
            for mat_file in mat_files:
                results = analyze(mat_file, shoe_condition, participant_id)
                for result in results:
                    new_line = pd.DataFrame(result, index=[0])
                    # df_2 = pd.merge(df, new_line, how="outer", on=["participant_id", "shoe_condition", "time_condition"])
                    # df = df.combine_first(new_line)
                    df = pd.concat([df, new_line], ignore_index=True, axis=0)
    # save the data frame
    save_data_frame(df)


if __name__ == '__main__':
    main()
