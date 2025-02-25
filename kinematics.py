import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt  # noqa
import numpy as np
import pandas as pd
from labtools.analyses.kinetics.event_detection import get_force_events_treadmill
from labtools.signal_processing.resampling import resize_signal, convert_sample_rate
from labtools.systems.zebris.utils import get_force
from labtools.utils.c3d import load_c3d
from scipy.io import loadmat

from pressure_data import get_c3d_path
from utils import get_path_data_root, get_plot_path


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


def get_kinematics_plot_path():
    path = get_plot_path()
    path = path.joinpath("kinematics")
    path.mkdir(exist_ok=True)
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


def get_peak_during_stance(signal: np.ndarray, ic_events: np.ndarray, tc_events: np.ndarray) -> float:
    ranges = np.zeros(len(ic_events))
    for i, (ic, tc) in enumerate(zip(ic_events, tc_events)):
        ranges[i] = np.max(signal[ic:tc])
    return np.mean(ranges, dtype=float)


def get_standing_trial_index(filename_array: np.ndarray) -> int:
    flat = [a[0] for a in filename_array.ravel()]
    index = next(i for i, s in enumerate(flat) if "Stand.c3d" in s)
    return index


def get_params(data: dict,
               file_index: int,
               events: dict) -> dict:
    vertical_pelvis_movement = get_get_vertical_pelvis_displacement(
        convert_sample_rate(signal=data['Pelvis_COG'][file_index][0][:, 2],
                            f_in=85,
                            f_out=300),
        events=events['right'],
    )
    params = ['peak_flexion_during_stance', 'flexion_at_initial_contact', 'flexion_rom_during_stance']
    joints = ['Hip', 'Knee', 'Ankle']
    sides = ['Left', 'Right']
    results = dict()

    for joint, param, side in product(joints, params, sides):
        param_name = f"{joint.lower()}_{param}"
        try:
            results[param_name][side] = None
        except KeyError:
            results[param_name] = dict()
            results[param_name][side] = None
        if param == 'peak_flexion_during_stance':
            signal = convert_sample_rate(signal=data[f'{side}_{joint}_Angles'][file_index][0][:, 0], f_in=85, f_out=300)
            results[param_name][side] = get_peak_during_stance(
                signal=signal,
                ic_events=events[side.lower()]['ic'],
                tc_events=events[side.lower()]['tc'])
        elif param == 'flexion_at_initial_contact':
            signal = convert_sample_rate(signal=data[f'{side}_{joint}_Angles'][file_index][0][:, 0], f_in=85, f_out=300)
            results[param_name][side] = get_value_at_initial_contact(
                signal=signal,
                ic_events=events[side.lower()]['ic'])
        elif param == 'flexion_rom_during_stance':
            signal = convert_sample_rate(signal=data[f'{side}_{joint}_Angles'][file_index][0][:, 0], f_in=85, f_out=300)
            results[param_name][side] = get_range_during_stance(
                signal=signal,
                ic_events=events[side.lower()]['ic'],
                tc_events=events[side.lower()]['tc'])

    # summarize the sided parameters
    for param, value in results.items():
        results[param] = np.mean(list(value.values()))

    results['vertical_pelvis_movement']: vertical_pelvis_movement
    return results


def get_pressure_events(pressure_file: Path) -> dict:
    pressure_data, meta = load_c3d(pressure_file)
    fz_r, fz_l = get_force(pressure_data, separate=True)
    events = dict()
    sample_rate = pressure_data['analog_rate']
    events['left'] = get_force_events_treadmill(f_z=fz_l, sample_rate=sample_rate)
    events['right'] = get_force_events_treadmill(f_z=fz_r, sample_rate=sample_rate)
    return events


def make_plots(data: dict, file_index: int, events: dict, participant_id: str, shoe_condition: str,
               time_condition: str):
    path_plot = get_kinematics_plot_path()
    path_plot = path_plot.joinpath("angles")
    path_plot.mkdir(exist_ok=True, parents=True)

    sided_params = ['Pelvis_COG', 'Hip_Angles', 'Knee_Angles', 'Ankle_Angles', "Foot_COG"]

    colors = ['red', 'blue']
    for param in sided_params:
        path_plot_param = path_plot.joinpath(param)
        path_plot_param.mkdir(exist_ok=True, parents=True)
        fig, axs = plt.subplots(3, 1, sharex=True)
        for i_side, side in enumerate(["Left", "Right"]):
            if param == "Pelvis_COG":
                p = "Pelvis_COG"
            else:
                p = f"{side}_{param}"
            events_ipsi = events[side.lower()]
            events_contra = events["left" if side == "Right" else "right"]
            signal = data[p][file_index][0]
            col = colors[i_side]
            col_contra = colors[1 - i_side]
            for axis, axis_name in enumerate(["x", "y", "z"]):
                time_normalized_length = 101
                strides = np.zeros((len(events_ipsi['ic']) - 1, time_normalized_length))
                contralateral_ics = list()
                resamples_axis_signal = convert_sample_rate(signal=signal[:, axis], f_in=85, f_out=300)
                for i, (ic, next_ic) in enumerate(zip(events_ipsi['ic'], events_ipsi['ic'][1:])):
                    # find the contra lateral ic event
                    ic_contra_index = np.where(ic < events_contra['ic'])[0][0]
                    ic_contralateral = events_contra['ic'][ic_contra_index]
                    ic_contralateral_rel = (ic_contralateral - ic) / (next_ic - ic) * (time_normalized_length - 1)
                    contralateral_ics.append(ic_contralateral_rel)
                    stride = resamples_axis_signal[ic:next_ic]
                    time_normalized_signal = resize_signal(signal=stride, new_length=time_normalized_length)
                    strides[i, :] = time_normalized_signal
                    axs[axis].plot(time_normalized_signal, linewidth=0.1, color=col)
                    axs[axis].axvline(ic_contralateral_rel, color=col_contra, linestyle='--', linewidth=0.1)
                mean_stride = np.mean(strides, axis=0)
                mean_counterlateral = np.mean(contralateral_ics)
                axs[axis].plot(mean_stride, linewidth=2, color=col)
                axs[axis].axvline(mean_counterlateral, color=col_contra, linestyle='--', linewidth=2)
                axs[axis].set_ylabel(f"{axis_name}")
        fig.suptitle(f'{participant_id}- {shoe_condition} - {time_condition} - {param.replace("_", " ")}', fontsize=16)
        axs[2].set_xlabel("Time normalized to stride (%)")
        path_plot_file = path_plot_param.joinpath(f"{participant_id}_{shoe_condition}_{time_condition}_{param}.png")
        plt.savefig(str(path_plot_file))
        plt.close()


def analyze(mat_file: Path, shoe_condition: str, participant_id: str) -> list[dict]:
    out = list()
    kinematic_data = loadmat(mat_file)
    filenames = kinematic_data.get("FILE_NAME")
    i_standing = get_standing_trial_index(filenames)
    # get data from standing trial
    # ...

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

        make_plots(data=kinematic_data,
                   file_index=f,
                   events=events,
                   participant_id=participant_id,
                   shoe_condition=shoe_condition,
                   time_condition=time_condition)

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
                    df = pd.concat([df, new_line], ignore_index=True, axis=0)
    # save the data frame
    save_data_frame(df)


if __name__ == '__main__':
    main()
