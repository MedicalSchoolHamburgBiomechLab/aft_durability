import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns
from labtools.systems.cosmed.convenience import read_cosmed_excel
from matplotlib import pyplot as plt

from utils import get_path_data_root, find_k_largest_elements, get_demographics


class SpiroParameter:
    def __init__(self, column_name: str, name: str, safe_name: str):
        self._column_name = column_name
        self._name = name
        self._safe_name = safe_name

    def __str__(self):
        return f"{self.name}"

    @property
    def column_name(self):
        return self._column_name

    @property
    def name(self):
        return self._name

    @property
    def safe_name(self):
        return self._safe_name


def peronnet_massicotte_1991(v_o2: float, v_co2: float):
    """Table of nonprotein respiratory quotient: an update. Peronnet F, Massicotte D. Can J Sport Sci. 1991;16(
    1):23-29.
     VO2 and VCO2 required in L/s"""
    # RETURNS IN J/s

    return 16.89 * v_o2 + 4.84 * v_co2


def interpolate(signal: np.ndarray, time: np.ndarray, interval: float = 1) -> np.ndarray:
    # interpolate the signal to have a uniform time step
    interpolated_signal = np.interp(
        np.arange(time[0], time[-1], interval), time, signal)
    return interpolated_signal


def get_spiro_path():
    path = get_path_data_root()
    path = path.joinpath("spiro")
    return path


def get_data_frame() -> pd.DataFrame | None:
    path = get_spiro_path()
    file_path = path.joinpath("spiro_results.xlsx")
    if not file_path.exists():
        return None
    return pd.read_excel(file_path)


def save_data_frame(df: pd.DataFrame):
    path = get_spiro_path()
    path = path.joinpath("spiro_results.xlsx")
    df.to_excel(path, index=False)


def get_change_points(signal: np.ndarray, penalty: int = 1750) -> np.ndarray:
    algo = rpt.Pelt(model="l2", min_size=90).fit(signal)
    return np.array(algo.predict(pen=penalty), dtype=int)


def get_end_of_bout_json() -> dict:
    filepath = get_end_of_bout_json_path()
    if not filepath.exists():
        return dict()
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def get_end_of_bout_json_path() -> Path:
    path = get_spiro_path()
    filename = "end_of_bout_times.json"
    return path.joinpath(filename)


def save_end_of_bout_json(data: dict):
    # read the json file
    file_data = get_end_of_bout_json()
    _id = list(data.keys())[0]
    if _id in file_data:
        file_data[_id].update(data[_id])
    else:
        file_data.update(data)
    path = get_end_of_bout_json_path()
    with open(path, "w") as file:
        json.dump(file_data, file, indent=4)


def get_ends_of_bouts(data: pd.DataFrame,
                      _id: str,
                      shoe: str,
                      from_file: bool = False) -> list[int]:
    # from file:
    if from_file:
        data = get_end_of_bout_json()
        if _id not in data:
            raise ValueError(f"ID {_id} not found in data")
        if shoe not in data[_id]:
            raise ValueError(f"Shoe {shoe} not found in data")
        return data[_id][shoe]

    # Convert t (s) to numeric seconds
    data['time_s'] = data['t (s)'].dt.total_seconds()

    # signal/data segmentation (change point detection) based on the VO2/Kg signal
    signal = data['VO2/Kg (mL/min/Kg)'].values
    time = data['t (s)'].dt.total_seconds().values
    interpolated_signal = interpolate(signal, time)

    elapsed_time_s = time[-1] - time[0]
    fifteen_minutes = 15 * 60

    expected_bouts = int(elapsed_time_s / fifteen_minutes)
    change_points = get_change_points(interpolated_signal, 1500)

    # now get only the n="expected_bouts" longest bouts
    # add 0 as the first change point (start of the signal)
    change_points = np.insert(change_points, 0, 0)
    lengths = np.diff(change_points)
    longest, l_indices = find_k_largest_elements(lengths, expected_bouts)
    l_indices += 1
    save_end_of_bout_json(
        {_id: {shoe: [int(i) for i in change_points[l_indices]]}
         }
    )
    return list(change_points[l_indices])


def get_results(data: pd.DataFrame, t_end: int, params: list[str], duration: int = 180) -> dict:
    # convert timedelta to seconds
    data['time_s'] = data['t (s)'].dt.total_seconds()
    # get the data for the bout
    data_bout = data[(data['time_s'] >= t_end - duration)
                     & (data['time_s'] <= t_end)]

    out = dict()
    for param in params:
        if param not in data.columns:
            raise ValueError(f"Parameter {param} not found in data frame")
        out[param] = data_bout[param].mean()
    return out


def add_patch(ax: plt.Axes, time_point: int, patch_width_phase: int = 180):
    patch_height = ax.get_ylim()[1]
    x_patch_start_phase = time_point - patch_width_phase
    patch_height = ax.get_ylim()[1]
    ax.add_patch(plt.Rectangle((x_patch_start_phase, 0),
                               patch_width_phase, patch_height, color='red', alpha=0.1))


def analyze(data: pd.DataFrame,
            _id: str,
            shoe: str,
            params: list[str]) -> tuple[dict, plt.figure]:
    bout_ends = get_ends_of_bouts(
        data=data, _id=_id, shoe=shoe, from_file=True)
    results = dict()
    bout_ends = [int(b - 15) for b in bout_ends]

    plot_signal = "VO2/Kg (mL/min/Kg)"
    fig, ax = plt.subplots(figsize=(16, 10))
    time = data['t (s)'].dt.total_seconds().values
    ax.plot(time, data[plot_signal])
    for n_bout, bout_end in enumerate(bout_ends, 1):
        if n_bout == 1:
            # additional analysis for the first bout from minute 2-5 ...
            t5 = bout_end - 600
            key = "T05"
            results[key] = get_results(data, t5, params=params)
            add_patch(ax, t5)
            # ... and 7-10
            t10 = bout_end - 300
            key = "T10"
            results[key] = get_results(data, t10, params=params)
            add_patch(ax, t10)

        key = f"T{str(n_bout * 15).zfill(2)}"
        results[key] = get_results(data, bout_end, params=params)

        # fig2, ax2 = plt.subplots(figsize=(16, 10))
        # bout_start = max((bout_end - 900), 0)
        # roi_end = bout_end + 60

        # ax2.plot(time[bout_start:roi_end],
        #          data[plot_signal][bout_start:roi_end])

        data_bout = data[(data['time_s'] >= bout_end - 180)
                         & (data['time_s'] <= bout_end)]
        # ax2.plot(data_bout["time_s"], data_bout[plot_signal],
        #          color='red', linestyle='-')
        # add_patch(ax2, bout_end)
        ax.plot(data_bout["time_s"], data_bout[plot_signal], color='red')
        add_patch(ax, bout_end)

    return results, fig


def add_economy(result: dict, body_weight_kg: float, running_speed_kmh: float) -> dict:
    out = dict()
    v_02 = result.get('VO2 (mL/min)')
    v_co2 = result.get('VCO2 (mL/min)')
    if v_02 is not None and v_co2 is not None:
        v_co2 = result['VCO2 (mL/min)'] / 60000  # convert to L/s
        v_02 = result['VO2 (mL/min)'] / 60000  # convert to L/s
        energetic_cost_kJ_s = peronnet_massicotte_1991(v_02, v_co2)
        energetic_cost_W_KG = energetic_cost_kJ_s * 1000 / body_weight_kg
        out['energetic_cost_W_KG'] = energetic_cost_W_KG
        energetic_cost_of_transport = energetic_cost_W_KG / \
                                      (running_speed_kmh / 3.6)
        out['ecot_J_kg_m'] = energetic_cost_of_transport
        # Add oxygen cost of transport in mL/kg/km
        vO2_rel = result.get('VO2/Kg (mL/min/Kg)')
        oxygen_cost_of_transport = vO2_rel * 60 / running_speed_kmh
        out['ocot_mL_kg_km'] = oxygen_cost_of_transport

    return out


def get_body_weight_estimate(body_weight_pre: float, body_weight_post: float, time_condition: str) -> float:
    # body_weight_pre was measured at T-15 minutes (before incremental test)
    # body_weight_post was measured at T+105 minutes (after incremental test)#
    # body weight is assumed to decrease linearly between T-15 and T+105
    time = int(time_condition[1:])

    m = (body_weight_post - body_weight_pre) / 120
    b = body_weight_pre

    return m * (time + 15) + b


def main(params: list[str]):
    path = get_spiro_path()
    df = get_data_frame()
    df_summary = get_demographics()

    if df is None:
        cols = ["participant_id", "shoe_condition",
                "time_condition", "time_min"]
        cols.extend(params)
        df = pd.DataFrame(columns=cols)
    # only loop over directories that contain DUR in their name
    for entry in path.glob("*DUR*"):
        _id = entry.stem
        print(_id)
        running_speed = df_summary.loc[df_summary['participant_id']
                                       == _id, 'dauerbelastung_pace_kmh'].values[0]
        body_weight_pre = df_summary.loc[df_summary['participant_id']
                                         == _id, 'weight_pre'].values[0]
        body_weight_post = df_summary.loc[df_summary['participant_id']
                                          == _id, 'weight_post'].values[0]
        for folder in entry.glob("*AFT*"):
            shoe_condition = folder.stem
            print(f"\t {shoe_condition}")
            excel_files = [f for f in folder.glob("*.xlsx")]
            if len(excel_files) != 1:
                print(
                    f"Expected exactly one excel file. Got {len(excel_files)} Skipping folder: {folder}")
                continue
            spiro_file = excel_files[0]

            data, meta = read_cosmed_excel(spiro_file)

            results, figure = analyze(
                data, _id=_id, shoe=shoe_condition, params=params)
            # safe the figure
            path_plot = path.joinpath(
                "plots", "timeseries", f"{_id}_{shoe_condition}.png")
            figure.suptitle(f"{_id} - {shoe_condition}")
            figure.savefig(path_plot)
            plt.close(figure)

            for time_condition, result in results.items():
                # concatenate results to existing data frame
                row_values = {
                    "participant_id": [_id],
                    "shoe_condition": [shoe_condition],
                    "time_condition": [time_condition],
                    "time_min": [int(time_condition[1:])],

                }
                row_values.update(
                    {k: v for k, v in result.items() if k in params})
                # add economy parameters
                # estimate body weight based on the weight loss from pre to post measurement
                body_weight = get_body_weight_estimate(body_weight_pre, body_weight_post, time_condition)
                print(f"Estimated body weight: {body_weight}")
                row_values.update(add_economy(result,
                                              body_weight_kg=body_weight,
                                              running_speed_kmh=running_speed))

                new_row = pd.DataFrame(row_values)
                # remove the existing row if it exists
                df = df[~(
                        (df["participant_id"] == _id) &
                        (df["shoe_condition"] == shoe_condition) &
                        (df["time_condition"] == time_condition)
                )]
                df = pd.concat([df, new_row], axis=0, ignore_index=True)

    save_data_frame(df)


def analysis(params: list[SpiroParameter]):
    df = get_data_frame()
    for param in params:
        # fig = violin_plot_for_param(df, param)
        fig = line_plot_for_param(df, param)
        filename = f"{param.safe_name}_line_plot.png"
        path_plot = get_spiro_path().joinpath("plots", "parameters", filename)
        fig.savefig(path_plot)


def line_plot_for_param(data: pd.DataFrame, param: SpiroParameter):
    print(param)
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("colorblind")
    sns.set_palette(palette)
    # add an offset to the time_min values to avoid overlapping of the error bars
    dat = data.copy()
    dat['time_min'] = dat['time_min'] + \
                      dat['shoe_condition'].apply(lambda x: 0 if x == 'AFT' else 0.5)

    sns.lineplot(data=dat,
                 x="time_min",
                 y=param.column_name,
                 hue="shoe_condition",
                 markers=True,
                 style="shoe_condition",
                 errorbar="sd",
                 err_style="bars",
                 err_kws={"capsize": 5},
                 ax=ax)
    ax.set_xticks(np.arange(0, 100, 10))
    fig.suptitle(f"{param.name}")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()

    return fig


def violin_plot_for_param(data: pd.DataFrame, param: SpiroParameter):
    fig, ax = plt.subplots(figsize=(16, 10))
    palette = sns.color_palette("colorblind")
    sns.set_palette(palette)

    sns.violinplot(data=data,
                   x="time_condition",
                   y=param.column_name,
                   positions=[0, 1, 2, 5, 8, 11, 14, 17],
                   hue="shoe_condition",
                   split=True,
                   inner="quart",
                   ax=ax)
    sns.lineplot(
        data=data,
        x="time_condition",
        y=param.column_name,
        hue="shoe_condition",
        markers=True,
        style="shoe_condition",
        ax=ax
    )
    fig.suptitle(f"{param.name}")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()

    return fig


def add_change_parameters():
    path = get_spiro_path()
    df = get_data_frame()
    df_summary = get_demographics()
    # add ecot and ocot changes
    participants = df.participant_id.unique()
    shoe_conditions = df.shoe_condition.unique()
    for participant, shoe in product(participants, shoe_conditions):
        print(participant, shoe)
        params = ['ecot_J_kg_m', 'ocot_mL_kg_km']
        new_param_names = ['ecot_change', 'ocot_change']
        baseline_times = ['T05', 'T15']

        for baseline_time, (param, new_param) in product(baseline_times, zip(params, new_param_names)):
            print(param, new_param, baseline_time)
            baseline_row = df.loc[
                (df.participant_id == participant) & (df.time_condition == baseline_time) & (df.shoe_condition == shoe)]
            if baseline_row.empty:
                print(f"Baseline row not found for {participant} - {shoe}")
                continue
            param_baseline = baseline_row[param].values[0]
            param_change = df.loc[(df.participant_id == participant) & (
                        df.shoe_condition == shoe), param].values - param_baseline
            param_change_oercent = param_change / param_baseline * 100
            new_param_name = f"{new_param}_{baseline_time}"
            df.loc[
                (df.participant_id == participant) & (df.shoe_condition == shoe), new_param_name] = param_change_oercent
    save_data_frame(df)



if __name__ == '__main__':
    parameters = [
        SpiroParameter(column_name='Af (1/min)',
                       safe_name='breathing_rate',
                       name="Breathing Rate"),
        SpiroParameter(column_name='VE (L/min)',
                       safe_name="ventilation",
                       name="Ventilation"),
        SpiroParameter(column_name='VT (L(btps))',
                       safe_name="tidal_volume",
                       name="Tidal Volume"),
        SpiroParameter(column_name='VO2 (mL/min)',
                       safe_name="oxygen_consumption",
                       name="Oxygen Consumption"),
        SpiroParameter(column_name='VCO2 (mL/min)',
                       safe_name="carbon_dioxide_production",
                       name="Carbon Dioxide Production"),
        SpiroParameter(column_name='VE/VO2 (---)',
                       safe_name="ventilatory_equivalent_oxygen",
                       name="Ventilatory Equivalent Oxygen"),
        SpiroParameter(column_name='VE/VCO2 (---)',
                       safe_name="ventilatory_equivalent_carbon_dioxide",
                       name="Ventilatory Equivalent Carbon Dioxide"),
        SpiroParameter(column_name='R (---)',
                       safe_name="rer",
                       name="Respiratory Exchange Ratio"),
        SpiroParameter(column_name='VO2/Kg (mL/min/Kg)',
                       safe_name="oxygen_consumption_per_kg",
                       name="Oxygen Consumption per Kg"),
        SpiroParameter(column_name='HF (bpm)',
                       safe_name="heart_rate",
                       name="Heart Rate"),
    ]
    params = [p.column_name for p in parameters]
    main(params=params)
    add_change_parameters()
    parameters.extend(
        [SpiroParameter(column_name='energetic_cost_W_KG', safe_name="energetic_cost", name="Energetic Cost"),
         SpiroParameter(column_name='ecot_J_kg_m', safe_name="ecot", name="Energetic Cost of Transport")]
    )
    # analysis(params=parameters)
