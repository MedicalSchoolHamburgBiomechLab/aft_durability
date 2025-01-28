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
    interpolated_signal = np.interp(np.arange(time[0], time[-1], interval), time, signal)
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
    return np.array(algo.predict(pen=penalty))


def get_ends_of_bouts(data: pd.DataFrame) -> list[int]:
    # Convert t (s) to numeric seconds
    data['time_s'] = data['t (s)'].dt.total_seconds()

    # signal/data segmentation (change point detection) based on the VO2/Kg signal
    signal = data['VO2/Kg (mL/min/Kg)'].values
    time = data['t (s)'].dt.total_seconds().values
    interpolated_signal = interpolate(signal, time)

    elapsed_time_s = time[-1] - time[0]
    fifteen_minutes = 15 * 60

    expected_bouts = int(elapsed_time_s / fifteen_minutes)
    change_points = get_change_points(interpolated_signal, 1750)

    # now get only the n="expected_bouts" longest bouts
    change_points = np.insert(change_points, 0, 0)  # add 0 as the first change point (start of the signal)
    lengths = np.diff(change_points)
    longest, l_indices = find_k_largest_elements(lengths, expected_bouts)
    l_indices += 1
    return list(change_points[l_indices])


def get_results(data: pd.DataFrame, t_end: int, params: list[str], duration: int = 180) -> dict:
    # convert timedelta to seconds
    data['time_s'] = data['t (s)'].dt.total_seconds()
    # get the data for the bout
    data_bout = data[(data['time_s'] >= t_end - duration) & (data['time_s'] <= t_end)]
    out = dict()
    for param in params:
        if param not in data.columns:
            raise ValueError(f"Parameter {param} not found in data frame")
        out[param] = data_bout[param].mean()
    return out


def analyze(data: pd.DataFrame, params: list[str]) -> dict:
    bout_ends = get_ends_of_bouts(data=data)
    results = dict()
    for n_bout, bout_end in enumerate(bout_ends, 1):
        if n_bout == 1:
            # additional analysis for the first bout from minute 2-5 ...
            t5 = bout_end - 600
            key = "T05"
            results[key] = get_results(data, t5, params=params)
            # ... and 7-10
            t10 = bout_end - 300
            key = "T10"
            results[key] = get_results(data, t10, params=params)

        key = f"T{str(n_bout * 15).zfill(2)}"
        results[key] = get_results(data, bout_end, params=params)

    return results


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
        energetic_cost_of_transport = energetic_cost_W_KG / (running_speed_kmh / 3.6)
        out['ecot_J_kg_m'] = energetic_cost_of_transport

    return out


def main(params: list[str]):
    path = get_spiro_path()
    df = get_data_frame()
    df_summary = get_demographics()

    if df is None:
        cols = ["participant_id", "shoe_condition", "time_condition", "time_min"]
        cols.extend(params)
        df = pd.DataFrame(columns=cols)
    for entry in path.glob("*DUR*"):  # only loop over directories that contain DUR in their name
        _id = entry.stem
        running_speed = df_summary.loc[df_summary['participant_id'] == _id, 'dauerbelastung_pace_kmh'].values[0]
        body_weight_pre = df_summary.loc[df_summary['participant_id'] == _id, 'weight_pre'].values[0]
        body_weight_post = df_summary.loc[df_summary['participant_id'] == _id, 'weight_post'].values[0]
        body_weight = (body_weight_pre + body_weight_post) / 2
        for folder in entry.glob("*AFT*"):
            shoe_condition = folder.stem
            print(shoe_condition)
            excel_files = [f for f in folder.glob("*.xlsx")]
            if len(excel_files) != 1:
                print(f"Expected exactly one excel file. Got {len(excel_files)} Skipping folder: {folder}")
                continue
            spiro_file = excel_files[0]

            data, meta = read_cosmed_excel(spiro_file)

            results = analyze(data, params=params)
            for time_condition, result in results.items():
                # concatenate results to existing data frame
                row_values = {
                    "participant_id": [_id],
                    "shoe_condition": [shoe_condition],
                    "time_condition": [time_condition],
                    "time_min": [int(time_condition[1:])],

                }
                row_values.update({k: v for k, v in result.items() if k in params})
                # add economy parameters
                row_values.update(add_economy(result,
                                              body_weight_kg=body_weight,
                                              running_speed_kmh=running_speed))

                new_row = pd.DataFrame(row_values)
                df = pd.concat([df, new_row], axis=0, ignore_index=True)
    save_data_frame(df)


def analysis(params: list[SpiroParameter]):
    df = get_data_frame()
    for param in params:
        fig = violin_plot_for_param(df, param)
        filename = f"{param.safe_name}_violin_plot.png"
        path_plot = get_spiro_path().joinpath("plots", filename)
        fig.savefig(path_plot)


def violin_plot_for_param(data: pd.DataFrame, param: SpiroParameter):
    fig, ax = plt.subplots(figsize=(16, 10))
    palette = sns.color_palette("colorblind")
    sns.set_palette(palette)
    sns.violinplot(data=data,
                   x="time_condition",
                   y=param.column_name,
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


if __name__ == '__main__':
    parameters = [
        SpiroParameter(column_name='Af (1/min)', safe_name='breathing_rate', name="Breathing Rate"),
        SpiroParameter(column_name='VE (L/min)', safe_name="ventilation", name="Ventilation"),
        SpiroParameter(column_name='VO2 (mL/min)', safe_name="oxygen_consumption", name="Oxygen Consumption"),
        SpiroParameter(column_name='VCO2 (mL/min)', safe_name="carbon_dioxide_production",
                       name="Carbon Dioxide Production"),
        SpiroParameter(column_name='R (---)', safe_name="rer", name="Respiratory Exchange Ratio"),
        SpiroParameter(column_name='VO2/Kg (mL/min/Kg)', safe_name="oxygen_consumption_per_kg",
                       name="Oxygen Consumption per Kg"),
        SpiroParameter(column_name='HF (bpm)', safe_name="heart_rate", name="Heart Rate"),
    ]
    params = [p.column_name for p in parameters]
    main(params=params)
    parameters.extend(
        [SpiroParameter(column_name='energetic_cost_W_KG', safe_name="energetic_cost", name="Energetic Cost"),
         SpiroParameter(column_name='ecot_J_kg_m', safe_name="ecot", name="Energetic Cost of Transport")]
    )
    analysis(params=parameters)
