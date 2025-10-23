import re
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils import get_path_data_root


def get_path_subanalysis_root() -> Path:
    path = get_path_data_root().joinpath("Durability Subset Analysis")
    return path


def get_step_test_data_frame() -> pd.DataFrame:
    path = get_path_subanalysis_root().joinpath("data_step_test_wide.xlsx")
    df = pd.read_excel(path)
    return df


def get_durability_test_data_frame() -> pd.DataFrame:
    path = get_path_subanalysis_root().joinpath("data_durability_test_wide.xlsx")
    df = pd.read_excel(path)
    return df


def get_path_dur_test_long() -> Path:
    path = get_path_subanalysis_root().joinpath("data_durability_test_long.xlsx")
    return path


def get_path_step_test_long() -> Path:
    path = get_path_subanalysis_root().joinpath("data_step_test_long.xlsx")
    return path


def get_df_step_test_long() -> pd.DataFrame:
    path = get_path_step_test_long()
    df = pd.read_excel(path)
    return df


def get_df_durability_test_long() -> pd.DataFrame:
    path = get_path_dur_test_long()
    df = pd.read_excel(path)
    return df


def make_durability_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    id_vars = ['participant_id', 'shoe_condition']
    pattern = r'^(HR|DFA|BLA)_(T\d+)$'

    value_vars = [c for c in df_wide.columns if c not in id_vars and re.match(pattern, c)]

    df_long = df_wide.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="variable_time",
        value_name="value"
    )

    df_long[['variable', 'time_point']] = df_long['variable_time'].str.extract(pattern)

    df_long = (
        df_long
        .pivot_table(
            index=id_vars + ['time_point'],
            columns='variable',
            values='value',
            aggfunc='first'
        )
        .reset_index()
    )

    df_long.columns.name = None
    return df_long


def make_step_test_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    id_vars = ['participant_id', 'shoe_condition', 'time_point']
    pattern = r'^(HR|DFA|BLA)_(step_\d+)$'
    value_vars = [c for c in df_wide.columns if c not in id_vars and re.match(pattern, c)]
    df_long = df_wide.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="variable_step",
        value_name="value"
    )
    df_long[['variable', 'step_number']] = df_long['variable_step'].str.extract(pattern)
    df_long = (
        df_long
        .pivot_table(
            index=id_vars + ['step_number'],
            columns='variable',
            values='value',
            aggfunc='first'
        )
        .reset_index()
    )
    df_long.columns.name = None
    return df_long


def add_left_censored_column(df: pd.DataFrame) -> pd.DataFrame:
    # add column "bla_left_censored" which is True if BLA == 0.0
    df['bla_left_censored'] = df['BLA'] == 0.0
    # set bla value 0.0 back to 0.5
    df.loc[df['BLA'] == 0.0, 'BLA'] = 0.5
    return df


def make_long_format_data_frames():
    # Step test data
    df_step_test_wide = get_step_test_data_frame()
    df_step_test_long = make_step_test_long_format(df_step_test_wide)
    df_step_test_long = add_left_censored_column(df_step_test_long)
    df_step_test_long.to_excel(get_path_step_test_long(), index=False)
    # Durability test data
    df_dur_test_wide = get_durability_test_data_frame()
    df_dur_test_long = make_durability_long_format(df_dur_test_wide)
    df_dur_test_long.to_excel(get_path_dur_test_long(), index=False)


def make_step_test_plot(df_long: pd.DataFrame, variable: str, title: str, filename: str, y_label: str,
                        add_single_lines: bool = False,

                        ):
    # Make step test plots (hue: shoe_condition, style: time_point)
    # remove step 6
    df_long = df_long[df_long['step_number'] != 'step_6']
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=df_long,
        x='step_number',
        y=variable,
        hue='shoe_condition',
        style='time_point',
        markers=True,
        dashes=True,
        ax=ax
    )
    if False:
        sns.lineplot(
            data=df_long,
            x='step_number',
            y=variable,
            hue='shoe_condition',
            style='participant_id',
            legend=False,
            alpha=0.3,
            ax=ax
        )
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Step No.")
    plt.savefig(get_path_subanalysis_root().joinpath(filename))
    plt.close(fig)


def make_durability_test_plot(df_long: pd.DataFrame,
                              variable: str,
                              title: str,
                              filename: str,
                              y_label: str,
                              add_single_lines: bool = False,
                              ):
    # Make durability test plots (hue: shoe_condition)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=df_long,
        x='time_point',
        y=variable,
        hue='shoe_condition',
        markers=True,
        dashes=True,
        ax=ax
    )
    if add_single_lines:
        sns.lineplot(
            data=df_long,
            x='time_point',
            y=variable,
            hue='shoe_condition',
            style='participant_id',
            legend=False,
            alpha=0.3,
            ax=ax
        )
    ax.set_title(title)
    ax.set_xlabel("Time Point")
    ax.set_ylabel(y_label)

    plt.savefig(get_path_subanalysis_root().joinpath(filename))
    plt.close(fig)


def make_plots():
    df_step = get_df_step_test_long()
    df_durability = get_df_durability_test_long()
    y_label_hr = "Heart Rate (bpm)"
    y_label_dfa = "DFAa1"
    y_label_bla = "Blood Lactate (mmol/l)"

    make_step_test_plot(
        df_step,
        variable='HR',
        title='Heart Rate during Step Test',
        filename='step_test_heart_rate.png',
        y_label=y_label_hr,
        add_single_lines=True
    )
    make_step_test_plot(
        df_step,
        variable='DFA',
        title='DFA during Step Test',
        filename='step_test_dfa.png',
        y_label=y_label_dfa,
        add_single_lines=True
    )
    make_step_test_plot(
        df_step,
        variable='BLA',
        title='Blood Lactate during Step Test',
        filename='step_test_bla.png',
        y_label=y_label_bla,
        add_single_lines=True
    )
    make_durability_test_plot(
        df_durability,
        variable='HR',
        title='Heart Rate during Durability Test',
        filename='durability_test_heart_rate.png',
        y_label=y_label_hr,
        add_single_lines=True
    )
    make_durability_test_plot(
        df_durability,
        variable='DFA',
        title='DFA during Durability Test',
        filename='durability_test_dfa.png',
        y_label=y_label_dfa,
        add_single_lines=True
    )
    make_durability_test_plot(
        df_durability,
        variable='BLA',
        title='Blood Lactate during Durability Test',
        filename='durability_test_bla.png',
        y_label=y_label_bla,
        add_single_lines=True
    )


def main():
    make_long_format_data_frames()
    make_plots()


if __name__ == '__main__':
    main()
