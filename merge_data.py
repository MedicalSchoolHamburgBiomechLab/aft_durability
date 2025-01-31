from pathlib import Path

import pandas as pd
import seaborn as sns

from pressure_data import get_data_frame as get_pressure_data_frame
from spiro_data import get_data_frame as get_spiro_data_frame
from utils import get_path_data_root, get_demographics


def make_plots(df: pd.DataFrame):
    # and a violin plot of the breath_step_ratio
    sns.violinplot(data=df, x="shoe_condition", y="breath_step_ratio")
    sns.violinplot(data=df,
                   x="time_condition",
                   y="steps_per_breath_ratio",
                   hue="shoe_condition",
                   split=True,
                   inner="quart")
    # and a boxplot
    sns.boxplot(data=df,
                x="time_condition",
                y="steps_per_breath_ratio",
                hue="shoe_condition")


def get_lactate_data_frame(df_demographics: pd.DataFrame) -> pd.DataFrame:
    lactate_columns = ['dauerbelastung_lactate_15min',
                       'dauerbelastung_lactate_30min',
                       'dauerbelastung_lactate_45min',
                       'dauerbelastung_lactate_60min',
                       'dauerbelastung_lactate_75min',
                       'dauerbelastung_lactate_90min']
    shoe_condition_col = 'session_shoe'
    participant_id_col = 'participant_id'
    cols = [participant_id_col, shoe_condition_col] + lactate_columns

    df_lactate = df_demographics[cols]
    # now, make the dataframe long
    df_lactate = correct_shoe_column(df_lactate)
    df_lactate_long = pd.melt(df_lactate, id_vars=[participant_id_col, "shoe_condition"], value_vars=lactate_columns,
                              var_name="time_condition", value_name="lactate")
    df_lactate_long = correct_time_condition_column(df_lactate_long)

    return df_lactate_long


def correct_time_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    # rename the time_condition values to "T15", "T30", "T45", "T60", "T75", "T90"
    df["time_condition"] = df["time_condition"].apply(
        lambda x: "T" + x.split("_")[-1].replace("min", ""))
    return df


def correct_shoe_column(df: pd.DataFrame) -> pd.DataFrame:
    # rename session_shoe column name to shoe_condition
    df.rename(columns={"session_shoe": "shoe_condition"}, inplace=True)
    # rename session shoe values based on their content to "AFT" (contains "Nike") or "NonAFT" (contains "Brooks")
    df["shoe_condition"] = df["shoe_condition"].apply(lambda x: "AFT" if "Nike" in x else "NonAFT")
    return df


def get_rpe_data_frame(df_demographics: pd.DataFrame) -> pd.DataFrame:
    rpe_columns = [
        "dauerbelastung_RPE_15min",
        "dauerbelastung_RPE_30min",
        "dauerbelastung_RPE_45min",
        "dauerbelastung_RPE_60min",
        "dauerbelastung_RPE_75min",
        "dauerbelastung_RPE_90min",
    ]
    shoe_condition_col = 'session_shoe'
    participant_id_col = 'participant_id'
    cols = [participant_id_col, shoe_condition_col] + rpe_columns
    df_rpe = df_demographics[cols]
    df_rpe = correct_shoe_column(df_rpe)
    df_rpe_long = pd.melt(df_rpe, id_vars=[participant_id_col, "shoe_condition"], value_vars=rpe_columns,
                          var_name="time_condition", value_name="rpe")
    df_rpe_long = correct_time_condition_column(df_rpe_long)
    return df_rpe_long


def add_breath_step_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["steps_per_breath_ratio"] = df["steps_per_minute"] / df["Af (1/min)"]
    return df


def get_merged_dataframe_path() -> Path:
    return get_path_data_root().joinpath("merged_data.xlsx")


def save_merged_dataframe(df: pd.DataFrame):
    path = get_merged_dataframe_path()
    df.to_excel(path, index=False)


def load_merged_dataframe() -> pd.DataFrame:
    path = get_merged_dataframe_path()
    return pd.read_excel(path)


def main():
    spiro_df = get_spiro_data_frame()
    pressure_df = get_pressure_data_frame()

    df_demographics = get_demographics()
    df_lactate = get_lactate_data_frame(df_demographics)
    df_rpe = get_rpe_data_frame(df_demographics)

    # merge dataframe on participant_id, shoe_condition, and time_condition
    merged_df = pd.merge(spiro_df, pressure_df, on=["participant_id", "shoe_condition", "time_condition", "time_min"])
    merged_df = pd.merge(merged_df, df_lactate, on=["participant_id", "shoe_condition", "time_condition"], how="left")
    merged_df = pd.merge(merged_df, df_rpe, on=["participant_id", "shoe_condition", "time_condition"], how="left")
    print(merged_df)
    # add ratio of breathing frequency and step rate to the merged dataframe
    # merged_df = add_breath_step_ratio(merged_df)
    # save the merged dataframe
    save_merged_dataframe(merged_df)
    # make_plots(merged_df)


if __name__ == "__main__":
    main()
