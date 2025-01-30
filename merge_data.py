import pandas as pd    
import matplotlib.pyplot as plt
import seaborn as sns

from spiro_data import get_data_frame as get_spiro_data_frame
from pressure_data import get_data_frame as get_pressure_data_frame
from utils import get_path_data_root



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

if __name__ == "__main__":
    spiro_df = get_spiro_data_frame()
    pressure_df = get_pressure_data_frame()
    # merge dataframe on participant_id, shoe_condition, and time_condition
    merged_df = pd.merge(spiro_df, pressure_df, on=["participant_id", "shoe_condition", "time_condition", "time_min"])
    print(merged_df)
    # add ratio of breathing frequency and step rate to the merged dataframe
    merged_df["steps_per_breath_ratio"] = merged_df["steps_per_minute"] / merged_df["Af (1/min)"] 
    merged_df.to_excel(get_path_data_root().joinpath("merged_data.xlsx"), index=False)
    make_plots(merged_df)

