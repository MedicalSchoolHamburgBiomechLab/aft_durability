from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import get_path_data_root, load_merged_dataframe


@dataclass
class PlottableParameter:
    title: str
    column_name: str
    filename: str
    unit: str = None


def get_plot_path() -> Path:
    return get_path_data_root().joinpath("plots")


plot_params = [
    PlottableParameter(
        column_name="ecot_J_kg_m",
        title="ECOT",
        filename="ecot_J_kg_m",
        unit="J/kg/m"),
    PlottableParameter(
        column_name="ocot_mL_kg_km",
        title="oCoT",
        filename="ocot_mL_kg_km",
        unit="mL/kg/km"),
    PlottableParameter(
        column_name="energetic_cost_W_KG",
        title="Energetic Cost",
        filename="energetic_cost_W_KG",
        unit="W/kg"),
    PlottableParameter(
        column_name="VO2/Kg (mL/min/Kg)",
        title="Oxygen Uptake",
        filename="VO2_Kg",
        unit="mL/min/kg"),
    PlottableParameter(
        column_name="lactate",
        title="Lactate",
        filename="lactate",
        unit="mmol/L"),
    PlottableParameter(
        column_name="rpe",
        filename="rpe",
        title="RPE (Borg)"
    ),
    PlottableParameter(
        column_name="steps_per_minute",
        title="Step Rate",
        filename="steps_per_minute",
        unit="steps/min"),
    PlottableParameter(
        column_name="contact_time_ms",
        title="Contact Time",
        filename="contact_time_ms",
        unit="ms"),
    PlottableParameter(
        column_name="flight_time_ms",
        title="Flight Time",
        filename="flight_time_ms",
        unit="ms"),
    # Kinematic parameters
    # Pelvis
    PlottableParameter(
        column_name="vertical_pelvis_movement",
        title="Vertical Pelvis Movement",
        filename="vertical_pelvis_movement",
        unit="cm"),
    # Hip
    PlottableParameter(
        column_name="hip_peak_flexion_during_stance",
        title="Hip Peak Flexion",
        filename="hip_peak_flexion_during_stance",
        unit="°"),
    PlottableParameter(
        column_name="hip_flexion_at_initial_contact",
        title="Hip Flexion at Initial Contact",
        filename="hip_flexion_at_initial_contact",
        unit="°"),
    PlottableParameter(
        column_name="hip_flexion_rom_during_stance",
        title="Hip Flexion ROM",
        filename="hip_flexion_rom_during_stance",
        unit="°"),
    # Knee
    PlottableParameter(
        column_name="knee_peak_flexion_during_stance",
        title="Knee Peak Flexion",
        filename="knee_peak_flexion_during_stance",
        unit="°"),
    PlottableParameter(
        column_name="knee_flexion_at_initial_contact",
        title="Knee Flexion at Initial Contact",
        filename="knee_flexion_at_initial_contact",
        unit="°"),
    PlottableParameter(
        column_name="knee_flexion_rom_during_stance",
        title="Knee Flexion ROM",
        filename="knee_flexion_rom_during_stance",
        unit="°"),
    # Ankle
    PlottableParameter(
        column_name="ankle_peak_flexion_during_stance",
        title="Ankle Peak Dorsiflexion",
        filename="ankle_peak_flexion_during_stance",
        unit="°"),
    PlottableParameter(
        column_name="ankle_flexion_at_initial_contact",
        title="Ankle Dorsiflexion at Initial Contact",
        filename="ankle_flexion_at_initial_contact",
        unit="°"),
    PlottableParameter(
        column_name="ankle_flexion_rom_during_stance",
        title="Ankle Dorsiflexion ROM",
        filename="ankle_flexion_rom_during_stance",
        unit="°"),
    PlottableParameter(
        column_name="overstriding_hip_cm",
        title="Overstriding Hip",
        filename="overstriding_hip_cm",
        unit="cm"),
    PlottableParameter(
        column_name="overstriding_hip_deg",
        title="Overstriding Hip",
        filename="overstriding_hip_deg",
        unit="°"),
    PlottableParameter(
        column_name="overstriding_knee_cm",
        title="Overstriding Knee",
        filename="overstriding_knee_cm",
        unit="cm"),
    PlottableParameter(
        column_name="overstriding_knee_deg",
        title="Overstriding Knee",
        filename="overstriding_knee_deg",
        unit="°")
]

# pairwise comparisons (T05...T90) from R (emmeans) for the following parameters:
asterisks = {
    "ecot_J_kg_m": ["*", "*", "*", "*", "*", "*", "*", "*"],
    "ocot_mL_kg_km": ["*", "*", "*", "*", "*", "*", "*", "*"],
    "energetic_cost_W_KG": ["*", "*", "*", "*", "*", "*", "*", "*"],
    "lactate": ["", "", "", "", "", "", "", ""],
    "VO2/Kg (mL/min/Kg)": ["*", "*", "*", "*", "*", "*", "*", "*"],
    "rpe": ["", "", "*", "*", "*", "*", "", ""],
    "steps_per_minute": ["*", "*", "*", "*", "*", "", "", ""],
    "contact_time_ms": ["*", "*", "*", "*", "*", "", "", ""],
    "flight_time_ms": ["*", "*", "*", "*", "*", "", "", ""],
}


def line_plot_for_param(data: pd.DataFrame, param: PlottableParameter):
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
    fig.suptitle(f"{param.title}")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()

    return fig


def violin_plot_for_param(data: pd.DataFrame, param: PlottableParameter) -> plt.Figure:
    data = data.copy()
    # Replace missing time_condition values and force string type
    data["time_condition"] = data["time_condition"].fillna("").astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {"AFT": "s", "NonAFT": "D"}
    light_grey = (0.7, 0.7, 0.7)
    darker_grey = (0.4, 0.4, 0.4)
    palette_violin = {"AFT": light_grey, "NonAFT": darker_grey}
    gs = 0.3
    palette_line = {"AFT": (gs, gs, gs), "NonAFT": (gs, gs, gs)}

    sns.violinplot(
        data=data,
        x="time_condition",
        y=param.column_name,
        hue="shoe_condition",
        split=True,
        inner="quart",
        palette=palette_violin,
        ax=ax
    )

    # Compute means per time_condition and shoe_condition, dropping missing values
    means = (data.dropna(subset=["time_condition", param.column_name])
             .groupby(["time_condition", "shoe_condition"])[param.column_name]
             .mean().reset_index())

    sns.lineplot(
        data=data,
        x="time_condition",
        y=param.column_name,
        hue="shoe_condition",
        style="shoe_condition",
        palette=palette_line,
        errorbar=None,
        linewidth=2,
        ax=ax
    )

    # Overlay scatter points using the same grayscale as in the violin plot
    for _, row in means.iterrows():
        ax.scatter(
            row["time_condition"],
            row[param.column_name],
            color=palette_violin[row["shoe_condition"]],
            marker=markers[row["shoe_condition"]],
            s=100,
            edgecolor="black",  # set border color
            linewidths=2
        )

    # Remove legend and adjust spines
    ax.get_legend().remove()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Instead of plt.text with numeric x-values, use the actual tick positions.
    asterisks_param = asterisks.get(param.column_name)
    if asterisks_param is not None:
        xticks = ax.get_xticks()  # numeric positions for each category
        # Get sorted unique time_condition labels in order of appearance on the axis
        xticklabels = ax.get_xticklabels()
        # Use the center of each tick label for the asterisk position
        for idx, pval in enumerate(asterisks_param):
            if idx < len(xticks):
                ax.text(
                    xticks[idx],
                    ax.get_ylim()[-1],
                    s=pval,
                    ha="center",
                    va="bottom"
                )
    else:
        print(f"No asterisks for {param.column_name}")

    # Set labels and title
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(param.unit if param.unit else param.title)
    fig.suptitle(f"{param.title}")

    fig.tight_layout()
    return fig


def make_violin_plots(df: pd.DataFrame):
    path = get_plot_path().joinpath("violin_plots")
    path.mkdir(exist_ok=True)
    for param in plot_params:
        fig = violin_plot_for_param(df, param)
        path_plot = path.joinpath(f"{param.filename}.png")
        fig.savefig(path_plot)


def make_box_plot(df: pd.DataFrame, param: PlottableParameter):
    # remove all timepoints but the first and the last
    data = df[df["time_condition"].isin(["T15", "T90"])]
    fig, ax = plt.subplots(1, 2, sharey=True)
    for i, time_condition in enumerate(["T15", "T90"]):
        sns.lineplot(
            data=data[data["time_condition"] == time_condition],
            x="shoe_condition",
            y=param.column_name,
            style="shoe_condition",
            hue="shoe_condition",
            err_style="bars",
            markers={"AFT": "o", "NonAFT": "o"},
            palette="colorblind",
            markersize=10,
            errorbar="sd",
            err_kws={"capsize": 10,
                     "elinewidth": 2,
                     "markeredgewidth": 2
                     },
            legend=False,
            ax=ax[i]
        )
        gs = 0.85  # grayscale value
        sns.lineplot(
            data=data[data["time_condition"] == time_condition],
            x="shoe_condition",
            y=param.column_name,
            hue="participant_id",
            palette=[(gs, gs, gs)] * data[data["time_condition"] == time_condition]["participant_id"].nunique(),
            errorbar=None,
            legend=False,
            ax=ax[i]
        )

        ax[i].set_title(time_condition)
        ax[i].set_xlabel("")
        ax[i].set_xlim([-0.2, 1.2])
        ax[i].set_ylabel(param.unit)
    fig.suptitle(param.title)
    return fig


def make_box_plots(df: pd.DataFrame):
    path = get_plot_path().joinpath("box_plots")
    path.mkdir(exist_ok=True)
    for param in plot_params:
        fig = make_box_plot(df, param)
        path_plot = path.joinpath(f"{param.title}.png")
        fig.savefig(path_plot)


def make_change_plot(df: pd.DataFrame):
    def add_line_plot(data: pd.DataFrame,
                      param,
                      err_bar_position,
                      marker_fill: bool):
        means = data.groupby("time_min")[param].mean()
        sds = data.groupby("time_min")[param].std()
        mfc = 'k' if marker_fill else 'w'
        plt.plot(means.index,
                 means,
                 color='k',
                 marker="o",
                 markerfacecolor=mfc,
                 markersize=10,
                 )
        yerr = [np.zeros_like(sds), sds] if err_bar_position == 'top' else [sds, np.zeros_like(sds)]
        plt.errorbar(means.index,
                     means,
                     yerr=yerr,
                     fmt='none',
                     capsize=5,
                     color='k')

    # filter out time points before T15
    df = df[df["time_min"] >= 15]
    data_aft = df[df["shoe_condition"] == "AFT"]
    data_non_aft = df[df["shoe_condition"] == "NonAFT"]
    param = "ocot_change_T15"

    fig, ax = plt.subplots()
    add_line_plot(data_non_aft,
                  param,
                  err_bar_position='bottom',
                  marker_fill=False)
    add_line_plot(data_aft,
                  param,
                  err_bar_position='top',
                  marker_fill=True)
    plt.axhline(0, color='k', linestyle='-')
    for tick in df["time_min"].unique():
        plt.plot([tick, tick],
                 [-0.1, 0.1],
                 color='k')
    ax.set_xticks(df["time_min"].unique())
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False)
    ax.set_ylabel("Change in oCoT (%)")
    ax.set_ylim([-2, 10])


def main():
    df = load_merged_dataframe()
    # print(df)
    make_violin_plots(df)
    # make_box_plots(df)
    # make_change_plot(df)


if __name__ == '__main__':
    main()
