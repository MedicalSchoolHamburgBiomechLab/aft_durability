from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import get_path_data_root, load_merged_dataframe


class PlottableParameter:
    def __init__(self, title: str,
                 column_name: str,
                 unit: str = None,
                 ):
        self._title = title
        self._column_name = column_name
        self._unit = unit

    @property
    def title(self):
        return self._title

    @property
    def column_name(self):
        return self._column_name

    @property
    def unit(self):
        return self._unit


def get_plot_path() -> Path:
    return get_path_data_root().joinpath("plots")


plot_params = [
    PlottableParameter(
        column_name="ecot_J_kg_m",
        title="ECOT",
        unit="J/kg/m"),
    PlottableParameter(
        column_name="ocot_mL_kg_km",
        title="oCoT",
        unit="mL/kg/km"),
    PlottableParameter(
        column_name="energetic_cost_W_KG",
        title="Energetic Cost",
        unit="W/kg"),
    PlottableParameter(
        column_name="VO2/Kg (mL/min/Kg)",
        title="Oxygen Uptake",
        unit="mL/min/kg"),
    PlottableParameter(
        column_name="lactate",
        title="Lactate",
        unit="mmol/L"),
    PlottableParameter(
        column_name="rpe",
        title="RPE (Borg)"
    ),
    PlottableParameter(
        column_name="steps_per_minute",
        title="Step Rate",
        unit="steps/min"),
    PlottableParameter(
        column_name="contact_time_ms",
        title="Contact Time",
        unit="ms"),
    PlottableParameter(
        column_name="flight_time_ms",
        title="Flight Time",
        unit="ms"),
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
    # exclude the first two time conditions for these plots ("T05" and "T10")
    # data = data[~data["time_condition"].isin(["T05", "T10"])]

    # fig, ax = plt.subplots(figsize=(16, 10))
    fig, ax = plt.subplots()
    markers = {"AFT": "s",  # square
               "NonAFT": "D"}  # diamond

    sns.violinplot(data=data,
                   x="time_condition",
                   y=param.column_name,
                   hue="shoe_condition",
                   split=True,
                   inner="quart",
                   palette={"AFT": (0.4, 0.4, 0.4),
                            "NonAFT": (0.7, 0.7, 0.7)},
                   ax=ax)

    gs = 0.3  # grascale value
    sns.lineplot(
        data=data,
        x="time_condition",
        y=param.column_name,
        hue="shoe_condition",
        style="shoe_condition",
        markers=markers,
        palette={"AFT": (gs, gs, gs),
                 "NonAFT": (gs, gs, gs)},
        errorbar=None,
        linewidth=2,
        markersize=8,
        ax=ax
    )
    # remove legend
    ax.get_legend().remove()
    # display only left and bottom borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # try to add asterisks to the plot
    asterisks_param = asterisks.get(param.column_name)
    if asterisks_param is not None:
        y_position = ax.get_ylim()[-1]
        for idx, pval in enumerate(asterisks_param):
            plt.text(x=idx, y=y_position, s=pval)
    else:
        print(f"No asterisks for {param.column_name}")

    if param.unit:
        # ax.set_ylabel(f"{param.title} ({param.unit})")
        ax.set_ylabel(param.unit)
    else:
        ax.set_ylabel(f"{param.title}")
    ax.set_xlabel("Time (min)")
    fig.suptitle(f"{param.title}")

    fig.tight_layout()

    # manually rename the x tick labels:
    xtl = ax.get_xticklabels()
    # ax.set_xticklabels([int(lab.get_text()[1:]) for lab in xtl])

    return fig


def make_violin_plots(df: pd.DataFrame):
    path = get_plot_path().joinpath("violin_plots")
    path.mkdir(exist_ok=True)
    for param in plot_params:
        fig = violin_plot_for_param(df, param)
        path_plot = path.joinpath(f"{param.title}.png")
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


def main():
    df = load_merged_dataframe()
    print(df)
    make_violin_plots(df)
    # make_box_plots(df)


if __name__ == '__main__':
    main()
