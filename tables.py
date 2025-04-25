import pandas as pd

from utils import load_merged_dataframe, get_path_data_root, get_demographics


def make_param_table():
    df = load_merged_dataframe()
    physiological_columns = ['VO2/Kg (mL/min/Kg)',
                             'HF (bpm)',
                             'energetic_cost_W_KG',
                             'ecot_J_kg_m',
                             'ocot_mL_kg_km',
                             'R (---)',
                             'VE (L/min)',
                             'Af (1/min)',
                             'ecot_change_T05',
                             'ocot_change_T05',
                             'ecot_change_T15',
                             'ocot_change_T15',
                             'lactate',
                             'rpe']
    kinematics_columns = ["hip_peak_flexion_during_stance",
                          "hip_flexion_at_initial_contact",
                          "hip_flexion_rom_during_stance",
                          "knee_peak_flexion_during_stance",
                          "knee_flexion_at_initial_contact",
                          "knee_flexion_rom_during_stance",
                          "ankle_peak_flexion_during_stance",
                          "ankle_flexion_at_initial_contact",
                          "ankle_flexion_rom_during_stance",
                          "overstriding_hip_cm",
                          "overstriding_knee_cm",
                          "overstriding_hip_deg",
                          "overstriding_knee_deg",
                          "vertical_pelvis_movement"]
    spatio_temporal_columns = ['steps_per_minute',
                               'contact_time_ms',
                               'flight_time_ms',
                               "normalized_ground_contact_time"]
    columns = physiological_columns.copy()
    columns.extend(spatio_temporal_columns)
    columns.extend(kinematics_columns)

    # Group by shoe_condition and time_condition
    grp = df.groupby(['shoe_condition', 'time_condition'])
    table_means = grp.mean(numeric_only=True)[columns].T
    table_stds = grp.std(numeric_only=True)[columns].T

    # Compute aggregated values over all shoe_conditions
    grp_all = df.groupby('time_condition')
    table_means_all = grp_all.mean(numeric_only=True)[columns].T
    table_stds_all = grp_all.std(numeric_only=True)[columns].T

    # Stack shoe_condition from columns to row index
    table_means = table_means.stack(level='shoe_condition', dropna=False)
    table_stds = table_stds.stack(level='shoe_condition', dropna=False)
    table_means.index.set_names(['Parameter', 'Shoe'], inplace=True)
    table_stds.index.set_names(['Parameter', 'Shoe'], inplace=True)

    # Create MultiIndex for aggregated rows with shoe_condition 'both'
    agg_index = pd.MultiIndex.from_tuples([(param, 'both') for param in table_means_all.index],
                                          names=['Parameter', 'Shoe'])
    table_means_all.index = agg_index
    table_stds_all.index = agg_index

    # Combine the detailed and aggregated tables
    table_means = pd.concat([table_means, table_means_all]).sort_index()
    table_stds = pd.concat([table_stds, table_stds_all]).sort_index()

    # Combine mean and std into "M ± STD"
    table_combined = table_means.copy()
    for col in table_means.columns:
        print(col)
        table_combined[col] = table_means[col].combine(table_stds[col],
                                                       lambda m, s: f"{m:.2f} ± {s:.2f}")
        # table_combined[col] = (
        #         table_means[col].astype(float).map("{:.2f}".format)
        #         + " ± " +
        #         table_stds[col].astype(float).map("{:.2f}".format)
        # )

    path = get_path_data_root().joinpath('tables')
    path.mkdir(exist_ok=True)
    with pd.ExcelWriter(path.joinpath('parameter_table.xlsx')) as writer:
        table_combined.to_excel(writer, sheet_name='combined')
        table_means.to_excel(writer, sheet_name='means')
        table_stds.to_excel(writer, sheet_name='stds')


def make_demographics_table():
    df_demo = get_demographics()
    # exclude DUR08 and DUR11 from the demographics table
    df_demo = df_demo[(df_demo['participant_id'] != 'DUR11') & (df_demo['participant_id'] != 'DUR08')]
    # only take values of the first session
    df_demo = df_demo[(df_demo['session_nr'] == 1)]
    df_demo.rename(columns={"sex ": "sex"}, inplace=True)

    columns = ['age', 'height', 'weight_pre', 'sex', 'dauerbelastung_pace_kmh', 'WA_max']

    df = df_demo[columns]
    df.loc[:, 'bmi'] = df['weight_pre'] / (df['height'] / 100) ** 2

    columns.remove('sex')
    columns.extend(['bmi'])
    # Get total means and stds
    table_means_total = df[columns].mean()
    table_stds_total = df[columns].std()
    # Get sex specific means and stds
    grp = df.groupby(['sex'])
    table_means = grp.mean(numeric_only=True)[columns].T
    table_stds = grp.std(numeric_only=True)[columns].T

    # Combine the detailed and aggregated tables
    table_means_total = pd.DataFrame(table_means_total)
    table_means_total.columns = ['Total']
    table_stds_total = pd.DataFrame(table_stds_total)
    table_stds_total.columns = ['Total']

    table_means = pd.concat([table_means, table_means_total], axis=1)
    table_stds = pd.concat([table_stds, table_stds_total], axis=1)

    # Combine mean and std into "M ± STD"
    table_combined = table_means.copy()
    for col in table_means.columns:
        print(col)
        table_combined[col] = table_means[col].combine(table_stds[col],
                                                       lambda m, s: f"{m:.2f} ± {s:.2f}")

    path = get_path_data_root().joinpath('tables')
    path.mkdir(exist_ok=True)
    with pd.ExcelWriter(path.joinpath('demographics_table.xlsx')) as writer:
        table_combined.to_excel(writer, sheet_name='combined')
        table_means.to_excel(writer, sheet_name='means')
        table_stds.to_excel(writer, sheet_name='stds')

    # df_summary = df.describe().T[['mean', 'std', 'min', 'max']]
    # path = get_path_data_root().joinpath('tables')
    # path.mkdir(exist_ok=True)
    # df_summary.to_excel(path.joinpath('demographics_table.xlsx'))


def make_keb_table():
    path_keb = get_path_data_root().joinpath("keb.xlsx")
    df_keb = pd.read_excel(path_keb)
    df_keb.drop(columns=['period'], inplace=True)
    df_mean = df_keb.groupby('shoe_condition').mean(numeric_only=True).T
    df_std = df_keb.groupby('shoe_condition').std(numeric_only=True).T
    df_combined = df_mean.copy()
    for col in df_mean.columns:
        print(col)
        df_combined[col] = df_mean[col].combine(df_std[col],
                                                lambda m, s: f"{m:.2f} ± {s:.2f}")

    path_out = get_path_data_root().joinpath('tables').joinpath('keb_summary.xlsx')
    path_out.parent.mkdir(exist_ok=True)
    df_combined.to_excel(path_out, sheet_name='combined')


if __name__ == '__main__':
    # make_param_table()
    # make_demographics_table()
    make_keb_table()
