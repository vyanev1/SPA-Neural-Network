import os
from typing import List

import pandas as pd
import numpy as np
import scipy.io

from angle_finder import get_curvature_and_positional_data, use_only_two_markers

pressure_data_dir = os.path.abspath("./Data/Pressure Data/")


def split_input_output(pressure_data: pd.DataFrame) -> (np.ndarray, np.ndarray):
    input_data = pressure_data[["pressure", "distance_mm"]].values
    output_data = pressure_data.drop(["pressure", "distance_mm"], axis=1).values
    return input_data, output_data


def split_two_halves(np_array: np.ndarray) -> (np.ndarray, np.ndarray):
    flat_list = np.ndarray.flatten(np_array)
    half = len(flat_list)//2
    return flat_list[:half], flat_list[half:]


def get_column_names(df: pd.DataFrame, df_num: int) -> List[str]:
    if df_num == 1:
        chambers = [0, 11] if use_only_two_markers else range(2, 12)
        return ["pressure", "distance_mm"] + [f"chamber {j+1}" for j in chambers]
    else:
        return list(df.columns)


def get_combined_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    curvature_data, positional_data = get_curvature_and_positional_data()
    pressure_curvature_d = []
    pressure_position_d = []
    pressure_force_d = []
    row_num = 0
    for exp_date in os.listdir(pressure_data_dir):
        if not(exp_date.startswith("23-07-21")):
            continue
        exp_date_path = os.path.join(pressure_data_dir, exp_date)
        for exp_num in os.listdir(exp_date_path):
            exp_path = os.path.join(exp_date_path, exp_num)
            for data_file in os.listdir(exp_path):
                data_path = os.path.join(exp_path, data_file)
                mat_contents = scipy.io.matlab.loadmat(data_path)

                pressure_all_timestamps = [item.flat[0] for item in mat_contents["pressure"]]
                force_all_timestamps = [item.flat[0] for item in mat_contents["ForceData"]]

                exp_pressure_curvature_dict = {
                    "pressure": round(sum(pressure_all_timestamps) / len(pressure_all_timestamps)),
                    "distance_mm": int(exp_date.split("_", 1)[-1].replace('mm', '')),
                    **curvature_data.drop("image", axis=1).iloc[row_num].to_dict()
                }
                exp_pressure_position_dict = {
                    "pressure": round(sum(pressure_all_timestamps) / len(pressure_all_timestamps)),
                    "distance_mm": int(exp_date.split("_", 1)[-1].replace('mm', '')),
                    **positional_data.drop("image", axis=1).iloc[row_num].to_dict()
                }
                exp_pressure_force_dict = {
                    "pressure": round(sum(pressure_all_timestamps) / len(pressure_all_timestamps)),
                    "distance_mm": int(exp_date.split("_", 1)[-1].replace('mm', '')),
                    "force": max(0, sum(force_all_timestamps) / len(force_all_timestamps))
                }

                pressure_curvature_d.append(exp_pressure_curvature_dict)
                pressure_position_d.append(exp_pressure_position_dict)
                pressure_force_d.append(exp_pressure_force_dict)

                row_num += 1
    return pd.DataFrame(pressure_curvature_d), pd.DataFrame(pressure_position_d), pd.DataFrame(pressure_force_d)


if __name__ == "__main__":
    pressure_curvature, pressure_position, pressure_force = get_combined_data()

    inputs, positional_data = split_input_output(pressure_position)

    data = []
    for i in range(len(inputs)):
        X_coords, Y_coords = split_two_halves(positional_data[i])
        coords = [(int(x), int(y)) for x, y in list(zip(X_coords, Y_coords))]
        data.append(list(inputs[i]) + coords)

    pressure_position_formatted = pd.DataFrame(data, columns=get_column_names(pressure_position, 1))

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
        print("Actual curvatures:")
        print(pressure_curvature)
        print("Actual positions:")
        print(pressure_position_formatted)
        print("Actual force outputs")
        print(pressure_force)
