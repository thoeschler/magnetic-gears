import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def read_data(dir_name):
    tau_12 = pd.read_csv(f"{dir_name}/tau_12.csv", index_col=0)
    tau_21 = pd.read_csv(f"{dir_name}/tau_21.csv", index_col=0)
    return tau_12, tau_21

def read_torque_values(dir_name):
    subdirs = os.listdir(dir_name)
    p_values = []
    tau_12_values = []
    tau_21_values = []
    for subdir in subdirs:
        p1 = int(subdir.split("_")[-1])
        p_values.append(p1)
        tau_12, tau_21 = read_data(os.path.join(dir_name, subdir))
        tau_12_max = np.absolute(tau_12.to_numpy(dtype=float)).max()
        tau_21_max = np.absolute(tau_21.to_numpy(dtype=float)).max()
        tau_12_values.append(tau_12_max)
        tau_21_values.append(tau_21_max)

    data_12 = np.array(list(zip(p_values, tau_12_values)))
    data_12_sorted = data_12[data_12[:, 0].argsort()]

    data_21 = np.array(list(zip(p_values, tau_21_values)))
    data_21_sorted = data_21[data_21[:, 0].argsort()]
    p_12, tau_12 = data_12_sorted.T
    p_21, tau_21 = data_21_sorted.T
    assert np.allclose(p_12, p_21)
    return p_12, tau_12, tau_21

def write_file(pole_numbers, torque_values, fname):
    with open(fname, "w") as f:
        for p, tau in zip(pole_numbers, torque_values):
            f.write(f"{p} {tau} \n")

if __name__ == "__main__":
    gear_ratio_names = ["gear_ratio_1p0"]#, "gear_ratio_1p4"]#, "gear_ratio_2p0", "gear_ratio_2p4"]
    #w1_names = ["w1_1p5", "w1_3p0", "w1_5p0"]
    w1_names = ["w1_3p0", "w1_5p0"]

    for gear_ratio in gear_ratio_names:
        for w1 in w1_names:
            # set directory names
            source_dir_name = os.path.join("sample_cylinder_segment_gear", gear_ratio, w1)
            #source_dir_name = os.path.join("../../tex/Data/sample_cylinder_segment_gear_d_0p1", gear_ratio, w1)
            target_dir_name = os.path.join("sample_cylinder", gear_ratio, w1)

            # load torque and pole number values
            p_values, tau_12_values, tau_21_values = read_torque_values(source_dir_name)

            # write files
            if not os.path.exists(target_dir_name):
                os.makedirs(target_dir_name)
            write_file(p_values, tau_12_values, os.path.join(target_dir_name, "tau_12.csv"))
            write_file(p_values, tau_21_values, os.path.join(target_dir_name, "tau_21.csv"))
