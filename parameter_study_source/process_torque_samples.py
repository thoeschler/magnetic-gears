import numpy as np
import pandas as pd
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
        if not(os.path.isdir(os.path.join(dir_name, subdir))):
            continue
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

def process_parameter_study_gr_R_p():
    gear_ratio_names = ["gear_ratio_1p0", "gear_ratio_1p4", "gear_ratio_2p0", "gear_ratio_2p4"]
    w1_names = ["w1_1p5", "w1_3p0", "w1_5p0"]
    # In the computation the lengths are scaled differently. In order to get equal values for the distance
    # d and the width as well as different radii R we need to rescale the torque values by a length factor
    # to the power of three
    R_ref = 10.0
    R_values = np.array([20., 10., 6.])
    length_factors = R_values / R_ref

    for gear_ratio in gear_ratio_names:
        for w1, length_factor in zip(w1_names, length_factors):
            # set directory names
            source_dir_name = os.path.join("/home/thilo/Dropbox/magnetic gears/python/parameter_study_gr_R_p_cylinder_segment_2D", gear_ratio, w1)
            target_dir_name = os.path.join(source_dir_name)

            # load torque and pole number values
            p_values, tau_12_values, tau_21_values = read_torque_values(source_dir_name)
            tau_12_values *= length_factor ** 3
            tau_21_values *= length_factor ** 3

            # write files
            if not os.path.exists(target_dir_name):
                os.makedirs(target_dir_name)
            write_file(p_values, tau_12_values, os.path.join(target_dir_name, "tau_12.csv"))
            write_file(p_values, tau_21_values, os.path.join(target_dir_name, "tau_21.csv"))

def process_parameter_study_a_p():
    d_names = ["d_0p05", "d_0p1", "d_0p2", "d_0p4"]

    for d in d_names:
        # set directory names
        source_dir_name = os.path.join("/home/thilo/Dropbox/magnetic gears/python/parameter_study_a_p", d)
        target_dir_name = os.path.join(source_dir_name)

        # load torque and pole number values
        p_values, tau_12_values, tau_21_values = read_torque_values(source_dir_name)

        # write files
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)
        write_file(p_values, tau_12_values, os.path.join(target_dir_name, "tau_12.csv"))
        write_file(p_values, tau_21_values, os.path.join(target_dir_name, "tau_21.csv"))

def compare_torques():
    gear_ratio_names = ["gear_ratio_1p0", "gear_ratio_1p4", "gear_ratio_2p0", "gear_ratio_2p4"]
    w1_names = ["w1_1p5", "w1_3p0", "w1_5p0"]
    source_dir_name = os.path.join("/home/thilo/Dropbox/magnetic gears/python/parameter_study_gr_R_p_cylinder_segment")
    for gear_ratio in gear_ratio_names:
        for w1 in w1_names:
            # set directory names
            target_dir_name = os.path.join(source_dir_name, gear_ratio, w1)
            tau_12 = pd.read_csv(os.path.join(target_dir_name, "tau_12.csv"), sep='\\s+', index_col=0).to_numpy(dtype=float)
            tau_21 = pd.read_csv(os.path.join(target_dir_name, "tau_21.csv"), sep='\\s+', index_col=0).to_numpy(dtype=float)
            print(tau_12 / tau_21)

def optimal_magnet_numbers():
    gear_ratio_names = ["gear_ratio_1p0", "gear_ratio_1p4", "gear_ratio_2p0", "gear_ratio_2p4"]
    w1_names = ["w1_1p5", "w1_3p0", "w1_5p0"]
    source_dir_name = os.path.join("/home/thilo/Dropbox/magnetic gears/python/parameter_study_gr_R_p_cylinder_segment")

    p_opt = np.empty((len(gear_ratio_names), len(w1_names)), dtype=int)
    for row, gear_ratio in enumerate(gear_ratio_names):
        for col, w1 in enumerate(w1_names):
            # set directory names
            target_dir_name = os.path.join(source_dir_name, gear_ratio, w1)
            tau_12 = pd.read_csv(os.path.join(target_dir_name, "tau_12.csv"), sep='\\s+').to_numpy(dtype=float)
            p_opt[row, col] = tau_12[tau_12[:, 1].argmax(), 0]
    pd.DataFrame(p_opt, dtype=float, index=[1.0, 1.4, 2.0, 2.4], columns=None).to_csv(
        os.path.join(source_dir_name, "p_opt.csv"), columns=None, header=None, sep=" ")

if __name__ == "__main__":
    process_parameter_study_a_p()
    process_parameter_study_gr_R_p()
    compare_torques()
    optimal_magnet_numbers()
