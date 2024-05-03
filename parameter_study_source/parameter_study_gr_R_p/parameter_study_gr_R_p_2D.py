from source.magnetic_gear_classes import SegmentGear
from spur_gears.spur_gears_problem import SpurGearsProblem
from source.cylinder_segment_2D import force_torque_cylinder_segment_2D

import os
import itertools as it
import numpy as np
import pandas as pd


def sample(spur_gears: SpurGearsProblem, data_dir=".", n_iterations=20):
    assert isinstance(spur_gears.gear_1, SegmentGear)
    assert isinstance(spur_gears.gear_2, SegmentGear)
    angle_1_lim = [- np.pi / spur_gears.gear_1.p, np.pi / spur_gears.gear_1.p]
    angle_2_lim = [- np.pi / spur_gears.gear_2.p, np.pi / spur_gears.gear_2.p]
    angles_1 = np.linspace(*angle_1_lim, num=n_iterations, endpoint=True)
    angles_2 = np.linspace(*angle_2_lim, num=n_iterations, endpoint=True)

    tau_12_values = np.zeros((n_iterations, n_iterations))
    tau_21_values = np.zeros((n_iterations, n_iterations))
    for i, angle_1 in enumerate(angles_1):
        for j, angle_2 in enumerate(angles_2):
            # compute torque
            f_12, tau_12 = force_torque_cylinder_segment_2D(w2=spur_gears.gear_2.width, \
                                                t1=spur_gears.gear_1.t, \
                                                t2=spur_gears.gear_2.t, \
                                                p1=spur_gears.gear_1.p, \
                                                p2=spur_gears.gear_2.p, \
                                                Rm1=spur_gears.gear_1.R, \
                                                Rm2=spur_gears.gear_2.R, \
                                                phi_1=angle_1, \
                                                phi_2=angle_2, \
                                                D=spur_gears.D, \
                                                nb_terms=80)

            # save torque values
            f_12_padded = np.array([0., f_12[0], f_12[1]])
            tau_21 = np.cross(spur_gears.gear_1.x_M - spur_gears.gear_2.x_M, f_12_padded)[0] - tau_12
            tau_12_values[i, j] = tau_12
            tau_21_values[i, j] = tau_21

            # update csv files
            pd.DataFrame(tau_12_values, index=angles_1, columns=angles_2).to_csv(os.path.join(data_dir, "tau_12.csv"))
            pd.DataFrame(tau_21_values, index=angles_1, columns=angles_2).to_csv(os.path.join(data_dir, "tau_21.csv"))

def sample_cylinder_segment(par_number, n_iterations=20):
    t1 = 1.0
    R1 = 10.
    d_ref = 0.1
    w_ref = 3.
    # create sample directory
    sample_dir = "sample_cylinder_segment_gear_2D"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # parameters
    gear_ratio_values = np.array([1.0, 1.4, 2.0, 2.4])
    R_ref_values = np.array([6., 10., 20.])
    scaling_factors = R1 / R_ref_values

    w1_values = w_ref * scaling_factors
    p1_values = list(range(6, 72, 2))
    par_list = list(it.product(gear_ratio_values, w1_values, p1_values))
    par = par_list[par_number]
    gear_ratio, w1, p1 = par
    w2 = w1

    # compute rest of values
    R2 = R1 * gear_ratio
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        return
    if int(p2) % 2 != 0:
        return
    p2 = int(p2)
    # scale d accordingly
    d = d_ref * w1 / w_ref
    # compute t2
    t2 = R2 / R1 * t1

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    width_dir = f"w1_{str(w1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = os.path.join(gear_ratio_dir, width_dir, pole_nb_dir)
    target_dir = os.path.join(sample_dir, data_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    width_dir = f"w1_{str(w1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = os.path.join(gear_ratio_dir, width_dir, pole_nb_dir)
    target_dir = os.path.join(sample_dir, data_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    gear_1 = SegmentGear(p1, R1 - t1 / 2, w1, t1, np.zeros(3))
    gear_2 = SegmentGear(p2, R2 - t2 / 2, w2, t2, np.array([0., 1., 0.]))

    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    D = gear_1.outer_radius + gear_2.outer_radius + d

    spur_gears = SpurGearsProblem(gear_1, gear_2, D)

    sample(spur_gears, data_dir=target_dir, n_iterations=n_iterations)


if __name__ == "__main__":
    for i in range(500):
        sample_cylinder_segment(i, n_iterations=5)
