import numpy as np
import os
from source.magnetic_gear_classes import MagneticBallGear
from spur_gears.spur_gears_problem import SpurGearsProblem
from tests.convergence_tests.ball_magnets_force_torque import compute_force_ana
from spur_gears.tests.convergence_tests.ball_gear_convergence_test_base import compute_torque_ana

p1 = 14
p2 = 14
R1 = 5
R2 = 5
d = 0.5
r1 = R1 * np.sin(np.pi / p1)
r2 = R2 * np.sin(np.pi / p2)

D = R1 + r1 + R2 + r2 + d

R_in_dir = "R_in_convergence"
if not os.path.exists(R_in_dir):
    os.mkdir(R_in_dir)

for D_ref in np.linspace(D - max(R1, R2), D + R1, num=20):
    gear_1 = MagneticBallGear(p1, R1, r1, np.zeros(3))
    gear_2 = MagneticBallGear(p2, R2, r2, np.array([0., 1., 0.]))
    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    all_magnets_1 = list(gear_1.magnets)
    all_magnets_2 = list(gear_2.magnets)
    D = gear_1.outer_radius + gear_2.outer_radius + d
    spur_gears = SpurGearsProblem(gear_1, gear_2, D)
    spur_gears.align_gears()

    # randomly rotate gears by angle pm pi / p
    d_angle_1 = (2 * np.random.rand(1).item() - 1) * np.pi / spur_gears.gear_1.n
    d_angle_2 = (2 * np.random.rand(1).item() - 1) * np.pi / spur_gears.gear_2.n

    spur_gears.remove_magnets(spur_gears.gear_1, D_ref=D_ref)
    spur_gears.remove_magnets(spur_gears.gear_2, D_ref=D_ref)

    print(f"""INFO
        {spur_gears.gear_1.n - len(spur_gears.gear_1.magnets)} out of {spur_gears.gear_1.n} magnets have been removed in gear 1 \n
        {spur_gears.gear_2.n - len(spur_gears.gear_2.magnets)} out of {spur_gears.gear_2.n} have been removed in gear 2 \n
    """)

    spur_gears.gear_1.update_magnets(all_magnets_1, d_angle_1, np.zeros(3))
    spur_gears.gear_2.update_magnets(all_magnets_2, d_angle_2, np.zeros(3))

    # compute force and torque using all magnets
    f_12_all = np.zeros(3)
    f_21_all = np.zeros(3)
    tau_12_all = np.zeros(3)
    tau_21_all = np.zeros(3)
    for mag_1 in all_magnets_1:
        for mag_2 in all_magnets_2:
            # force
            f_21_mag = compute_force_ana(mag_2, mag_1)
            f_12_mag = compute_force_ana(mag_1, mag_2)
            assert np.allclose(f_12_mag, -f_21_mag)
            f_12_all += f_12_mag
            f_21_all += f_21_mag

            # torque
            tau_12_all += compute_torque_ana(mag_1, mag_2, f_12_mag, gear_2.x_M)
            tau_21_all += compute_torque_ana(mag_1, mag_2, f_21_mag, gear_2.x_M)

    # use magnet selection
    f_12_sel = np.zeros(3)
    f_21_sel = np.zeros(3)
    tau_12_sel = np.zeros(3)
    tau_21_sel = np.zeros(3)
    for mag_1 in gear_1.magnets:
        for mag_2 in gear_2.magnets:
            # force
            f_21_mag = compute_force_ana(mag_2, mag_1)
            f_12_mag = compute_force_ana(mag_1, mag_2)
            assert np.allclose(f_12_mag, -f_21_mag)
            f_12_sel += f_12_mag
            f_21_sel += f_21_mag

            # torque
            tau_12_sel += compute_torque_ana(mag_1, mag_2, f_12_mag, gear_2.x_M)
            tau_21_sel += compute_torque_ana(mag_1, mag_2, f_21_mag, gear_2.x_M)

    # compute errors
    e_tau_12 = np.linalg.norm(tau_12_sel - tau_12_all) / np.linalg.norm(tau_12_all)
    e_tau_21 = np.linalg.norm(tau_21_sel - tau_21_all) / np.linalg.norm(tau_21_all)
    e_f_12 = np.linalg.norm(f_12_sel - f_12_all) / np.linalg.norm(f_12_all)
    e_f_21 = np.linalg.norm(f_21_sel - f_21_all) / np.linalg.norm(f_21_all)

    names = ("f_12", "f_21", "tau_12", "tau_21")
    errors = (e_tau_12, e_tau_21, e_f_12, e_f_21)

    print(errors)

    for e, name in zip(errors, names):
        with open(f"{R_in_dir}/{name}.csv", "a+") as f:
            removed_magnets = (spur_gears.gear_1.n - len(spur_gears.gear_1.magnets)) / spur_gears.gear_1.n
            f.write(f"{D_ref} {removed_magnets} {e} \n")
