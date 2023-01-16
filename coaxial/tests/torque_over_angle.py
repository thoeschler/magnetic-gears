import numpy as np
import os
from coaxial.coaxial_gears import CoaxialGearsWithBallMagnets, CoaxialGearsWithBarMagnets


def compute_torque_over_angle(CoaxialGears, n_it_1, n_it_2):
    angles_1 = np.linspace(0., 2 * np.pi / CoaxialGears.n1, n_it_1, endpoint=False)
    angles_2 = np.linspace(0., 2 * np.pi / CoaxialGears.n2, n_it_2, endpoint=False)

    CoaxialGears.set_gear_meshes(mesh_size_space=1.0, mesh_size_magnets=0.2, write_to_pvd=True, verbose=False)

    open("test_coaxial_gears.csv", "w").close()
    for angle_2 in angles_2:
        # compute angle increment
        d_angle_2 = angle_2 - CoaxialGears.gear_2.angle
        CoaxialGears.gear_2.update_parameters(d_angle_2)

        # reset torque lists
        tau1_vals = []
        tau2_vals = []
        for angle_1 in angles_1:
            # compute angle increment
            d_angle_1 = angle_1 - CoaxialGears.gear_1.angle
            CoaxialGears.gear_1.update_parameters(d_angle_1)
 
            # interpolate magnetic field
            B1 = CoaxialGears.interpolate_field_gear(CoaxialGears.gear_1, CoaxialGears.gear_2.mesh, "B", "CG", 1, 0.3, 4.0)
            # compute torque
            tau2_vals += [CoaxialGears.compute_torque_on_gear(CoaxialGears.gear_2, B1)]

            # interpolate magnetic field
            B2 = CoaxialGears.interpolate_field_gear(CoaxialGears.gear_2, CoaxialGears.gear_1.mesh, "B", "CG", 1, 0.3, 4.0)
            # compute torque
            tau1_vals += [CoaxialGears.compute_torque_on_gear(CoaxialGears.gear_1, B2)]

        with open("test_coaxial_gears.csv", "a+") as f:
            for angle_1, t1, t2 in zip(angles_1, tau1_vals, tau2_vals):
                f.write(f"{angle_1} {angle_2} {t1} {t2}\n")
            f.write("\n")

if __name__ == "__main__":
    if not os.path.exists("testdir"):
        os.mkdir("testdir")
    os.chdir("testdir")


    par_ball = {
        "n1": 16,
        "n2": 12,
        "r1": 1.,
        "r2": 1.,
        "R1": 10.0,
        "R2": 6.0,
        "D": 1.0,
        "x_M_1": np.array([0., 0., 0.]),
        "magnetization_strength_1": 1.,
        "magnetization_strength_2": 1.,
        "init_angle_1": 0.,
        "init_angle_2": 0.
    }

    par_bar = {
        "n1": 4,
        "n2": 6,
        "h1": .5,
        "h2": .5,
        "w1": .5,
        "w2": .5,
        "d1": .5,
        "d2": .5,
        "R1": 3.0,
        "R2": 5.0,
        "D": 1.0,
        "x_M_1": np.array([0., 0., 0.]),
        "magnetization_strength_1": 1.,
        "magnetization_strength_2": 1.,
        "init_angle_1": 0.,
        "init_angle_2": 0.
    }

    CoaxialGears = CoaxialGearsWithBallMagnets(**par_ball)
    compute_torque_over_angle(CoaxialGears, 4, 3)