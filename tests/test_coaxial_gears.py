import numpy as np
from source.coaxial_gears import CoaxialGearsWithBallMagnets, CoaxialGearsWithBarMagnets


def test_coaxial_gears(par, gear_cls):
    CoaxialGears = gear_cls(**par)
    CoaxialGears.create_gear_meshes(mesh_size_space=1.0, mesh_size_magnets=0.2, write_to_file=True, verbose=False)

    n_iterations = 19
    d_alpha = 2. * np.pi / par_ball["n1"] / n_iterations
 
    for _ in range(n_iterations):
        B2 = CoaxialGears.interpolate_field_gear(CoaxialGears.gear_2, CoaxialGears.gear_1.mesh, "B", "CG", 1, 0.3, 4.0)
        B1 = CoaxialGears.interpolate_field_gear(CoaxialGears.gear_1, CoaxialGears.gear_2.mesh, "B", "CG", 1, 0.3, 4.0)

        tau1 = CoaxialGears.compute_torque_on_gear(CoaxialGears.gear_1, B2)
        tau2 = CoaxialGears.compute_torque_on_gear(CoaxialGears.gear_2, B1)

        with open("data/test_coaxial_gears.csv", "a+") as f:
            f.write(f"{CoaxialGears.angle_1} {tau1} {tau2}\n")
        CoaxialGears.update_parameters(d_alpha, 0)


if __name__ == "__main__":
    par_ball = {"n1": 12,
            "n2": 16,
            "r1": 1.,
            "r2": 1.5,
            "R1": 8.0,
            "R2": 12.0,
            "D": 1.0,
            "x_M_1": np.array([0., 0., 0.]),
            "magnetization_strength_1": 1.,
            "magnetization_strength_2": 1.,
            "init_angle_1": 0.,
            "init_angle_2": 0.
            }
    test_coaxial_gears(par_ball, CoaxialGearsWithBallMagnets)
