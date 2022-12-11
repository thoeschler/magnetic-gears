import numpy as np
from coaxial.coaxial_gears import CoaxialGearsWithBallMagnets, CoaxialGearsWithBarMagnets
import subprocess
import os


def test_coaxial_gears(par, gear_cls):
    # create test directory
    test_dir = os.getcwd() + "/test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    
    CoaxialGears = gear_cls(**par, main_dir=test_dir)
    CoaxialGears.set_gear_meshes(mesh_size_space=1.0, mesh_size_magnets=0.2, write_to_pvd=True, verbose=False)

    n_iterations = 4
    d_alpha = 2. * np.pi / par["n1"] / n_iterations
 
    for _ in range(n_iterations):
        B2 = CoaxialGears.interpolate_field_gear(CoaxialGears.gear_2, CoaxialGears.gear_1.mesh, "B", "CG", 1, 0.3, 4.0)
        B1 = CoaxialGears.interpolate_field_gear(CoaxialGears.gear_1, CoaxialGears.gear_2.mesh, "B", "CG", 1, 0.3, 4.0)

        tau1 = CoaxialGears.compute_torque_on_gear(CoaxialGears.gear_1, B2)
        tau2 = CoaxialGears.compute_torque_on_gear(CoaxialGears.gear_2, B1)

        with open(test_dir + "/test_coaxial_gears.csv", "a+") as f:
            f.write(f"{CoaxialGears.angle_1} {tau1} {tau2}\n")
        CoaxialGears.update_parameters(d_alpha, 0.)
    
    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)



if __name__ == "__main__":
    par_ball = {
        "n1": 4,
        "n2": 6,
        "r1": 1.,
        "r2": 1.5,
        "R1": 6.0,
        "R2": 8.0,
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
    
    test_coaxial_gears(par_ball, CoaxialGearsWithBallMagnets)
    test_coaxial_gears(par_bar, CoaxialGearsWithBarMagnets)
