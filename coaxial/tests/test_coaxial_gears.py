import numpy as np
from source.magnetic_gear_classes import MagneticGearWithBallMagnets, MagneticGearWithBarMagnets
from coaxial.coaxial_gears import CoaxialGearsWithBallMagnets, CoaxialGearsWithBarMagnets, CoaxialGearsBase
import subprocess
import os


def test_coaxial_gears(coaxial_gears: CoaxialGearsBase, n_iterations=4):
    coaxial_gears.set_gear_meshes(mesh_size_space=1.0, mesh_size_magnets=0.2, write_to_pvd=True, verbose=False)

    d_alpha = 2. * np.pi / coaxial_gears.gear_1.n / n_iterations

    for _ in range(n_iterations):
        B1 = coaxial_gears.interpolate_field_gear(coaxial_gears.gear_1, coaxial_gears.gear_2, "B", "CG", 1, 0.3, 4.0)
        B2 = coaxial_gears.interpolate_field_gear(coaxial_gears.gear_2, coaxial_gears.gear_1, "B", "CG", 1, 0.3, 4.0)

        tau1 = coaxial_gears.compute_torque_on_gear(coaxial_gears.gear_1, B2)
        tau2 = coaxial_gears.compute_torque_on_gear(coaxial_gears.gear_2, B1)

        with open(test_dir + "/test_coaxial_gears.csv", "a+") as f:
            f.write(f"{coaxial_gears.gear_1.angle} {tau1} {tau2}\n")
        coaxial_gears.gear_1.update_parameters(d_alpha)


if __name__ == "__main__":
    # create test directory
    test_dir = os.getcwd() + "/test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    gear_1_ball = MagneticGearWithBallMagnets(n=4, r=1.5, R=10, x_M=np.zeros(3), \
        magnetization_strength=1.0, init_angle=0.)
    gear_2_ball = MagneticGearWithBallMagnets(n=6, r=2.0, R=15, x_M=np.zeros(3), \
        magnetization_strength=1.0, init_angle=0.)
    gear_1_bar = MagneticGearWithBarMagnets(n=4, h=0.5, w=0.5, d=0.5, R=3.0, \
        x_M=np.zeros(3), magnetization_strength=1.0, init_angle=0.)
    gear_2_bar = MagneticGearWithBarMagnets(n=6, h=0.5, w=0.5, d=0.5, R=5.0, \
        x_M=np.zeros(3), magnetization_strength=1.0, init_angle=0.)

    coaxial_ball_gears = CoaxialGearsWithBallMagnets(gear_1_ball, gear_2_ball, D=2.0, main_dir=test_dir)
    coaxial_bar_gears = CoaxialGearsWithBarMagnets(gear_1_bar, gear_2_bar, D=2.0, main_dir=test_dir)
    test_coaxial_gears(coaxial_ball_gears, n_iterations=40)
    test_coaxial_gears(coaxial_bar_gears, n_iterations=40)

    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)
