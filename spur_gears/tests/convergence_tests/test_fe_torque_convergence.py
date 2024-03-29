import numpy as np
import os
from source.magnetic_gear_classes import MagneticBallGear
from spur_gears.tests.convergence_tests.ball_gear_convergence_test_base import convergence_test, SpurGearsConvergenceTest


def create_convergence_test(main_dir=None):
    # create two ball gears
    gear_1 = MagneticBallGear(24, 12., 1., np.zeros(3))
    gear_1.create_magnets(magnetization_strength=1.0)
    gear_2 = MagneticBallGear(16, 8., 1., np.array([0., 1., 0.]))
    gear_2.create_magnets(magnetization_strength=1.0)

    # create coaxial gears problem
    D = gear_1.outer_radius + gear_2.outer_radius + 0.1
    ct = SpurGearsConvergenceTest(gear_1, gear_2, D, main_dir=main_dir)
    return ct

def test_fe_torque_convergence(mesh_size):
    conv_dir = "test_fe_torque_convergence"
    if not os.path.exists(conv_dir):
        os.mkdir(conv_dir)
    os.chdir(conv_dir)
    # for different mesh sizes compare errors
    p_deg = 2
    ma = False

    ct = create_convergence_test()
    errors, names = convergence_test(ct, mesh_size=mesh_size, p_deg=p_deg, \
                                    interpolate="twice", use_Vm=True, \
                                    mesh_all_magnets=ma, D_ref=ct.D, \
                                    analytical_solution=False, R_inf_mult=50)

    print(f"In gear 1 {ct.gear_1.p - len(ct.gear_1.magnets)} out of {ct.gear_1.p} magnets have been deleted.")
    print(f"In gear 2 {ct.gear_2.p - len(ct.gear_2.magnets)} out of {ct.gear_2.p} magnets have been deleted.")

    for error, name in zip(errors, names):
        with open(f"{name}.csv", "a+") as f:
            f.write(f"{mesh_size} {error} \n")

if __name__ == "__main__":
    import sys
    ms = float(sys.argv[1])
    test_fe_torque_convergence(ms)
