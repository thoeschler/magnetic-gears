import numpy as np
import os
from source.magnetic_gear_classes import MagneticBallGear
from spur_gears.tests.convergence_tests.ball_gear_convergence_test_base import convergence_test, SpurGearsConvergenceTest


def create_convergence_test(main_dir=None):
    # create two ball gears
    gear_1 = MagneticBallGear(14, 6., 1., np.zeros(3))
    gear_1.create_magnets(magnetization_strength=1.0)
    gear_2 = MagneticBallGear(14, 6., 1., np.array([0., 1., 0.]))
    gear_2.create_magnets(magnetization_strength=1.0)

    # create coaxial gears problem
    D = gear_1.outer_radius + gear_2.outer_radius + 0.1
    ct = SpurGearsConvergenceTest(gear_1, gear_2, D, main_dir=main_dir)
    return ct

def version_test():
    conv_dir = "version_test"
    if not os.path.exists(conv_dir):
        os.mkdir(conv_dir)
    os.chdir(conv_dir)
    mesh_sizes = np.geomspace(1e-1, 1.0, num=6)
    for p_deg in (1, 2):
        for interpolate in ("never", "once", "twice"):
            for use_Vm in (True, False):
                for mesh_size in mesh_sizes[::-1]:
                    dirname = f"interpolate_{interpolate}_pdeg_{p_deg}_useVm_{use_Vm}"
                    if not os.path.exists(dirname):
                        os.mkdir(dirname)
                    ct = create_convergence_test(dirname)
                    errors, names = convergence_test(ct, mesh_size=mesh_size, p_deg=p_deg, \
                                                    interpolate=interpolate, use_Vm=use_Vm, \
                                                    D_ref=ct.D, mesh_all_magnets=False, \
                                                    analytical_solution=True)
                    print(f"In gear 1 {ct.gear_1.p - len(ct.gear_1.magnets)} out of {ct.gear_1.p} magnets have been deleted.")
                    print(f"In gear 2 {ct.gear_2.p - len(ct.gear_2.magnets)} out of {ct.gear_2.p} magnets have been deleted.")
                    for error, name in zip(errors, names):
                        with open(f"{dirname}/{name}.csv", "a+") as f:
                            f.write(f"{mesh_size} {error} \n")

if __name__ == "__main__":
    version_test()