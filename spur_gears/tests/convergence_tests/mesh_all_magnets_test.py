import numpy as np
import os
from spur_gears.tests.convergence_tests.ball_gear_convergence_test_base import convergence_test, SpurGearsConvergenceTest
from source.magnetic_gear_classes import MagneticBallGear


def create_convergence_test(main_dir=None):
    # create two ball gears
    gear_1 = MagneticBallGear(24, 12., 1., np.zeros(3))
    gear_1.create_magnets(magnetization_strength=1.0)
    gear_2 = MagneticBallGear(16, 8., 2. / 3., np.array([0., 1., 0.]))
    gear_2.create_magnets(magnetization_strength=1.0)

    # create coaxial gears problem
    D = gear_1.outer_radius + gear_2.outer_radius + 1.0
    ct = SpurGearsConvergenceTest(gear_1, gear_2, D, main_dir=main_dir)
    return ct

def reference_distance_test():
    conv_dir = "reference_distance_test"
    if not os.path.exists(conv_dir):
        os.mkdir(conv_dir)
    os.chdir(conv_dir)
    mesh_size = 0.15
    p_deg = 2
    ct = create_convergence_test()
    D_values = np.linspace(ct.D, ct.domain_size, num=10)

    for D in D_values:
        # 1. interpolate never, use Vm
        #main(mesh_sizes, p_deg, interpolate="never", use_Vm=True, mesh_all_magnets=ma, \
        #     D_ref=D, dir=f"Vm_interpol_never_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)
        # 2. interpolate once, use Vm
        #main(mesh_sizes, p_deg, interpolate="once", use_Vm=True, mesh_all_magnets=ma, \
        #     D_ref=D, dir=f"Vm_interpol_once_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)
        # 3. interpolate twice, use Vm

        ct = create_convergence_test()
        errors, names = convergence_test(ct, mesh_size=mesh_size, p_deg=p_deg, \
                                         interpolate="twice", use_Vm=True, \
                                            mesh_all_magnets=False, D_ref=D, \
                                                analytical_solution=True)

        # 4. interpolate never, use B directly
        #main(mesh_sizes, p_deg, interpolate="never", use_Vm=False, mesh_all_magnets=ma, \
        #     D_ref=D, dir=f"B_interpol_never_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)
        # 5. interpolate once, use B directly
        #main(mesh_sizes, p_deg, interpolate="once", use_Vm=False, mesh_all_magnets=ma, \
        #     D_ref=D, dir=f"B_interpol_once_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)
        # 6. interpolate twice, use B directly
        #main(mesh_sizes[1:], p_deg, interpolate="twice", use_Vm=False, mesh_all_magnets=ma, \
        #     D_ref=D, dir=f"B_interpol_twice_pdeg_{p_deg}_{mesh_str}", analytical_solution=True)

        print(f"In gear 1 {ct.gear_1.n - len(ct.gear_1.magnets)} out of {ct.gear_1.n} magnets have been deleted.")
        print(f"In gear 2 {ct.gear_2.n - len(ct.gear_2.magnets)} out of {ct.gear_2.n} magnets have been deleted.")

        for error, name in zip(errors, names):
            with open(f"{name}.csv", "a+") as f:
                f.write(f"{mesh_size} {error} {D} {ct.gear_1.R} {ct.gear_2.R} {D - ct.gear_1.R - ct.gear_2.R} \n")
    os.chdir("..")

def mesh_all_magnets_test():
    conv_dir = "mesh_all_magnets_test"
    if not os.path.exists(conv_dir):
        os.mkdir(conv_dir)
    os.chdir(conv_dir)
    mesh_sizes = np.geomspace(1e-1, 1.0, num=6)
    p_deg = 2
    ct = create_convergence_test()

    for ma in (False, True):
        # 3. interpolate twice, use Vm
        dir = "all" if ma else "not_all"
        print(dir)
        if not os.path.exists(dir):
            os.mkdir(dir)
        ct = create_convergence_test(main_dir=dir)
        for mesh_size in mesh_sizes[::-1]:
            errors, names = convergence_test(ct, mesh_size=mesh_size, p_deg=p_deg, \
                                            interpolate="twice", use_Vm=True, \
                                            mesh_all_magnets=ma, D_ref=ct.D, \
                                            analytical_solution=True)

            print(f"In gear 1 {ct.gear_1.n - len(ct.gear_1.magnets)} out of {ct.gear_1.n} magnets have been deleted.")
            print(f"In gear 2 {ct.gear_2.n - len(ct.gear_2.magnets)} out of {ct.gear_2.n} magnets have been deleted.")

            for error, name in zip(errors, names):
                with open(f"{dir}/{name}.csv", "a+") as f:
                    f.write(f"{mesh_size} {error} \n")
        os.chdir("..")


if __name__ == "__main__":
    mesh_all_magnets_test()