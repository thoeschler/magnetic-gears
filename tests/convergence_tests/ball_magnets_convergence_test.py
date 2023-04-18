import dolfin as dlf
from dolfin import LagrangeInterpolator
import numpy as np
import os
from source.magnet_classes import BallMagnet
from source.tools.math_tools import get_rot
from source.tools.mesh_tools import read_mesh
from source.tools.tools import interpolate_field, create_reference_mesh
from source.tools.fenics_tools import compute_current_potential, rotate_vector_field
from tests.convergence_tests.ball_magnets_grid_generator import create_single_magnet_mesh
from tests.convergence_tests.ball_magnets_force_torque import compute_force_ana, \
      compute_torque_ana, compute_force_num, compute_torque_num


def convergence_test(distance_values, mesh_size_values, p_deg=1, interpolation=False, use_Vm=False, directory=None):
    """
    Convergence test for force and torque between two magnets.

    Args:
        distance_values (np.ndarray): Distance values between magnets.
        mesh_size_values (np.ndarray): _description_
        p_deg (int, optional): Polynomial degree for numerical computation. Defaults to 1.
        interpolation (bool, optional): Whether or not to use intermediate interpolation
                                        to a reference mesh. Defaults to False.
        use_Vm (bool, optional): Wheter or not to use the magnetic potential. If False use
                                 the magnetic field B directly. Defaults to False.
        directory (str, optional): The working directory. Defaults to None.
    """
    # change to directory
    if directory is not None:
        if not os.path.exists(directory):
            os.mkdir(directory)
        os.chdir(directory)

    # initialize csv files
    for fname in ("force_error.csv", "torque_error.csv"):
        with open(fname, "w") as f:
            f.write(f"{0.0} ")
            for d in distance_values:
                f.write(f"{d} ")
            f.write("\n")

    for ms in mesh_size_values[::-1]:
        # create magnets
        angle_1 = np.random.rand(1).item()
        angle_2 = np.random.rand(1).item()
        mag_1 = BallMagnet(1., 1., np.random.rand(3), get_rot(angle_1))
        mag_2 = BallMagnet(1., 1., np.random.rand(3), get_rot(angle_2))

        # mesh second magnet
        mesh_mag_2 = create_single_magnet_mesh(mag_2, mesh_size=ms, verbose=False)

        force_errors = []
        torque_errors = []

        if interpolation:
            # first evaluate the field (Vm or B) on reference mesh
            ref_magnet = BallMagnet(1., 1., np.zeros(3), np.eye(3))
            D_max = distance_values.max() + mag_1.R + mag_2.R
            create_reference_mesh(ref_magnet, domain_radius=(D_max + mag_2.R + 1e-1) / mag_1.R,
                                  mesh_size_min=ms / mag_1.R, mesh_size_max=ms / mag_1.R,
                                    shape="sphere", fname="reference_mesh")
            reference_mesh = read_mesh("reference_mesh.xdmf")
            if use_Vm:
                # scale Vm of reference magnet by radius of mag_1
                Vm_ref = interpolate_field(ref_magnet.Vm, reference_mesh, "CG", p_deg, scale=mag_1.R)
            else:
                B_ref = interpolate_field(ref_magnet.B, reference_mesh, "CG", p_deg, fname="B", write_pvd=True)
                rotate_vector_field(B_ref, get_rot(angle_1))

            # position reference mesh according to mag_1
            reference_mesh.scale(mag_1.R)
            reference_mesh.translate(dlf.Point(*mag_1.x_M))
            reference_mesh.rotate(180. / np.pi * angle_1, 0, dlf.Point(*mag_1.x_M))

        # compute force and torque for all distance values
        for d in distance_values:
            ########################################
            ######## Numerical computation #########
            ########################################

            # randomly move mag_1 (and reference mesh)
            d_x_M_1 = np.random.rand(3)  # random translation
            d_angle_1 = np.random.rand(1).item()
            Q = get_rot(d_angle_1)  # random rotation
            mag_1.update_parameters(mag_1.x_M + d_x_M_1, Q.dot(np.array(mag_1.Q)))
            if interpolation:
                # translate reference mesh first
                reference_mesh.translate(dlf.Point(d_x_M_1))
                # then rotate reference mesh
                reference_mesh.rotate(180. / np.pi * d_angle_1, 0, dlf.Point(mag_1.x_M))
                if not use_Vm:
                    rotate_vector_field(B_ref, Q)

            # move mag_2 to a point at distance d from mag_1 (and the mesh)
            vec = np.random.rand(3)
            vec /= np.linalg.norm(vec)
            assert np.isclose(vec.dot(vec), 1.0)
            D = mag_1.R + mag_2.R + d
            x_M = mag_1.x_M + D * vec  # random translation
            mesh_mag_2.translate(dlf.Point(*(x_M - mag_2.x_M)))
            d_angle_2 = np.random.rand(1).item()  # random rotation
            mag_2.update_parameters(x_M, get_rot(d_angle_2).dot(np.array(mag_2.Q)))
            assert np.isclose(np.linalg.norm(mag_2.x_M - mag_1.x_M), D)
            # no need to rotate the mesh of mag_2 because it is spherical

            if interpolation:
                if use_Vm:
                    # interpolate Vm_ref to second magnet
                    V = dlf.FunctionSpace(mesh_mag_2, "CG", p_deg)
                    Vm = dlf.Function(V)
                    LagrangeInterpolator.interpolate(Vm, Vm_ref)
                    # compute gradient and project
                    B = compute_current_potential(Vm, project=False)
                else:
                    # interpolate B_ref to second magnet
                    V = dlf.VectorFunctionSpace(mesh_mag_2, "CG", p_deg)
                    B = dlf.Function(V)
                    LagrangeInterpolator.interpolate(B, B_ref)
            else:
                if use_Vm:
                    Vm = interpolate_field(mag_1.Vm, mesh_mag_2, "CG", p_deg)
                    # compute gradient and project
                    B = compute_current_potential(Vm, project=False)
                else:
                    B = interpolate_field(mag_1.B, mesh_mag_2, "CG", p_deg)

            # perform numerical computation
            F_num = compute_force_num(mag_2, B, mesh_mag_2)
            tau_num = compute_torque_num(mag_2, B, mesh_mag_2)

            ########################################
            ######## Analytical computation ########
            ########################################

            F_ana = compute_force_ana(mag_1, mag_2, coordinate_system="laboratory")
            tau_ana = compute_torque_ana(mag_1, mag_2, coordinate_system="laboratory")

            # compute numerical error
            e_F = np.linalg.norm(F_ana - F_num) / np.linalg.norm(F_ana)
            e_tau = np.linalg.norm(tau_ana - tau_num) / np.linalg.norm(tau_ana)
            force_errors.append(e_F)
            torque_errors.append(e_tau)

            print(f"""INFO: \n
                mesh size = {ms:.2f} \n
                distance = {d:.2f} \n
                force error = {e_F:.6f} \n
                torque error = {e_tau:.6f} \n
            """)

        with open("force_error.csv", "a+") as f:
            f.write(f"{ms} ")
            for fe in force_errors:
                f.write(f"{fe} ")
            f.write("\n")

        with open("torque_error.csv", "a+") as f:
            f.write(f"{ms} ")
            for te in torque_errors:
                f.write(f"{te} ")
            f.write("\n")
    os.chdir("..")


if __name__ == "__main__":
    # create directory
    conv_dir = "ball_magnets_convergence_test"
    if not os.path.exists(conv_dir):
        os.mkdir(conv_dir)
    os.chdir(conv_dir)

    # perform convergence test
    distance_values = np.array([0.1])
    mesh_size_values = np.geomspace(5e-1, 1.0, num=6)

    for p_deg in (1, 2):
        # 1. no interpolation, use B directly
        convergence_test(distance_values, mesh_size_values, p_deg=p_deg,
                        interpolation=False, use_Vm=False, directory=f"B_no_interpol_pdeg_{p_deg}")
        # 2. no interpolation, use Vm
        convergence_test(distance_values, mesh_size_values, p_deg=p_deg,
                        interpolation=False, use_Vm=True, directory=f"Vm_no_interpol_pdeg_{p_deg}")
        # 3. with interpolation, use B directly
        convergence_test(distance_values, mesh_size_values, p_deg=p_deg,
                        interpolation=True, use_Vm=False, directory=f"B_interpol_pdeg_{p_deg}")
        # 4. with interpolation, use Vm
        convergence_test(distance_values, mesh_size_values[1:], p_deg=p_deg,
                        interpolation=True, use_Vm=True, directory=f"Vm_interpol_pdeg_{p_deg}")
