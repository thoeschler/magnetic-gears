import dolfin as dlf
from dolfin import LagrangeInterpolator
import numpy as np
import os
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet
from source.tools.mesh_tools import read_mesh
from source.tools.tools import interpolate_field, create_reference_mesh
from source.tools.fenics_tools import compute_magnetic_field
from tests.convergence_tests.grid_generator import create_single_magnet_mesh
from tests.convergence_tests.ball_force_torque import compute_force_analytically, \
    compute_force_analytically_special, compute_torque_analytically, \
        compute_torque_analytically_special, compute_force_numerically, \
        compute_torque_numerically


def compute_force_and_torque(n_iterations, mesh_size):
    # create magnets
    mag_1 = BallMagnet(1., 1., np.zeros(3), np.eye(3))
    mag_2 = BallMagnet(1., 1., np.zeros(3), np.eye(3))
    # distance between magnets
    d = 1.
    # move second magnet according to d
    x_M_2 = mag_1.x_M + np.array([0., mag_1.R + mag_2.R + d, 0.])
    mag_2.x_M = x_M_2

    # get magnet force for different angles
    #mesh, cell_marker, facet_marker, magnet_volume_tag, magnet_boundary_tag = 
    mesh = create_single_magnet_mesh(mag_2, mesh_size, verbose=False)
    print("Interpolate magnetic field...", end="")
    B = interpolate_field(mag_1.B_as_expression(), mesh, "CG", 1)
    print("Done.")

    # compute force
    angles = np.linspace(0, 2 * np.pi, n_iterations + 1, endpoint=True)
    f_ana = []
    f_ana_special = []
    f_num = []
    tau_ana = []
    tau_ana_special = []
    tau_num = []

    for angle in angles:
        # set angle for magnet 2
        Q = Rotation.from_rotvec(angle * np.array([1., 0., 0.])).as_matrix()
        mag_2.update_parameters(mag_2.x_M, Q)

        # compute force
        f_ana += [compute_force_analytically(mag_1, mag_2, coordinate_system="laboratory")]
        f_ana_special += [compute_force_analytically_special(mag_1, mag_2, angle, coordinate_system="laboratory")]
        f_num += [compute_force_numerically(mag_2, mesh, B)]

        # compute torque
        tau_ana += [compute_torque_analytically(mag_1, mag_2, coordinate_system="laboratory")]
        tau_ana_special += [compute_torque_analytically_special(mag_1, mag_2, angle, coordinate_system="laboratory")]
        tau_num += [compute_torque_numerically(mag_2, mesh, B)]

    # write result to text files
    fnames = ("fAna", "fAnaSpecial", "fNum", "tauAna", "tauAnaSpecial", "tauNum")
    for data, fname in zip((f_ana, f_ana_special, f_num, tau_ana, tau_ana_special, tau_num), fnames):
        write_text_file(angles, data, fname)

def write_text_file(angles, forces_or_torques, fname):
    fname = fname.rstrip(".csv")
    with open(f"{fname}.csv", "w") as f:
        for angle, values in zip(angles, forces_or_torques):
            f.write(f"{angle} ")
            for c in values:
                f.write(f"{c} ")
            f.write("\n")

def convergence_test(distance_values, mesh_size_values, p_deg=1, interpolation=False, use_Vm=False, dir=None):
    if dir is not None:
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.chdir(dir)

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
        mag_1 = BallMagnet(1., 1., np.ones(3), Rotation.from_rotvec(angle_1 * np.array([1., 0., 0.])).as_matrix())
        mag_2 = BallMagnet(1., 1., np.zeros(3), Rotation.from_rotvec(angle_2 * np.array([1., 0., 0.])).as_matrix())

        # get magnet force for different angles
        mesh_mag_2 = create_single_magnet_mesh(mag_2, mesh_size=ms, verbose=False)

        force_error = []
        torque_error = []

        if interpolation:
            # first evaluate on reference mesh
            ref_magnet = BallMagnet(1., 1., np.zeros(3), np.eye(3))
            D_max = distance_values.max() + mag_1.R + mag_2.R
            create_reference_mesh(ref_magnet, domain_radius=(D_max + mag_2.R + 1e-1) / mag_1.R, mesh_size_min=ms, mesh_size_max=ms,\
                                    fname="reference_mesh")
            reference_mesh = read_mesh("reference_mesh.xdmf")
            if use_Vm:
                ########################################
                ################ use Vm ################
                ########################################
                Vm_ref = interpolate_field(mag_1.R * ref_magnet.Vm_as_expression(), reference_mesh, "CG", p_deg)
            else:
                ########################################
                ########### use B directly #############
                ########################################
                B_ref = interpolate_field(ref_magnet.B_as_expression(), reference_mesh, "CG", p_deg)

            # position reference mesh in the center of magnet 1
            reference_mesh.scale(mag_1.R)
            reference_mesh.translate(dlf.Point(*mag_1.x_M))
            reference_mesh.rotate(180. / np.pi * angle_1, 0, dlf.Point(*mag_1.x_M))

        for d in distance_values:
            ########################################
            ########## Numerical solution ##########
            ########################################

            # randomly rotate mag_1 (and reference mesh)
            d_angle_1 = np.random.rand(1).item()
            new_Q = (Rotation.from_rotvec(d_angle_1 * np.array([1., 0., 0.])).as_matrix()).dot(np.array(mag_1.Q))
            mag_1.update_parameters(mag_1.x_M, new_Q)
            if interpolation:
                # rotate reference mesh
                reference_mesh.rotate(180. / np.pi * d_angle_1, 0, dlf.Point(mag_1.x_M))

                """# copy reference function
                ref_mesh_copy = dlf.Mesh(reference_mesh)
                if use_Vm:
                    V = dlf.FunctionSpace(ref_mesh_copy, "CG", p_deg)
                    Vm_ref_copy = dlf.Function(V, Vm_ref._cpp_object.vector())
                else:
                    V = dlf.VectorFunctionSpace(ref_mesh_copy, "CG", p_deg)
                    B_ref_copy = dlf.Function(V, B_ref._cpp_object.vector())"""

            # move mag_2 and mesh
            # Move second magnet to random point in space at distance d
            # create random unit vector
            vec = np.random.rand(3)
            vec /= np.linalg.norm(vec)
            assert np.isclose(vec.dot(vec), 1.0)
            # new center point (distance d from first magnet)
            D = mag_1.R + mag_2.R + d
            x_M = mag_1.x_M + D * vec
            # translation
            mesh_mag_2.translate(dlf.Point(*(x_M - mag_2.x_M)))
            # rotation
            d_angle_2 = np.random.rand(1).item()
            new_Q = (Rotation.from_rotvec(d_angle_2 * np.array([1., 0., 0.])).as_matrix()).dot(np.array(mag_2.Q))
            mag_2.update_parameters(x_M, new_Q)
            assert np.isclose(np.linalg.norm(mag_2.x_M - mag_1.x_M), D)
            # no need to rotate the mesh because it is a sphere

            if interpolation:
                if use_Vm:
                    # interpolate Vm_ref on second magnet
                    V = dlf.FunctionSpace(mesh_mag_2, "CG", p_deg)
                    Vm = dlf.Function(V)
                    LagrangeInterpolator.interpolate(Vm, Vm_ref)
                    # compute gradient and project
                    B = compute_magnetic_field(Vm)
                else:
                    # interpolate B_ref on second magnet
                    V = dlf.VectorFunctionSpace(mesh_mag_2, "CG", p_deg)
                    B = dlf.Function(V)
                    LagrangeInterpolator.interpolate(B, B_ref)
            else:
                if use_Vm:
                    Vm = interpolate_field(mag_1.Vm_as_expression(), mesh_mag_2, "CG", p_deg)
                    # compute gradient and project
                    B = compute_magnetic_field(Vm)
                else:
                    B = interpolate_field(mag_1.B_as_expression(), mesh_mag_2, "CG", p_deg)

            F_num = compute_force_numerically(mag_2, mesh_mag_2, B)
            tau_num = compute_torque_numerically(mag_2, mesh_mag_2, B, p_deg)

            ########################################
            ########## Analytical solution #########
            ########################################
            F_ana = compute_force_analytically(mag_1, mag_2, coordinate_system="laboratory")
            tau_ana = compute_torque_analytically(mag_1, mag_2, coordinate_system="laboratory")

            print(f"""INFO \n
                    FANA = {F_ana} \n
                    FNUM = {F_num} \n
                    TAUANA = {tau_ana} \n
                    TAUNUM = {tau_num} \n
            """)
            force_error.append(np.linalg.norm(F_ana - F_num) / np.linalg.norm(F_ana))
            torque_error.append(np.linalg.norm(tau_ana - tau_num) / np.linalg.norm(tau_ana))

        with open("force_error.csv", "a+") as f:
            f.write(f"{ms} ")
            for fe in force_error:
                f.write(f"{fe} ")
            f.write("\n")

        with open("torque_error.csv", "a+") as f:
            f.write(f"{ms} ")
            for te in torque_error:
                f.write(f"{te} ")
            f.write("\n")
    os.chdir("..")


if __name__ == "__main__":
    # create directory
    if not os.path.exists("test_dir"):
        os.mkdir("test_dir")
    os.chdir("test_dir")

    # perform convergence test
    distance_values = np.array([0.1])#0.5, 1., 5.])
    mesh_size_values = np.geomspace(1e-1, 1.0, num=3)

    for p_deg in (1, 2):
        # 1. no interpolation, use B directly
        convergence_test(distance_values, mesh_size_values, p_deg=p_deg, \
                         interpolation=False, use_Vm=False, dir=f"B_no_interpol_pdeg_{p_deg}")
        # 2. no interpolation, use Vm
        convergence_test(distance_values, mesh_size_values, p_deg=p_deg, \
                         interpolation=False, use_Vm=True, dir=f"Vm_no_interpol_pdeg_{p_deg}")
        # 3. with interpolation, use B directly
        convergence_test(distance_values, mesh_size_values, p_deg=p_deg, \
                        interpolation=True, use_Vm=False, dir=f"B_interpol_pdeg_{p_deg}")
        # 4. with interpolation, use Vm
        convergence_test(distance_values, mesh_size_values, p_deg=p_deg, \
                        interpolation=True, use_Vm=True, dir=f"Vm_interpol_pdeg_{p_deg}")
