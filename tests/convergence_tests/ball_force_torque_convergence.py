import dolfin as dlf
import gmsh
import numpy as np
import os
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet
from source.tools.mesh_tools import generate_mesh_with_markers
from tests.convergence_tests.ball_force_torque import compute_force_analytically, \
    compute_force_analytically_special, compute_torque_analytically, \
        compute_torque_analytically_special, compute_force_numerically, \
        compute_torque_numerically


def create_single_magnet_mesh(magnet, mesh_size, verbose=True):
    """Create finite element mesh of a single ball magnet inside a box.

    Args:
        magnet (BallMagnet): The spherical magnet.
        mesh_size (float): The global mesh size.
        verbose (bool, optional): If True print gmsh info.
                                  Defaults to True.

    Returns:
        tuple: A tuple containing mesh, cell_marker, facet_marker,
               magnet_volume_tag, magnet_boundary_tag.
    """
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    # file name
    fname = "mesh"
    model = gmsh.model()

    # create magnet
    magnet_tag = model.occ.addSphere(*magnet.x_M, magnet.R)

    # cut magnet from box
    #model.occ.cut([(3, box_tag)], [(3, magnet_tag)], removeObject=True, removeTool=False)
    model.occ.synchronize()

    # add physical groups
    magnet_boundary = model.getBoundary([(3, magnet_tag)], oriented=False)[0][1]
    magnet_boundary_tag = model.addPhysicalGroup(2, [magnet_boundary])
    magnet_volume_tag = model.addPhysicalGroup(3, [magnet_tag])

    # set mesh size
    model.mesh.setSize(model.occ.getEntities(0), mesh_size)

    # generate mesh
    model.mesh.generate(dim=3)
    gmsh.write(f"{fname}.msh")

    # create mesh and markers
    mesh, cell_marker, facet_marker = generate_mesh_with_markers("mesh", delete_source_files=False)
    dlf.File(fname + "_mesh.pvd") << mesh
    dlf.File(fname + "_cell_marker.pvd") << cell_marker
    dlf.File(fname + "_facet_marker.pvd") << facet_marker
    gmsh.finalize()

    return mesh, cell_marker, facet_marker, magnet_volume_tag, magnet_boundary_tag

def interpolate_magnetic_field(magnet, mesh, degree=1):
    """Interpolate magnetic field on a given mesh.

    Args:
        magnet (BallMagnet): The magnet whose magnetic field should be used.
        mesh (dlf.Mesh): Finite element mesh.
        degree (int, optional): Polynomial degree. Defaults to 1.

    Returns:
        dlf.Function: The interpolated field.
    """
    B = magnet.B_as_expression()
    V = dlf.VectorFunctionSpace(mesh, "CG", degree)
    B_interpol = dlf.interpolate(B, V)
    return B_interpol

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
    mesh, cell_marker, facet_marker, magnet_volume_tag, magnet_boundary_tag = create_single_magnet_mesh(mag_2, mesh_size, verbose=False)
    print("Interpolate magnetic field...", end="")
    B = interpolate_magnetic_field(mag_1, mesh)
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
        f_num += [compute_force_numerically(mag_2, mesh, B, facet_marker, magnet_boundary_tag)]

        # compute torque
        f = compute_force_analytically(mag_1, mag_2, coordinate_system="laboratory")
        print("ANA", f)
        tau_ana += [compute_torque_analytically(mag_1, mag_2, coordinate_system="laboratory")]
        tau_ana_special += [compute_torque_analytically_special(mag_1, mag_2, angle, coordinate_system="laboratory")]
        tau_num += [compute_torque_numerically(mag_2, mesh, B, facet_marker, magnet_boundary_tag)]

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

def convergence_test(distance_values, mesh_size_values, degree=1):
    # create magnets
    mag_1 = BallMagnet(1.5, 10., np.zeros(3), np.eye(3))
    mag_2 = BallMagnet(2., 20., np.zeros(3), Rotation.from_rotvec(np.pi/8 * np.array([1., 0., 0.])).as_matrix())

    with open("force_error.csv", "w") as f:
        f.write(f"{0.0} ")
        for d in distance_values:
            f.write(f"{d} ")
        f.write("\n")

    with open("torque_error.csv", "w") as f:
        f.write(f"{0.0} ")
        for d in distance_values:
            f.write(f"{d} ")
        f.write("\n")

    for ms in mesh_size_values[::-1]:
        # get magnet force for different angles
        mesh, cell_marker, facet_marker, magnet_volume_tag, magnet_boundary_tag = create_single_magnet_mesh(mag_2, mesh_size=ms, verbose=False)

        force_error = []
        torque_error = []
        for d in distance_values:
            ########## 1. Numerical solution ##########
            ##### Move second magnet to random point in space at distance d
            # create random unit vector
            vec = np.random.rand(3)
            vec /= np.linalg.norm(vec)
            assert np.isclose(vec.dot(vec), 1.0)

            # move magnet and mesh
            # new center point (distance d from first magnet)
            x_M = mag_1.x_M + (d + mag_1.R + mag_2.R) * vec
            # move mesh
            mesh.translate(dlf.Point(*(x_M - mag_2.x_M)))
            # update magnet's center point
            mag_2.x_M = x_M
            assert np.isclose(np.linalg.norm(mag_2.x_M - mag_1.x_M), d + mag_1.R + mag_2.R)

            #### Get magnetic field
            B = interpolate_magnetic_field(mag_1, mesh, degree)
            F_num = compute_force_numerically(mag_2, mesh, B, facet_marker, magnet_boundary_tag)
            tau_num = compute_torque_numerically(mag_2, mesh, B, facet_marker, magnet_boundary_tag, degree)

            ########## 2. Analytical solution ##########
            F_ana = compute_force_analytically(mag_1, mag_2, coordinate_system="laboratory")
            print("ANA", F_ana)
            print(F_ana / F_num)
            tau_ana = compute_torque_analytically(mag_1, mag_2, coordinate_system="laboratory")

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


if __name__ == "__main__":
    # create directory
    if not os.path.exists("test_dir"):
        os.mkdir("test_dir")
    os.chdir("test_dir")

    compute_force_and_torque(n_iterations=5, mesh_size=0.5)

    # some variables
    distance_values = np.array([0.1, 0.5, 1., 5.])
    mesh_size_values = np.linspace(1e-1, 5e-1, num=4)
    convergence_test(distance_values, mesh_size_values, degree=1)
