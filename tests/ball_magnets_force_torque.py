import dolfin as dlf
import gmsh
import numpy as np
import os
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet
from source.tools.mesh_tools import generate_mesh_with_markers
from source.transform import sph_to_cart


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

    # create surrounding box
    pad = magnet.R / 4  # padding value
    A = magnet.x_M - (pad + magnet.R) * np.ones(3)
    B = 2 * (pad + magnet.R) * np.ones(3)
    box_tag = model.occ.addBox(*A, *B)

    # create magnet
    magnet_tag = model.occ.addSphere(*magnet.x_M, magnet.R)

    # cut magnet from box
    model.occ.cut([(3, box_tag)], [(3, magnet_tag)], removeObject=True, removeTool=False)
    model.occ.synchronize()

    # add physical groups
    magnet_boundary = model.getBoundary([(3, magnet_tag)], oriented=False)[0][1]
    magnet_boundary_tag = model.addPhysicalGroup(2, [magnet_boundary])
    box_volume_tag = model.addPhysicalGroup(3, [box_tag])
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

def compute_force_analytically(magnet_1: BallMagnet, magnet_2: BallMagnet, coordinate_system="laboratory"):
    """Compute force on magnet_2 caused by magnetic field of magnet_1.

    Args:
        magnet_1 (BallMagnet): First magnet.
        magnet_2 (BallMagnet): Second magnet.
        coordinate_system (str, optional): Coordinate system used to represent the
                                           force vector. Defaults to "laboratory".
    
    Returns:
        np.ndarray: The force in the specified coordinated system.
    """
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2")
    factor = 4. / 3. * magnet_1.M0 * magnet_2.M0 * np.pi * \
        (magnet_1.R * magnet_2.R) ** 3

    # some quantities
    r = np.linalg.norm(magnet_1.x_M - magnet_2.x_M)
    # position vector in eigen coordinates of magnet_1
    x_M_2 = magnet_1.Q.T.dot(magnet_2.x_M - magnet_1.x_M)
    x, y, z = x_M_2
    rho = np.sqrt(x ** 2 + y ** 2)

    # compute force
    # all quantities are represented in the cartesian eigenbasis of magnet 1 
    cos_theta = z / r
    sin_theta = rho / r
    M_2 = magnet_1.Q.T.dot(magnet_2.M)
    # gradient of B in spherical basis
    gB_sph = np.array([
        [-2 * cos_theta, - sin_theta, 0.],
        [- sin_theta, cos_theta, 0.],
        [0., 0., cos_theta]
        ])
    gB_cart = sph_to_cart(gB_sph, x_M_2, "cart")

    force = factor / r ** 4 * M_2.dot(gB_cart)

    # return force w.r.t. chosen basis
    if coordinate_system == "laboratory":
        return magnet_1.Q.dot(force)
    elif coordinate_system == "cartesian_1":
        return force
    elif coordinate_system == "cartesian_2":
        return magnet_2.Q.T.dot(magnet_1.Q.dot(force))
    else:
        raise RuntimeError()

def compute_force_analytically_special(magnet_1, magnet_2, angle, coordinate_system="laboratory"):
    """Compute force on magnet_2 caused by magnetic field of magnet_1 for a special case.
    The special choice is that of theta=pi/2 and phi=pi/2 (azimuthal and polar angles).
    In other words, the two magnets are aligned in y-direction. Both x- and z-coordinates
    of the centers of mass are the same for both magents.

    Args:
        magnet_1 (BallMagnet): First magnet.
        magnet_2 (BallMagnet): Second magnet.
        angle (float): The relative angle between the magnets around the x-axis.
        coordinate_system (str, optional): Coordinate system used to represent the
                                           force vector. Defaults to "laboratory".
    
    Returns:
        np.ndarray: The force in the specified coordinated system.
    """
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2")

    r = np.linalg.norm(magnet_1.x_M - magnet_2.x_M)
    force = 4. / 3. * magnet_1.M0 * magnet_2.M0 * np.pi * (magnet_1.R * magnet_2.R) ** 3 / \
        r ** 4 * np.array([0., np.cos(angle), - np.sin(angle)])

    # return force w.r.t. chosen basis
    if coordinate_system == "laboratory":
        return magnet_1.Q.dot(force)
    elif coordinate_system == "cartesian_1":
        return force
    elif coordinate_system == "cartesian_2":
        return magnet_2.Q.T.dot(magnet_1.Q.dot(force))
    else:
        raise RuntimeError() 

def compute_force_numerically(magnet, mesh, B, facet_marker, magnet_boundary_tag):
    """Compute force on magnet caused by magnetic field.

    Args:
        magnet (BallMagnet): The magnet.
        mesh (dlf.Mesh): The finite element mesh of the magnet.
        B (dlf.Function): The magnetic field.
        facet_marker (dolfin.cpp.mesh.MeshFunctionSizet): Facet marker.
        magnet_boundary_tag (int): Tag of magnet boundary.

    Returns:
        np.ndarray: Force in specified coordinate system.
    """
    dA = dlf.Measure('dS', domain=mesh, subdomain_data=facet_marker)

    # compute force
    M_jump = dlf.as_vector(- magnet.M)  # jump of magnetization
    n = dlf.FacetNormal(mesh)
    t = dlf.cross(dlf.cross(n('-'), M_jump), B)  # traction vector
    F = np.ones(3)
    for i, c in enumerate(t):
        a = dlf.assemble(c * dA(magnet_boundary_tag))
        F[i] = a
    return F

def compute_torque_analytically(magnet_1, magnet_2, force, coordinate_system="laboratory"):
    """Compute torque on magnet_2 caused by magnetic field of magnet_1.

    Args:
        magnet_1 (BallMagnet): First magnet.
        magnet_2 (BallMagnet): Second magnet.
        force (np.ndarray): The force on magnet_2 caused by magnet_1 given in the
                            laboratory system.
        coordinate_system (str, optional): Coordinate system used to represent the
                                           force vector. Defaults to "laboratory".

    Returns:
        np.ndarray: The torque in the specified coordinated system.
    """
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2")

    # all quantities are represented in the cartesian eigenbasis of magnet 1 
    x_M_2 = magnet_1.Q.T.dot(magnet_2.x_M - magnet_1.x_M)
    M_2 = magnet_1.Q.T.dot(magnet_2.M)

    # compute torque
    tau = np.zeros(3)  # initialize
    # first term
    B = magnet_1.B_eigen_plus(x_M_2)
    vol_magnet_2 = 4. / 3. * np.pi * magnet_2.R ** 3
    tau += vol_magnet_2 * np.cross(M_2, B)

    # force in cartesian eigenbasis of magnet 1
    f = magnet_1.Q.T.dot(force)
    tau += np.cross(x_M_2, f)

    # return force w.r.t. chosen basis
    if coordinate_system == "laboratory":
        return magnet_1.Q.dot(tau)
    elif coordinate_system == "cartesian_1":
        return tau
    elif coordinate_system == "cartesian_2":
        return magnet_2.Q.T.dot(magnet_1.Q.dot(tau))
    else:
        raise RuntimeError()

def compute_torque_analytically_special(magnet_1, magnet_2, angle, coordinate_system="laboratory"):
    """Compute torque on magnet_2 caused by magnetic field of magnet_1 for a special case.
    The special choice is that of theta=pi/2 and phi=pi/2 (azimuthal and polar angles).
    In other words, the two magnets are aligned in y-direction. Both x- and z-coordinates
    of the centers of mass are the same for both magents.

    Args:
        magnet_1 (BallMagnet): First magnet.
        magnet_2 (BallMagnet): Second magnet.
        angle (float): The relative angle between the magnets around the x-axis.
        coordinate_system (str, optional): Coordinate system used to represent the
                                           force vector. Defaults to "laboratory".
    
    Returns:
        np.ndarray: The force in the specified coordinated system.
    """
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2")

    r = np.linalg.norm(magnet_1.x_M - magnet_2.x_M)

    tau = - 8. / 9. * magnet_1.M0 * magnet_2.M0 * np.pi * (magnet_1.R * magnet_2.R) ** 3 / \
        r ** 3 * np.array([np.sin(angle), 0., 0.])

    # return force w.r.t. chosen basis
    if coordinate_system == "laboratory":
        return magnet_1.Q.dot(tau)
    elif coordinate_system == "cartesian_1":
        return tau
    elif coordinate_system == "cartesian_2":
        return magnet_2.Q.T.dot(magnet_1.Q.dot(tau))
    else:
        raise RuntimeError() 

def compute_torque_numerically(magnet_1, magnet_2, mesh, B, facet_marker, magnet_boundary_tag, degree=1):
    """Compute torque on magnet caused by magnetic field.

    Args:
        magnet_1 (BallMagnet): The magnet causing the magnetic field.
        magnet_2 (BallMagnet): The magnet the torque acts on.
        mesh (dlf.Mesh): The finite element mesh of the second magnet.
        B (dlf.Function): The interpolated magnetic field on the respective mesh.
        facet_marker (dolfin.cpp.mesh.MeshFunctionSizet): Facet marker.
        magnet_boundary_tag (int): Tag of magnet boundary.
        degree (int): Polynomial degree of finite element space.

    Returns:
        np.ndarray: Force in specified coordinate system.
    """
    dA = dlf.Measure('dS', domain=mesh, subdomain_data=facet_marker)
    n = dlf.FacetNormal(mesh)

    x = dlf.Expression(("x[0]", "x[1]", "x[2]"), degree=degree)
    x_M = dlf.as_vector(magnet_1.x_M)
    M_jump = dlf.as_vector(- magnet_2.M)  # jump of magnetization
    t = dlf.cross(dlf.cross(n('-'), M_jump), B)  # traction vector
    m = dlf.cross(x - x_M, t)  # torque density

    tau = np.empty(3)
    for i, c in enumerate(m):
        tau[i] = dlf.assemble(c * dA(magnet_boundary_tag))
    return tau

def compute_force_and_torque(n_iterations, mesh_size):
    # create magnets
    # first magnet
    r1 = 1.
    M_0_1 = 1.
    x_M_1 = np.zeros(3)
    angle_1 = 0.
    Q1 = Rotation.from_rotvec(angle_1 * np.array([1., 0., 0.])).as_matrix()
    mag_1 = BallMagnet(r1, M_0_1, x_M_1, Q1)
    # distance between magnets
    d = 1.
    # second magnet
    r2 = 1.
    M_0_2 = 1.
    x_M_2 = x_M_1 + np.array([0., r1 + r2 + d, 0.])
    angle_2 = 0.
    Q2 = Rotation.from_rotvec(angle_2 * np.array([1., 0., 0.])).as_matrix()
    mag_2 = BallMagnet(r2, M_0_2, x_M_2, Q2)

    # get magnet force for different angles
    mesh, cell_marker, facet_marker, magnet_volume_tag, magnet_boundary_tag = create_single_magnet_mesh(mag_2, mesh_size)
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
        tau_ana += [compute_torque_analytically(mag_1, mag_2, f, coordinate_system="laboratory")]
        tau_ana_special += [compute_torque_analytically_special(mag_1, mag_2, angle, coordinate_system="laboratory")]
        tau_num += [compute_torque_numerically(mag_1, mag_2, mesh, B, facet_marker, magnet_boundary_tag)]
    
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
    # first magnet
    r1 = 1.
    M_0_1 = 1.
    x_M_1 = np.random.rand(3)
    angle_1 = np.random.rand(1).item()
    Q1 = Rotation.from_rotvec(angle_1 * np.array([1., 0., 0.])).as_matrix()
    mag_1 = BallMagnet(r1, M_0_1, x_M_1, Q1)
    # distance between magnets
    d = 1.
    # second magnet
    r2 = 1.
    M_0_2 = 1.
    x_M_2 = x_M_1 + np.array([0., r1 + r2 + d, 0.])
    angle_2 = np.random.rand(1).item()
    Q2 = Rotation.from_rotvec(angle_2 * np.array([1., 0., 0.])).as_matrix()
    mag_2 = BallMagnet(r2, M_0_2, x_M_2, Q2)

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
        mesh, cell_marker, facet_marker, magnet_volume_tag, magnet_boundary_tag = create_single_magnet_mesh(mag_2, mesh_size=ms)

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
            mesh.translate(dlf.Point(*((d + mag_1.R + mag_2.R) * vec - (mag_2.x_M - mag_1.x_M))))
            mag_2.x_M = mag_1.x_M + (d + mag_1.R + mag_2.R) * vec
            assert np.isclose(np.linalg.norm(mag_2.x_M - mag_1.x_M), d + mag_1.R + mag_2.R)

            #### Get magnetic field
            B = interpolate_magnetic_field(mag_1, mesh, degree)
            F_num = compute_force_numerically(mag_2, mesh, B, facet_marker, magnet_boundary_tag)
            tau_num = compute_torque_numerically(mag_1, mag_2, mesh, B, facet_marker, magnet_boundary_tag, degree)

            ########## 2. Analytical solution ##########
            F_ana = compute_force_analytically(mag_1, mag_2, coordinate_system="laboratory")
            tau_ana = compute_torque_analytically(mag_1, mag_2, F_ana, coordinate_system="laboratory")

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

    compute_force_and_torque(n_iterations=5, mesh_size=1.0)

    # some variables
    distance_values = np.array([0.1, 0.5, 1., 5.])
    mesh_size_values = np.linspace(1e-1, 5e-1, num=4)
    convergence_test(distance_values, mesh_size_values, degree=1)
