import dolfin as dlf
import gmsh
import numpy as np
import os
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet
from source.mesh_tools import generate_mesh_with_markers

def cart_to_sph(v, x_vec):
    """Transform tensor components from cartesian to
    spherical basis.


    Args:
        v (np.ndarray): Tensor of rank 1 (vector) or 2.
        x_vec (bp.ndarray): Position vector in cartesian
                            coordinates and basis.

    Returns:
        np.ndarray: Tensor component w.r.t spherical basis.
    """
    r = np.linalg.norm(x_vec)
    rho = np.linalg.norm(x_vec[:-1])
    x, y, z = x_vec
    # transformation matrix
    Q = np.array([
        [x / r, y / r, z / r],
        [x * z / r / rho, y * z / r / rho, - rho / r],
        [- y / rho, x / rho, 0]
        ])
    assert np.allclose(Q.T.dot(Q), np.eye(3))
    if v.ndim == 1:
        return Q.dot(v)
    elif v.ndim == 2:
        return Q.T.dot(v).dot(Q)
    else:
        raise RuntimeError()

def sph_to_cart(v, x_vec):
    """Transform tensor components from spherical to
    cartesian basis.
    

    Args:
        v (np.ndarray): Tensor of rank 1 (vector) or 2.
        x_vec (bp.ndarray): Position vector in cartesian
                            coordinates and basis.

    Returns:
        np.ndarray: Tensor component w.r.t cartesian basis.
    """
    r = np.linalg.norm(x_vec)
    rho = np.linalg.norm(x_vec[:-1])
    x, y, z = x_vec
    Q = np.array([
        [x / r, y / r, z / r],
        [x * z / r / rho, y * z / r / rho, - rho / r],
        [- y / rho, x / rho, 0]
    ])
    assert np.allclose(Q.T.dot(Q), np.eye(3))
    if v.ndim == 1:
        return Q.dot(v)
    elif v.ndim == 2:
        return Q.T.dot(v).dot(Q)
    else:
        raise RuntimeError()
    
def create_mesh(magnet, mesh_size=None):
    # set mesh size
    if mesh_size is None:
        mesh_size = magnet.R / 5.
    # file name
    fname = "mesh"

    gmsh.initialize()

    model = gmsh.model()
    # create surrounding box
    pad = magnet.R / 4
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

def interpolate_magnetic_field(magnet, mesh):
    B = magnet.B_as_expression()
    V = dlf.VectorFunctionSpace(mesh, "CG", 1)
    B_interpol = dlf.interpolate(B, V)
    return B_interpol

def compute_force_analytically(magnet_1, magnet_2, coordinate_system="laboratory"):
    """Compute force on magnet_2 caused by magnetic field of magnet_1.

    Args:
        magnet_1 (BallMagnet): First magnet.
        magnet_2 (BallMagnet): Second magnet.
        coordinate_system (str, optional): Coordinate system used to represent the
                                           force vector. Defaults to "laboratory".
    """
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2")
    factor = 4 / 3 * magnet_1.M0 * magnet_2.M0 * np.pi * \
        (magnet_1.R * magnet_2.R) ** 3

    # some quantities
    r = np.linalg.norm(magnet_1.x_M - magnet_2.x_M)
    rho = np.linalg.norm(magnet_1.x_M[:-1] - magnet_2.x_M[:-1])

    # compute force
    # all quantities are represented in the cartesian eigenbasis of magnet 1 
    x_M_2 = magnet_1.Q.T.dot(magnet_2.x_M - magnet_1.x_M)
    cos_theta = x_M_2[2] / r
    sin_theta = rho / r
    M_2 = magnet_1.Q.T.dot(magnet_2.M)
    # gradient of B in spherical basis
    grad_B_sph = np.array([
        [- 2. * cos_theta, - sin_theta, 0.],
        [- sin_theta, cos_theta, 0.],
        [0., 0., cos_theta]
    ])
    grad_B_cart = sph_to_cart(grad_B_sph, x_M_2)
    force = factor / r ** 4 * M_2.dot(grad_B_cart)

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
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2")

    r = np.linalg.norm(magnet_1.x_M - magnet_2.x_M)
    force = 4. / 3. * magnet_1.M0 * magnet_2.M0 * np.pi * (magnet_1.R * magnet_2.R) ** 3 * \
        r ** (-4) * np.array([0., np.cos(angle), - np.sin(angle)])

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
    dA = dlf.Measure('dS', domain=mesh, subdomain_data=facet_marker)

    # compute force
    M = dlf.as_vector(magnet.M)  # magnetization
    n = dlf.FacetNormal(mesh)
    t = dlf.cross(dlf.cross(n('+'), M), B)  # traction vector
    F = np.ones(3)
    for i, c in enumerate(t):
        a = dlf.assemble(c * dA(magnet_boundary_tag))
        F[i] = a
    return F

def compute_torque_analytically(magnet_1, magnet_2, force, coordinate_system="laboratory"):
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2")

    # all quantities are represented in the cartesian eigenbasis of magnet 1 
    x_M_2 = magnet_1.Q.T.dot(magnet_2.x_M - magnet_1.x_M)
    M_2 = magnet_1.Q.T.dot(magnet_2.M)

    # compute torque 
    tau = np.zeros(3)
    # first term
    B = magnet_1.B_eigen_plus(x_M_2)
    vol_magnet_2 = 4. / 3. * np.pi * magnet_2.R ** 3
    tau += vol_magnet_2 * np.cross(M_2, B)

    # second term
    tau += np.cross(x_M_2, force)

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
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2")

    r = np.linalg.norm(magnet_1.x_M - magnet_2.x_M)
    tau = - 8. / 9. * magnet_1.M0 * magnet_2.M0 * np.pi * (magnet_1.R * magnet_2.R) ** 3 * \
        r ** (- 3) * np.array([np.sin(angle), 0., 0.])

    # return force w.r.t. chosen basis
    if coordinate_system == "laboratory":
        return magnet_1.Q.dot(tau)
    elif coordinate_system == "cartesian_1":
        return tau
    elif coordinate_system == "cartesian_2":
        return magnet_2.Q.T.dot(magnet_1.Q.dot(tau))
    else:
        raise RuntimeError() 

def compute_torque_numerically(magnet_1, magnet_2, mesh, B, facet_marker, magnet_boundary_tag):
    dA = dlf.Measure('dS', domain=mesh, subdomain_data=facet_marker)
    n = dlf.FacetNormal(mesh)

    x = dlf.Expression(("x[0]", "x[1]", "x[2]"), degree=1)
    x_M = dlf.as_vector(magnet_1.x_M)
    M = dlf.as_vector(magnet_2.M)  # magnetization
    t = dlf.cross(dlf.cross(n('+'), M), B)  # traction vector
    m = dlf.cross(x - x_M, t)  # torque density

    tau = np.empty(3)
    for i, c in enumerate(m):
        tau[i] = dlf.assemble(c * dA(magnet_boundary_tag))
    return tau

def compute_force_and_torque(n_iterations):
    # create directory
    if not os.path.exists("test_dir"):
        os.mkdir("test_dir")
    os.chdir("test_dir")

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
    mesh, cell_marker, facet_marker, magnet_volume_tag, magnet_boundary_tag = create_mesh(mag_2)
    B = interpolate_magnetic_field(mag_1, mesh)

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


if __name__ == "__main__":
    compute_force_and_torque(n_iterations=20)