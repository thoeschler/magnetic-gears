import dolfin as dlf
import gmsh
import numpy as np
import os
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet, CustomVectorExpression
from source.mesh_tools import generate_xdmf_mesh

def cart_to_sph(v, x_vec):
    r = np.linalg.norm(x_vec)
    rho = np.linalg.norm(x_vec[:-1])
    x, y, z = x_vec
    Q = np.array([[x / r, y / r, z / r], [x * z / r / rho, y * z / r / rho, - rho / r], [- y / rho, x / rho, 0]])
    assert np.allclose(Q.T.dot(Q), np.eye(3))
    if v.ndim == 1:
        return Q.dot(v)
    elif v.ndim == 2:
        return Q.T.dot(v).dot(Q)
    else:
        raise RuntimeError()

def sph_to_cart(v, x_vec):
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

def compute_force_analytically(magnet_1, magnet_2, coordinate_system="cartesian"):
    """Compute force on magnet_2 caused by magnetic field of magnet_1.

    Args:
        magnet_1 (BallMagnet): First magnet.
        magnet_2 (BallMagnet): Second magnet.
        coordinate_system (str, optional): Coordinate system used to represent the
                                           force vector. Defaults to "laboratory".
    """
    assert coordinate_system in ("laboratory", "cartesian_1", "cartesian_2", "spherical_1", "spherical_2")
    factor = 4 / 3 * magnet_1.M0 * magnet_2.M0 * np.pi * \
        (magnet_1.R * magnet_2.R) ** 3
    # distance between magnet midpoints
    r = np.linalg.norm(magnet_1.x_M - magnet_2.x_M)
    rho = np.linalg.norm(magnet_1.x_M[:-1] - magnet_2.x_M[:-1])

    # compute force
    # represent all quantities in the cartesian eigenbasis of magnet 1 
    x_M_2 = magnet_1.Q.T.dot(magnet_2.x_M - magnet_1.x_M)
    cos_theta = x_M_2[2] / r
    sin_theta = rho / r
    M_2 = magnet_1.Q.T.dot(magnet_2.M)
    Grad_B = np.array([
        [-2., sin_theta, 0.],
        [- sin_theta, cos_theta, 0.],
        [0., 0., cos_theta]
    ])
    Grad_B_eigen_1 = sph_to_cart(Grad_B, x_M_2)
    force = 1 / r**4 * factor * M_2.dot(Grad_B_eigen_1)

    if coordinate_system == "laboratory":
        force = magnet_1.Q.dot(force)
    elif coordinate_system == "cartesian_1":
        pass
    elif coordinate_system == "cartesian_2":
        force = magnet_2.Q.T.dot(magnet_1.Q.dot(force))
    else:
        raise RuntimeError()
    return force
    

def create_mesh(magnet_1, magnet_2):
    # create mesh
    gmsh.initialize()
    model = gmsh.model()
    magnet = model.occ.addSphere(*magnet_1.x_M, magnet_1.R)
    model.occ.synchronize()
    magnet_boundary = model.getBoundary([(3, magnet)], oriented=False)[0][1]
    magnet_boundary_phy = model.addPhysicalGroup(2, [magnet_boundary])
    magnet_phy = model.addPhysicalGroup(3, [magnet])
    model.mesh.setSize(model.occ.getEntities(0), magnet_2.R / 10)  # global mesh size
    model.mesh.generate(3)
    gmsh.write("mesh.msh")
    generate_xdmf_mesh("mesh.msh", delete_source_files=True)
    gmsh.finalize()
    mesh_file = dlf.XDMFFile("mesh.xdmf")
    mesh = dlf.Mesh()
    mesh_file.read(mesh)
    return mesh

def compute_force_numerically(magnet_1, magnet_2, mesh):
    dA = dlf.Measure("dS", domain=mesh)

    # interpolate B
    B = lambda x: magnet_1.B(x)
    B_expr = CustomVectorExpression(B)
    V = dlf.VectorFunctionSpace(mesh, "CG", 1)
    B_interpol = dlf.interpolate(B_expr, V)

    # compute force
    M = dlf.as_vector(magnet_2.M)  # magnetization
    n = dlf.FacetNormal(mesh)
    t = dlf.cross(dlf.cross(n('+'), M), B_interpol)  # traction vector
    F = np.zeros(3)
    for i, c in enumerate(t):
        F[i] = dlf.assemble(c * dA)
    return F


if __name__ == "__main__":
    os.chdir(os.getcwd())
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
    mag_1 = BallMagnet(radius=r1, magnetization_strength=M_0_1, \
    position_vector=x_M_1, rotation_matrix=Q1)
    # distance between magnets
    d = 1.
    # second magnet
    r2 = 2.
    M_0_2 = 1.
    x_M_2 = x_M_1 + np.array([0., r1 + r2 + d, 0.])
    angle_2 = 0.
    Q2 = Rotation.from_rotvec(angle_2 * np.array([1., 0., 0.])).as_matrix()
    mag_2 = BallMagnet(radius=r2, magnetization_strength=M_0_2, \
        position_vector=x_M_2, rotation_matrix=np.eye(3))

    # get magnet force for different angles
    n_iterations = 15
    mesh = create_mesh(mag_1, mag_2)
    for n in range(n_iterations):
        angle_2 = 2 * np.pi * n / n_iterations
        Q = Rotation.from_rotvec(angle_2 * np.array([1., 0., 0.])).as_matrix()
        mag_2.update_parameters(mag_2.x_M, Q)
        f_ana = compute_force_analytically(mag_1, mag_2, "laboratory")
        f_num = compute_force_numerically(mag_1, mag_2, mesh)
        with open("forceAna.csv", "a+") as f:
            f.write(f"{angle_2} ")
            for c in f_ana:
                f.write(f"{c} ")
            f.write("\n")
        with open("forceNum.csv", "a+") as f:
            f.write(f"{angle_2} ")
            for c in f_num:
                f.write(f"{c} ")
            f.write("\n")