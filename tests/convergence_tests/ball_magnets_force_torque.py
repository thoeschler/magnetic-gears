import dolfin as dlf
import numpy as np
from source.magnet_classes import BallMagnet
from source.transform import sph_to_cart


####################################################
############## Analytical computation ##############
####################################################

def compute_force_ana(magnet_1: BallMagnet, magnet_2: BallMagnet, coordinate_system="laboratory"):
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
    factor = 4. / 3. * np.pi * (magnet_1.R * magnet_2.R) ** 3

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

def compute_torque_ana(magnet_1, magnet_2, coordinate_system="laboratory"):
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

    # return force w.r.t. chosen basis
    if coordinate_system == "laboratory":
        return magnet_1.Q.dot(tau)
    elif coordinate_system == "cartesian_1":
        return tau
    elif coordinate_system == "cartesian_2":
        return magnet_2.Q.T.dot(magnet_1.Q.dot(tau))
    else:
        raise RuntimeError()

####################################################
############## Numerical computation ###############
####################################################

def compute_force_num(magnet, B, mesh=None):
    """Compute force on magnet caused by magnetic field.

    Args:
        magnet (BallMagnet): The magnet.
        B (dlf.Function): The magnetic field.
        mesh (dlf.Mesh): Finite element mesh. Only needs to be specified
                         if B is not a dlf.Function.

    Returns:
        np.ndarray: Force in specified coordinate system.
    """
    if isinstance(B, dlf.Function):
        mesh = B.function_space().mesh()
    else:
        assert mesh is not None 

    dA = dlf.Measure('ds', domain=mesh)

    # compute force
    M_jump = dlf.as_vector(- magnet.M)  # jump of magnetization
    n = dlf.FacetNormal(mesh)
    t = dlf.cross(dlf.cross(n, M_jump), B)  # traction vector
    F = np.zeros(3)
    for i, c in enumerate(t):
        a = dlf.assemble(c * dA)
        F[i] = a
    return F

def compute_torque_num(magnet, B, mesh):
    """Compute torque on magnet caused by magnetic field.

    Args:
        magnet (BallMagnet): The magnet the torque acts on.
        B (dlf.Function): The interpolated magnetic field on the respective mesh.
        mesh (dlf.Mesh): Finite element mesh. Only needs to be specified
                         if B is not a dlf.Function.

    Returns:
        np.ndarray: Force in specified coordinate system.
    """
    if isinstance(B, dlf.Function):
        mesh = B.function_space().mesh()
    else:
        assert mesh is not None 

    dA = dlf.Measure('ds', domain=mesh)
    n = dlf.FacetNormal(mesh)

    x = dlf.SpatialCoordinate(mesh)
    x_M = dlf.as_vector(magnet.x_M)
    M_jump = dlf.as_vector(- magnet.M)  # jump of magnetization
    t = dlf.cross(dlf.cross(n, M_jump), B)  # traction vector
    m = dlf.cross(x - x_M, t)  # torque density

    tau = np.zeros(3)
    for i, c in enumerate(m):
        tau[i] = dlf.assemble(c * dA)
    return tau
