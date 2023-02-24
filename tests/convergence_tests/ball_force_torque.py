import dolfin as dlf
import numpy as np
from source.magnet_classes import BallMagnet
from source.transform import sph_to_cart


####################################################
############## Analytical computation ##############
####################################################

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
    force = 4. / 3. * np.pi * (magnet_1.R * magnet_2.R) ** 3 / r ** 4 * \
        np.array([0., np.cos(angle), - np.sin(angle)])

    # return force w.r.t. chosen basis
    if coordinate_system == "laboratory":
        return magnet_1.Q.dot(force)
    elif coordinate_system == "cartesian_1":
        return force
    elif coordinate_system == "cartesian_2":
        return magnet_2.Q.T.dot(magnet_1.Q.dot(force))
    else:
        raise RuntimeError()

def compute_torque_analytically(magnet_1, magnet_2, coordinate_system="laboratory"):
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

    tau = 4. / 9. * np.pi * (magnet_1.R * magnet_2.R) ** 3 / \
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

####################################################
############## Numerical computation ###############
####################################################

def compute_force_numerically(magnet, mesh, B):
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
    assert isinstance(B, dlf.Function)
    assert B.function_space().mesh() is mesh
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

def compute_torque_numerically(magnet, mesh, B, degree=1):
    """Compute torque on magnet caused by magnetic field.

    Args:
        magnet (BallMagnet): The magnet the torque acts on.
        mesh (dlf.Mesh): The finite element mesh of the second magnet.
        B (dlf.Function): The interpolated magnetic field on the respective mesh.
        facet_marker (dolfin.cpp.mesh.MeshFunctionSizet): Facet marker.
        magnet_boundary_tag (int): Tag of magnet boundary.
        degree (int): Polynomial degree of finite element space.

    Returns:
        np.ndarray: Force in specified coordinate system.
    """
    dA = dlf.Measure('ds', domain=mesh)
    n = dlf.FacetNormal(mesh)

    x = dlf.Expression(("x[0]", "x[1]", "x[2]"), degree=degree)
    x_M = dlf.as_vector(magnet.x_M)
    M_jump = dlf.as_vector(- magnet.M)  # jump of magnetization
    t = dlf.cross(dlf.cross(n, M_jump), B)  # traction vector
    m = dlf.cross(x - x_M, t)  # torque density

    tau = np.zeros(3)
    for i, c in enumerate(m):
        tau[i] = dlf.assemble(c * dA)
    return tau
