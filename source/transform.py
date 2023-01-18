import numpy as np


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