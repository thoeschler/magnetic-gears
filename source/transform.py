import numpy as np


def sph_to_cart_matrix(x_vec, cs="sph"):
    """Transformation matrix between spherical and cartesian basis.

    Args:
        x_vec (np.ndarray): Tuple of cartesian or spherical coordinates.
        cs (str): The coordinate system of x_vec.

    Returns:
        np.ndarray: Transformation matrix.
    """
    assert len(x_vec) == 3
    if cs=="cart":
        x, y, z = x_vec
    elif cs=="sph":
        r, theta, phi = x_vec
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
    else:
        raise RuntimeError()
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rho = np.sqrt(x ** 2 + y ** 2)
    Q = np.array([
        [x / r, y / r, z / r],
        [x * z / r / rho, y * z / r / rho, - rho / r],
        [- y / rho, x / rho, 0]
        ])
    return Q

def cart_to_sph(v, x_vec, cs="cart"):
    """Transform tensor components from cartesian to
    spherical basis.

    Args:
        v (np.ndarray): Tensor of rank 1 (vector) or 2.
        x_vec (np.ndarray): Tuple of cartesian or spherical coordinates.
        cs (str): cs (str): The coordinate system of x_vec.

    Returns:
        np.ndarray: Tensor components w.r.t spherical basis.
    """
    # transformation matrix
    Q = sph_to_cart_matrix(x_vec, cs).T
    assert np.allclose(Q.T.dot(Q), np.eye(3))
    if v.ndim == 1:
        return Q.T.dot(v)
    elif v.ndim == 2:
        return Q.T.dot(v).dot(Q)
    else:
        raise RuntimeError()

def sph_to_cart(v, x_vec, cs="sph"): 
    """Transform tensor components from spherical to
    cartesian basis.

    Args:
        v (np.ndarray): Tensor of rank 1 (vector) or 2.
        x_vec (np.ndarray): Tuple of cartesian or spherical coordinates.
        cs (str): The coordinate system of x_vec.

    Returns:
        np.ndarray: Tensor components w.r.t cartesian basis.
    """
    Q = sph_to_cart_matrix(x_vec, cs)
    assert np.allclose(Q.T.dot(Q), np.eye(3))
    if v.ndim == 1:
        return Q.T.dot(v)
    elif v.ndim == 2:
        return Q.T.dot(v).dot(Q)
    else:
        raise RuntimeError()

if __name__ == "__main__":
    theta = np.pi / 2
    phi = np.pi / 2
    mat_sph = np.array([
        [-2 * np.cos(theta), - np.sin(theta), 0],
        [- np.sin(theta), np.cos(theta), 0],
        [0, 0, np.cos(theta)]
        ])
    mat_cart = sph_to_cart(mat_sph, x_vec=(1, theta, phi), cs="sph")
    print(mat_cart)