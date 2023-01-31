import sympy as sy
import numpy as np


def gradient_spherical_coordinates_sympy(field: sy.Matrix):
    """Right gradient of vector field in spherical coordinates.

    The vector field is given in both spherical coordinates (r, theta, phi)
    and its physical basis.

    Args:
        field (sy.Matrix): The gradient of the vector field in both spherical
                           coordinates and basis.
    """
    r, theta, phi = sy.symbols("r vartheta varphi", positive=True)
    assert isinstance(field, sy.Matrix)
    assert len(field) == 3

    # gradient in spherical coordinates
    # (see https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates)
    Ar, Atheta, Aphi = field
    grad = sy.Matrix([
        [Ar.diff(r), 1 / r * Ar.diff(theta) - Atheta / r, 1 / r / sy.sin(theta) * Ar.diff(phi) - Aphi / r],
        [Atheta.diff(r), 1 / r * Atheta.diff(theta) + Ar / r, 1 / r / sy.sin(theta) * Atheta.diff(phi) - sy.cot(theta) * Aphi / r],
        [Aphi.diff(r), 1 / r * Aphi.diff(theta), 1 / r / sy.sin(theta) * Aphi.diff(phi) + sy.cot(theta) * Atheta / r + Ar / r]
        ])
    return grad

def sph_to_cart_matrix(cs="sph"):
    """Transformation matrix between spherical and cartesian basis.

    Can be given in spherical or cartesian coordinates.

    Args:
        cs (str, optional): Coordinate system. Defaults to "sph".
    """
    assert cs in ("sph", "cart")
    if cs=="sph":
        r, theta, phi = sy.symbols("r vartheta varphi", positive=True)
        return sy.Matrix([
            [sy.sin(theta) * sy.cos(phi), sy.sin(theta) * sy.sin(phi), sy.cos(theta)],
            [sy.cos(theta) * sy.cos(phi), sy.cos(theta) * sy.sin(phi), - sy.sin(theta)],
            [- sy.sin(phi), sy.cos(phi), 0]
        ])
    elif cs=="cart":
        x, y, z = sy.symbols("x y z")
        r = sy.sqrt(x ** 2 + y ** 2 + z ** 2)
        rho = sy.sqrt(x ** 2 + y ** 2)
        return sy.Matrix([
            [x / r, y / r, z / r],
            [x * z / r / rho, y * z / r / rho, - rho / r],
            [- y / rho, x / rho, 0]
        ])
    else:
        raise RuntimeError() 

def sph_to_cart_sympy(v_sph: sy.Matrix, cs="sph"):
    """Transform tensor components v from spherical to cartesian basis.

    Coordinates can be given in terms of spherical or cartesian coordinates.

    Args:
        v_sph (sy.Matrix): Tensor components.
        cs (str, optional): Coordinate system the components are given in. Can
                            be either "sph" or "cart". Defaults to "sph".
    """
    assert cs in ("sph", "cart")
    Q = sph_to_cart_matrix(cs)
    dim = (np.array(v_sph.shape) > 1).sum()
    if dim == 1:
        v_cart = Q.T * v_sph
    elif dim == 2:
        v_cart = Q.T * v_sph * Q
    else:
        raise RuntimeError()

    v_cart.simplify()
    return v_cart

def cart_to_sph_sympy(v_cart: sy.Matrix, cs="cart"):
    """Transform tensor components v from cartesian to spherical basis.

    Coordinates can be given in terms of spherical or cartesian coordinates.

    Args:
        v_cart (sy.Matrix): Tensor components.
        cs (str, optional): Coordinate system the components are given in. Can
                            be either "sph" or "cart". Defaults to "sph".
    """
    assert cs in ("sph", "cart")
    Q = sph_to_cart_matrix(cs).T
    dim = (np.array(v_sph.shape) > 1).sum()
    if dim == 1:
        v_sph = Q.T * v_cart
    elif dim == 2:
        v_sph = Q.T * v_cart * Q
    else:
        raise RuntimeError()

    v_sph.simplify()
    return v_sph

if __name__ == "__main__":
    theta, phi = sy.symbols("vartheta varphi", positive=True)
    mat_sph = sy.Matrix([
        [-2 * sy.cos(theta), - sy.sin(theta), 0],
        [- sy.sin(theta), sy.cos(theta), 0],
        [0, 0, sy.cos(theta)]
    ])
    mat_cart = sph_to_cart_sympy(mat_sph, cs="sph")
    mat_cart = mat_cart.subs({theta: np.pi / 2, phi: np.pi / 2})
    print(np.array(mat_cart, dtype=float))