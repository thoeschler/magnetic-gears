import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation


def get_rot(angle, axis=0):
    """
    Get rotation matrix.

    Args:
        angle (float): Angle.
        axis (int, optional): Rotation axis. Defaults to 0.

    Returns:
        np.ndarray: Rotation matrix.
    """
    return Rotation.from_rotvec(angle * np.eye(3)[axis]).as_matrix()

def get_interpolater(data, x, y):
    """
    Get interpolator object from two dimensional data.

    Args:
        data (np.ndarray): Two dimensional data set.
        x (np.ndarray): The x values.
        y (np.ndarray): The y values.

    Returns:
        RegularGridInterpolator: Interpolator object.
    """
    assert data.ndim == 2
    interp = RegularGridInterpolator((x, y), data)
    return interp

def is_between(val, min, max):
    if (val >= min) and (val <= max):
        return True
    return False

def periodic_interpolation(interpolator: RegularGridInterpolator, val):
    """
    Evaluate interpolator function assuming periodicity.

    If the input value is outside the interpolators own grid
    assume periodicity of the interpolation function to compute
    the output value.

    Args:
        interpolator (RegularGridInterpolator): _description_
        val (np.ndarray): Value to evaluate the interpolation function for.

    Returns:
        float: Interpolated value.
    """
    assert len(val) == 2
    x, y = val
    assert isinstance(interpolator, RegularGridInterpolator)
    x_values = interpolator.grid[0]
    y_values = interpolator.grid[1]

    x_period = x_values.max() - x_values.min()
    y_period = y_values.max() - y_values.min()

    if not is_between(x, x_values.min(), x_values.max()):
        if x > x_values.max():
            xc = float(x)
            x = xc - (abs(xc - x_values.max()) // x_period + 1) * x_period
        else:
            xc = float(x)
            x = xc + (abs(xc - x_values.min()) // x_period + 1) * x_period
    if not is_between(y, y_values.min(), y_values.max()):
        if y > y_values.max():
            yc = float(y)
            y = yc - (abs(yc - y_values.max()) // y_period + 1) * y_period
        else:
            yc = float(y)
            y = yc + (abs(yc - y_values.min()) // y_period + 1) * y_period

    return interpolator((x, y))