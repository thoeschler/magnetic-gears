import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation


def get_rot(angle, axis=0):
    return Rotation.from_rotvec(angle * np.eye(3)[axis]).as_matrix()

def get_interpolater(data, x, y):
    assert data.ndim == 2
    interp = RegularGridInterpolator((x, y), data)
    return interp

def is_between(val, min, max):
    if (val >= min) and (val <= max):
        return True
    return False

def interpolate(interpol_func, coords):
    assert len(coords) == 2
    x, y = coords
    assert isinstance(interpol_func, RegularGridInterpolator)
    x_values = interpol_func.grid[0]
    y_values = interpol_func.grid[1]

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

    return interpol_func((x, y))