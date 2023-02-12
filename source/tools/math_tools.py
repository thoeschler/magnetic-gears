import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def get_perpendicular(vec, normalized=True):
    # create random unit vector different from vec
    while True:
        rand = np.random.rand(3)
        if not np.allclose(rand, vec):
            break
    # create vector perpendicular to vec
    perp_vec = np.cross(vec, rand)
    if normalized:
        perp_vec /= np.linalg.norm(perp_vec)
    assert np.isclose(np.dot(vec, perp_vec), 0.)
    return perp_vec

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