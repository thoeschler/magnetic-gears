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
    if val >= min and val <= max:
        return True
    return False

def interpolate(interpol_func, coords):
    assert len(coords) == 2
    x, y = coords
    assert isinstance(interpol_func, RegularGridInterpolator)
    x_values = interpol_func.grid[0]
    y_values = interpol_func.grid[1]
    x_period = y_values.size * (x_values[1] - x_values[0])
    y_period = y_values.size * (y_values[1] - y_values[0])

    if is_between(x, x_values.min(), x_values.max()) and \
        is_between(y, y_values.min(), y_values.max()):
        return interpol_func((x, y))
    else:
        # shift x to fit in the range
        shifted_x = x - np.sign(x) * (abs(x) // x_period) * x_period
        shifted_y = y - np.sign(y) * (abs(y) // y_period) * y_period
        return interpol_func((shifted_x, shifted_y))

if __name__ == "__main__":
    torque_data = pd.read_csv("sample/sample_torque_ball.csv", sep="\t").to_numpy()
    n = np.sqrt(torque_data.shape[0])
    phi_1 = torque_data[:, 0].reshape(30, 30)[:, 0]
    phi_2 = torque_data[:, 1].reshape(30, 30)[0]
    torque_1 = torque_data[:, 2].reshape(30, 30)
    torque_2 = torque_data[:, 3].reshape(30, 30)
    t1 = get_interpolater(torque_1, phi_1, phi_2)
    t2 = get_interpolater(torque_2, phi_1, phi_2)

    t = interpolate(t1, (0, 0.3))
    print(t)