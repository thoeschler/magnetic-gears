import numpy as np
from source.tools.math_tools import get_interpolater


if __name__ == "__main__":
    nx = 10
    ny = 20

    x_vals = np.arange(nx)
    y_vals = np.arange(ny)
    data = np.random.rand(nx * ny, 2)
    data_sel = data[:, 0].reshape(nx, ny)
    interp = get_interpolater(data_sel, x_vals, y_vals)

    val = np.random.rand(2)
    print(val, interp(val))