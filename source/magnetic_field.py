import sympy as sy
import numpy as np


def magnetic_potential(Q1, Q2, c1, c2, width, depth, lambdify=True):
    # some symbols
    x, y, z = sy.symbols("x y z", real=True)
    x_vec = sy.Matrix([x, y, z])

    # from here on general
    # eigen coordinates for both surfaces
    x_eigen_1, y_eigen_1, z_eigen_1 = Q1.T * (x_vec - c1)
    x_eigen_2, y_eigen_2, z_eigen_2 = Q2.T * (x_vec - c2)

    # eigen coordinates as sympy matrix
    x_eigen = sy.Matrix([x_eigen_1, x_eigen_2])
    y_eigen = sy.Matrix([y_eigen_1, y_eigen_2])
    z_eigen = sy.Matrix([z_eigen_1, z_eigen_2])

    X = sy.zeros(2, 2)
    Y = sy.zeros(2, 2)
    Z = sy.zeros(2, 1)
    R = [sy.zeros(2, 2), sy.zeros(2, 2)]

    # indices need to be shifted by one compared to analytical expression 
    for k in range(2):
        for l in range(2):
            X[k, l] = x_eigen[k] + (-1) ** (l + 1) * width
            Y[k, l] = y_eigen[k] + (-1) ** (l + 1) * depth
        Z[k] = z_eigen[k]

    for k in range(2):
        for l in range(2):
            for m in range(2):
                R[k][l, m] = sy.sqrt(X[k, l] ** 2 + Y[k, m] ** 2 + Z[k] ** 2)

    # compute magnetic potential
    # split magnetic potential into terms t1, t2 and t3
    t1, t2, t3 = 0., 0., 0.

    for k in range(2):
        for l in range(2):
                for m in range(2):
                    t1 += (-1) ** (k + l + m) * (
                        - Z[k] * sy.atan(X[k, l] * Y[k, m] / Z[k] / R[k][l, m])
                    )
                    t2 += (-1) ** (k + l + m) * (
                        X[k, l] * sy.log(Y[k, m] + R[k][l, m])
                    )
                    t3 += (-1) ** (k + l + m) * (
                        Y[k, m] * sy.log(X[k, l] * R[k][l, m])
                    )
    t1 = t1.simplify()
    t2 = t2.simplify()
    t3 = t3.simplify()

    # add terms to get magnetic potential
    V_m = t1 + t2 + t3

    if lambdify:
        return sy.lambdify([(x, y, z)], V_m)
    else:
        return V_m

def magnetic_potential_bar_magnet(width, depth, height, lambdify=True):
    Q1 = sy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    Q2 = sy.Identity(3)
    c1 = sy.Matrix([0, 0, - height])
    c2 = sy.Matrix([0, 0, height])

    return magnetic_potential(Q1, Q2, c1, c2, width, depth, lambdify=lambdify)


def H_x_bar(w, d, h, x, y, z):
    cx11 = (h + z) ** 2 * ((-d + y) / ((h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * \
        np.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2)) + \
            (d - y) / ((h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2)) - \
                (d + y) / ((h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2)) + \
                    (d + y) / ((h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2))) 
    cx12 = (h - z) ** 2 * ((d - y) / (np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * h * z + z ** 2)) + \
        (d + y) / (np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * h * z + z ** 2)) + \
            (-d + y) / (np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * h * z + z ** 2)) - \
                (d + y) / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * h * z + z ** 2)))
    cx2 = np.log(-d + y + np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - np.log(-d + y + np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
        np.log(d + y + np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) + np.log(d + y + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) - \
            np.log(-d + y + np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2)) + np.log(-d + y + np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2)) + \
                np.log(d + y + np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2)) - np.log(d + y + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2))
    cx3 = -((-w + x) ** 2 / ((-d + y + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) * np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2))) + \
        (w + x) ** 2 / ((-d + y + np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) * np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) + \
            (-w + x) ** 2 / ((d + y + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) * np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) - \
                (w + x) ** 2 / ((d + y + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) * np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) + \
                    (-w + x) ** 2 / (np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2) * (-d + y + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + \
                        (-h + z) ** 2))) - (w + x) ** 2 / (np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2) * (-d + y + np.sqrt((w + x) ** 2 + \
                            (-d + y) ** 2 + (-h + z) ** 2))) - (-w + x) ** 2 / (np.sqrt((-w + x) ** 2 + (d + y) ** 2 + \
                                (-h + z) ** 2) * (d + y + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2))) + \
                                    (w + x) ** 2 / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2) * (d + y + np.sqrt((w + x) ** 2 + \
                                        (d + y) ** 2 + (-h + z) ** 2)))
    cx4 = (-d + y) / np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) + (d - y) / np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) + \
        (-d - y) / np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) + (d + y) / np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) + \
            (d - y) / np.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) + \
                (-d + y) / np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2) + (d + y) / np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2) + \
                    (-d - y) / np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)
    return - 1 / 4 / np.pi * (cx11 + cx12 + cx2 + cx3 + cx4)

def H_y_bar(w, d, h, x, y, z):
    cy11 = (h + z) ** 2 * ((-w + x) / ((d ** 2 + h ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * \
        np.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2)) - \
            (w + x) / ((d ** 2 + h ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * np.sqrt((w + x) ** 2 + (d - y) ** 2 + \
                (h + z) ** 2)) + (w - x) / ((d ** 2 + h ** 2 + 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * np.sqrt((w - x) ** 2 + \
                    (d + y) ** 2 + (h + z) ** 2)) + (w + x) / ((d ** 2 + h ** 2 + 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * \
                        np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)))

    cy12 = (h - z) ** 2 * ((w - x) / (np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 - \
        2 * h * z + z ** 2)) + (w + x) / (np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 - \
            2 * h * z + z ** 2)) + (-w + x) / (np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (d ** 2 + h ** 2 + 2 * d * y + y ** 2 - \
                2 * h * z + z ** 2)) - (w + x) / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (d ** 2 + h ** 2 + 2 * d * y + y ** 2 - \
                    2 * h * z + z ** 2)))

    cy2 = (-w + x) / np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) + (-w - x) / np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) + \
        (w - x) / np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) + (w + x) / np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) + (w - x) / \
            np.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) + (w + x) / np.sqrt((w + x) ** 2 + \
                (d - y) ** 2 + (h + z) ** 2) + (-w + x) / np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2) + (-w - x) / np.sqrt((w + x) ** 2 + \
                    (d + y) ** 2 + (h + z) ** 2)

    cy3 = -np.log(-w + x + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) + np.log(w + x + np.sqrt((w + x) ** 2 + (-d + y) ** 2 + \
        (-h - z) ** 2)) + np.log(-w + x + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) - np.log(w + x + np.sqrt((w + x) ** 2 + \
            (d + y) ** 2 + (-h - z) ** 2)) + np.log(-w + x + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2)) - np.log(w + x + \
                np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2)) - np.log(-w + x + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2)) + \
                    np.log(w + x + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2))

    cy4 = (d - y) ** 2 / ((-w + x + np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) * np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
        (d - y) ** 2 / ((w + x + np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) * np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
            (d + y) ** 2 / ((-w + x + np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) * np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) + \
                (d + y) ** 2 / ((w + x + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) * np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) - \
                    (d - y) ** 2 / (np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2) * (-w + x + np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
                        (d - y) ** 2 / (np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2) * (w + x + np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
                            (d + y) ** 2 / (np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2) * (-w + x + np.sqrt((w - x) ** 2 + (d + y) ** 2 + \
                                (h + z) ** 2))) - (d + y) ** 2 / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2) * (w + x + np.sqrt((w + x) ** 2 + \
                                    (d + y) ** 2 + (h + z) ** 2)))
    return - 1 / 4 / np.pi * (cy11 + cy12 + cy2 + cy3 + cy4)

def H_z_bar(w, d, h, x, y, z):
    cz1 = np.arctan(((w + x) * (d - y)) / (np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h - z))) + \
        np.arctan(((w - x) * (d + y)) / (np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h - z))) - \
            np.arctan(((-w + x) * (-d + y)) / (np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (-h + z))) - \
                np.arctan(((w + x) * (d + y)) / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (-h + z))) + \
                    np.arctan(((w - x) * (d - y)) / ((h + z) * np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
                        np.arctan(((w + x) * (d - y)) / ((h + z) * np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
                            np.arctan(((w - x) * (d + y)) / ((h + z) * np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2))) + \
                                np.arctan(((w + x) * (d + y)) / ((h + z) * np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)))

    cz21 = (h + z) * (-(((w - x) * (d - y) * (d ** 2 + 2 * h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 4 * h * z + 2 * z ** 2)) / \
        ((h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * \
            np.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2))) - ((w + x) * (d - y) * \
                (d ** 2 + 2 * h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 4 * h * z + 2 * z ** 2)) / ((h ** 2 + w ** 2 + \
                    2 * w * x + x ** 2 + 2 * h * z + z ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * np.sqrt((w + x) ** 2 + \
                        (d - y) ** 2 + (h + z) ** 2)) - ((w - x) * (d + y) * (d ** 2 + 2 * h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * d * y + \
                            y ** 2 + 4 * h * z + 2 * z ** 2)) / ((h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * (d ** 2 + h ** 2 + \
                                2 * d * y + y ** 2 + 2 * h * z + z ** 2) * np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2)) - ((w + x) * (d + y) * \
                                    (d ** 2 + 2 * h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * d * y + y ** 2 + 4 * h * z + 2 * z ** 2)) / ((h ** 2 + \
                                        w ** 2 + 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * (d ** 2 + h ** 2 + 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * \
                                            np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)))

    cz22 = (h - z) * (-(((w - x) * (d - y) * (d ** 2 + 2 * h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 - 4 * h * z + 2 * z ** 2)) / \
        (np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * h * z + z ** 2) * \
            (d ** 2 + h ** 2 - 2 * d * y + y ** 2 - 2 * h * z + z ** 2))) - ((w + x) * (d - y) * (d ** 2 + 2 * h ** 2 + w ** 2 + 2 * w * x + x ** 2 - \
                2 * d * y + y ** 2 - 4 * h * z + 2 * z ** 2)) / (np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 + 2 * w * x + \
                    x ** 2 - 2 * h * z + z ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 - 2 * h * z + z ** 2)) - ((w - x) * (d + y) * \
                        (d ** 2 + 2 * h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * d * y + y ** 2 - 4 * h * z + 2 * z ** 2)) / \
                            (np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * h * z + z ** 2) * \
                                (d ** 2 + h ** 2 + 2 * d * y + y ** 2 - 2 * h * z + z ** 2)) - ((w + x) * (d + y) * (d ** 2 + 2 * h ** 2 + w ** 2 + \
                                    2 * w * x + x ** 2 + 2 * d * y + y ** 2 - 4 * h * z + 2 * z ** 2)) / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + \
                                        (h - z) ** 2) * (h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * h * z + z ** 2) * (d ** 2 + h ** 2 + 2 * d * y + \
                                            y ** 2 - 2 * h * z + z ** 2)))

    cz3 = ((-w + x) * (-h - z)) / ((-d + y + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) * np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + \
        (-h - z) ** 2)) - ((w + x) * (-h - z)) / ((-d + y + np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) * np.sqrt((w + x) ** 2 + \
            (-d + y) ** 2 + (-h - z) ** 2)) - ((-w + x) * (-h - z)) / ((d + y + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) * \
                np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) + ((w + x) * (-h - z)) / ((d + y + np.sqrt((w + x) ** 2 + (d + y) ** 2 + \
                    (-h - z) ** 2)) * np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) + ((-w + x) * (-h + z)) / (np.sqrt((-w + x) ** 2 + \
                        (-d + y) ** 2 + (-h + z) ** 2) * (-d + y + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2))) - ((w + x) * (-h + z)) / \
                            (np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2) * (-d + y + np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2))) - \
                                ((-w + x) * (-h + z)) / (np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2) * (d + y + np.sqrt((-w + x) ** 2 + \
                                    (d + y) ** 2 + (-h + z) ** 2))) + ((w + x) * (-h + z)) / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2) * \
                                        (d + y + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2)))

    cz4 = ((d - y) * (h - z)) / ((-w + x + np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) * np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
        ((d - y) * (h - z)) / ((w + x + np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) * np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) + \
            ((d + y) * (h - z)) / ((-w + x + np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) * np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) + \
                ((d + y) * (-h + z)) / ((w + x + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) * np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) + \
                    ((d - y) * (h + z)) / (np.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2) * (-w + x + np.sqrt((w - x) ** 2 + (d - y) ** 2 + \
                        (h + z) ** 2))) - ((d - y) * (h + z)) / (np.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2) * (w + x + np.sqrt((w + x) ** 2 + \
                            (d - y) ** 2 + (h + z) ** 2))) + ((d + y) * (h + z)) / (np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2) * \
                                (-w + x + np.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2))) - ((d + y) * (h + z)) / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + \
                                    (h + z) ** 2) * (w + x + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)))

    return - 1. / 4. / np.pi * (cz1 + cz21 + cz22 + cz3 + cz4)


def free_current_potential_bar_magnet(w, d, h):
    return lambda coords: np.array([H_x_bar(w, d, h, *coords),
                                    H_y_bar(w, d, h, *coords),
                                    H_z_bar(w, d, h, *coords)
                                    ])
