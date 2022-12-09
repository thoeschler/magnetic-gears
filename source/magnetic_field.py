import sympy as sy


def magnetic_potential(height, width, depth, lambdify=True):
    # some symbols
    x, y, z = sy.symbols("x y z", real=True)

    Q1 = sy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    Q2 = sy.Identity(3)
    c1 = sy.Matrix([0, 0, - height])
    c2 = sy.Matrix([0, 0, height])

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
        return sy.lambdify([(x, y, z)], V_m), (x, y, z)
    else:
        return V_m, (x, y, z)

def free_current_potential_bar_magnet(height, width, depth):
    # get magnetic potential
    V_m, (x, y, z) = magnetic_potential(height, width, depth, lambdify=False)

    # compute free current potential
    H = (- V_m.diff(x), - V_m.diff(y), - V_m.diff(z))

    return sy.lambdify([(x, y, z)], H)
