import sympy as sy


def magnetic_potential_bar_magnet(height, width, depth, lambdify=True):
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

"""def free_current_potential_bar_magnet(height, width, depth):
    # get magnetic potential
    V_m, (x, y, z) = magnetic_potential_bar_magnet(height, width, depth, lambdify=False)

    # compute free current potential
    H = (- V_m.diff(x), - V_m.diff(y), - V_m.diff(z))

    return sy.lambdify([(x, y, z)], H)"""


def free_current_potential_bar_magnet(h, w, d, lambdify=True):
    x, y, z = sy.symbols("x y z", real=True)

    tx1_1 = (h + z) ** 2 * (
        (-d + y) / ((h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * \
            sy.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2)) + \
                (d - y) / ((h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2)) -  \
                    (d + y) / ((h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2)) + \
                        (d + y) / ((h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2))
                    )

    tx1_2 = (h - z) ** 2 * (
        (d - y) / (sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * h * z + z ** 2)) + \
            (d + y) / (sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * h * z + z ** 2)) + \
                (-d + y) / (sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * h * z + z ** 2)) - \
                    (d + y) / (sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * h * z + z ** 2))
                    )

    tx2 = sy.log(-d + y + sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
        sy.log(-d + y + sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
            sy.log(d + y + sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) + \
                sy.log(d + y + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) - \
                    sy.log(-d + y + sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2)) + \
                        sy.log(-d + y + sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2)) + \
                            sy.log(d + y + sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2)) - \
                                sy.log(d + y + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2))
    
    tx3_1 = - ((- w + x) ** 2 / (
        (-d + y + sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) * sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2))) + \
            (w + x) ** 2 / ((-d + y + sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) * sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) + \
                (-w + x) ** 2 / ((d + y + sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) * sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) - \
                    (w + x) ** 2 / ((d + y + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) * sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) 
                    
                    
    tx3_2 = (-w + x) ** 2 / (sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2) * (-d + y + sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2))) - \
        (w + x) ** 2 / (sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2) * (-d + y + sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2))) - \
            (-w + x) ** 2 / (sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2) * (d + y + sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2))) + \
                (w + x) ** 2 / (sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2) * (d + y + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2)))


    tx4 = (-d + y) / sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) + (d - y) / sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) + \
        (-d - y) / sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) + (d + y) / sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) + \
            (d - y) / sy.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) + \
                (-d + y) / sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2) + (d + y) / sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2) + \
                    (-d - y) / sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)
    
    
    ty1_1 = (h + z) ** 2 * (
        (-w + x) / ((d ** 2 + h ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * \
            sy.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2)) - \
                (w + x) / ((d ** 2 + h ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2)) + \
                    (w - x) / ((d ** 2 + h ** 2 + 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2)) + \
                        (w + x) / ((d ** 2 + h ** 2 + 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2))
                        )

    ty1_2 = (h - z) ** 2 * (
        (w - x) / (sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 - 2 * h * z + z ** 2)) + \
            (w + x) / (sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 - 2 * h * z + z ** 2)) + \
                (-w + x) / (sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (d ** 2 + h ** 2 + 2 * d * y + y ** 2 - 2 * h * z + z ** 2)) - \
                    (w + x) / (sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (d ** 2 + h ** 2 + 2 * d * y + y ** 2 - 2 * h * z + z ** 2))
                    )

    ty2 = (-w + x) / sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) + (-w - x) / sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) + \
        (w - x) / sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) + (w + x) / sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) + \
            (w - x) / sy.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) + \
                (w + x) / sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2) + (-w + x) / sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2) + \
                    (-w - x) / sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)

    ty3 = - sy.log(-w + x + sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) + \
        sy.log(w + x + sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) + \
            sy.log(-w + x + sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) - \
                sy.log(w + x + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) + \
                    sy.log(-w + x + sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2)) - \
                        sy.log(w + x + sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2)) - \
                            sy.log(-w + x + sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2)) + \
                                sy.log(w + x + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2))


    ty4_1 = (d - y) ** 2 / ((-w + x + sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) * sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
        (d - y) ** 2 / ((w + x + sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) * sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
            (d + y) ** 2 / ((-w + x + sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) * sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) + \
                (d + y) ** 2 / ((w + x + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) * sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2))
    ty4_2 = - (d - y) ** 2 / (sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2) * (-w + x + sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
        (d - y) ** 2 / (sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2) * (w + x + sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
            (d + y) ** 2 / (sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2) * (-w + x + sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2))) - \
                (d + y) ** 2 / (sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2) * (w + x + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)))

    tz1 = sy.atan(((w + x) * (d - y)) / (sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h - z))) + \
        sy.atan(((w - x) * (d + y)) / (sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h - z))) - \
            sy.atan(((-w + x) * (-d + y)) / (sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (-h + z))) - \
                sy.atan(((w + x) * (d + y)) / (sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (-h + z))) + \
                    sy.atan(((w - x) * (d - y)) / ((h + z) * sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
                        sy.atan(((w + x) * (d - y)) / ((h + z) * sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
                            sy.atan(((w - x) * (d + y)) / ((h + z) * sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2))) + \
                                sy.atan(((w + x) * (d + y)) / ((h + z) * sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)))

    tz2_1_1 = (h + z) * (
        - (((w - x) * (d - y) * (d ** 2 + 2 * h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 4 * h *z + 2 * z ** 2)) / \
            ((h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * \
                sy.sqrt(d ** 2 + h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2))) - \
                    ((w + x) * (d - y) * (d ** 2 + 2 * h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * d * y + y ** 2 + 4 * h * z + 2 * z ** 2)) / \
                        ((h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * (d ** 2 + h ** 2 - 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * \
                            sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2))
                            )
                            
    tz2_1_2 = (h + z) * (
        - ((w - x) * (d + y) * (d ** 2 + 2 * h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * d * y + y ** 2 + 4 * h * z + 2 * z ** 2)) / \
            ((h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * (d ** 2 + h ** 2 + 2 * d * y + y ** 2 + 2 * h * z +  z ** 2) * \
                sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2)) - \
                    ((w + x) * (d + y) * (d ** 2 + 2 * h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * d * y + y ** 2 + 4 * h * z + 2 * z ** 2)) / \
                        ((h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * h * z + z ** 2) * (d ** 2 + h ** 2 + 2 * d * y + y ** 2 + 2 * h * z + z ** 2) * \
                            sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2))
                            )

    tz2_2_1 = (h - z) * (
        - (((w - x) * (d - y) * (d ** 2 + 2 * h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * d * y + y ** 2 - 4 * h * z + 2 * z ** 2)) / \
            (sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * h * z + z ** 2) * \
                (d ** 2 + h ** 2 - 2 * d * y + y ** 2 - 2 * h * z + z ** 2))) - \
                    ((w + x) * (d - y) * (d ** 2 + 2 * h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * d * y + y ** 2 - 4 * h * z + 2 * z ** 2)) / \
                        (sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * h * z + z ** 2) * \
                            (d ** 2 + h ** 2 - 2 * d * y + y ** 2 - 2 * h * z + z ** 2))
                            )
                            
    tz2_2_2 = (h - z) * (
        - ((w - x) * (d + y) * (d ** 2 + 2 * h ** 2 + w ** 2 - 2 * w * x + x ** 2 + 2 * d * y + y ** 2 - 4 * h * z + 2 * z ** 2)) / \
            (sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 - 2 * w * x + x ** 2 - 2 * h * z + z ** 2) * \
                (d ** 2 + h ** 2 + 2 * d * y + y ** 2 - 2 * h * z + z ** 2)) - \
                    ((w + x) * (d + y) * (d ** 2 + 2 * h ** 2 + w ** 2 + 2 * w * x + x ** 2 + 2 * d * y + y ** 2 - 4 * h * z + 2 * z ** 2)) / \
                        (sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2) * (h ** 2 + w ** 2 + 2 * w * x + x ** 2 - 2 * h * z + z ** 2) * \
                            (d ** 2 + h ** 2 + 2 * d * y + y ** 2 - 2 * h * z + z ** 2))
                            )

    tz3_1 = ((-w + x) * (-h - z)) / ((-d + y + sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) * sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) \
        - ((w + x) * (-h - z)) / ((-d + y + sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) * sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) - \
            ((-w + x) * (-h - z)) / ((d + y + sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) * sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) + \
                ((w + x) * (-h - z)) / ((d + y + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) * sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) 
                
                
    tz3_2 = ((-w + x) * (-h + z)) / (sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2) * (-d + y + sy.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2))) - \
        ((w + x) * (-h + z)) / (sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2) * (-d + y + sy.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2))) - \
            ((-w + x) * (-h + z)) / (sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2) * (d + y + sy.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2))) + \
                ((w + x) * (-h + z)) / (sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2) * (d + y + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2)))

    tz4_1 = ((d - y) * (h - z)) / ((-w + x + sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) * sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) - \
        ((d - y) * (h - z)) / ((w + x + sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) * sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h - z) ** 2)) + \
            ((d + y) * (h - z)) / ((-w + x + sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) * sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) + \
                ((d + y) * (-h + z)) / ((w + x + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2)) * sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h - z) ** 2))
                
    tz4_2 = ((d - y) * (h + z)) / (sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2) * (-w + x + sy.sqrt((w - x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) - \
        ((d - y) * (h + z)) / (sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2) * (w + x + sy.sqrt((w + x) ** 2 + (d - y) ** 2 + (h + z) ** 2))) + \
            ((d + y) * (h + z)) / (sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2) * (-w + x + sy.sqrt((w - x) ** 2 + (d + y) ** 2 + (h + z) ** 2))) - \
                ((d + y) * (h + z)) / (sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2) * (w + x + sy.sqrt((w + x) ** 2 + (d + y) ** 2 + (h + z) ** 2)))

    H = (- 1. / 4. / sy.pi * (tx1_1 + tx1_2 + tx2 + tx3_1 + tx3_2 + tx4), \
        - 1. / 4. / sy.pi * (ty1_1 + ty1_2 + ty2 + ty3 + ty4_1 + ty4_2), \
            - 1. / 4. / sy.pi * (tz1 + tz2_1_1 + tz2_1_2 + tz2_2_1 + tz2_2_2 + tz3_1 + tz3_2 + tz4_1 + tz4_2))
    
    if lambdify:
        return sy.lambdify([(x, y, z)], H)
    else:
        return H