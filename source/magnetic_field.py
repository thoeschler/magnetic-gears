import sympy as sy

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


if __name__ == '__main__':
    H = free_current_potential_bar_magnet(1, 1, 1, lambdify=False)
    print(H)