import numpy as np


def magnetic_potential_bar_magnet(w, d, h):
    # the function uses half the actual values
    w /= 2
    d /= 2
    h /= 2
    def V(coords):
        x, y, z = coords
        Vm_val = -(h + z) * np.arctan(((-w + x) * (-d + y)) / (np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2) * (-h - z))) + (h + z) * np.arctan(((w + x) * (-d + y)) / (np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2) * (-h - z))) + (h + z) * np.arctan(((-w + x) * (d + y)) / (np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2) * (-h - z))) - (h + z) * np.arctan(((w + x) * (d + y)) / (np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2) * (-h - z))) + (h - z) * np.arctan(((-w + x) * (-d + y)) / ((-h + z) * np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2))) - (h - z) * np.arctan(((w + x) * (-d + y)) / ((-h + z) * np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2))) - (h - z) * np.arctan(((-w + x) * (d + y)) / ((-h + z) * np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2))) + (h - z) * np.arctan(((w + x) * (d + y)) / ((-h + z) * np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2))) - (-d + y) * np.log(-w + x + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) - (-w + x) * np.log(-d + y + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) + (-d + y) * np.log(w + x + np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) + (w + x) * np.log(-d + y + np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h - z) ** 2)) + (d + y) * np.log(-w + x + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) + (-w + x) * np.log(d + y + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) - (d + y) * np.log(w + x + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) - (w + x) * np.log(d + y + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h - z) ** 2)) + (-d + y) * np.log(-w + x + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2)) + (-w + x) * np.log(-d + y + np.sqrt((-w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2)) - (-d + y) * np.log(w + x + np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2)) - (w + x) * np.log(-d + y + np.sqrt((w + x) ** 2 + (-d + y) ** 2 + (-h + z) ** 2)) - (d + y) * np.log(-w + x + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2)) - (-w + x) * np.log(d + y + np.sqrt((-w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2)) + (d + y) * np.log(w + x + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2)) + (w + x) * np.log(d + y + np.sqrt((w + x) ** 2 + (d + y) ** 2 + (-h + z) ** 2))
        return Vm_val / 4. / np.pi
    return V

def H_x_bar(w, d, h, x, y, z):
    # the function uses half the actual values
    w /= 2
    d /= 2
    h /= 2
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
    # the function uses half the actual values
    w /= 2
    d /= 2
    h /= 2
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
    # the function uses half the actual values
    w /= 2
    d /= 2
    h /= 2
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
