import os
import re
from collections import OrderedDict


def to_python(expr):
    assert isinstance(expr, str)
    # 1. replace paratheses
    expr = expr.replace("[", "(")
    expr = expr.replace("]", ")")
    expr = re.sub(r"\\\((\w+)\)", r"\1", expr)  # \(Alpha) --> Alpha

    # 2. insert some "*", replace names
    map_dir = OrderedDict((
        ("\\\n", " "),  # \\\n --> nothing
        ("\)\s*\(", ") * ("),  # ) ( -> ) * ( 
        ("^", " ** "),
        ("ArcTan", "np.arctan"),
        ("Sin", "np.sin"),
        ("Cos","np.cos"),
        ("Tan", "np.tan"),
        ("Log", "np.log"),
        ("Sqrt", "np.sqrt"),
        ("Alpha", "alpha")
    ))

    for math, py in map_dir.items():
        expr = expr.replace(math, py)
    
    # fill in some "*"
    expr = re.sub(r"\s{2,}", " ", expr)
    """fill = re.findall("[\w\)]\s+[\w\(]", expr)
    for f in fill:
        expr = expr.replace(f, f.replace(" ", " * "))"""
    expr = re.sub(r"([\w\)])\s+([\w\(])", r"\1 * \2", expr)
    expr = re.sub(r"([\w\)])\s+([\w\(])", r"\1 * \2", expr)

    # some whitespace stuff
    expr = re.sub(r"(\()\s+", r"(", expr)
    expr = re.sub(r"(\S)[/](\S)", r"\1 / \2", expr)
    return expr

def bar_magnet_python():
    dir_name = os.path.dirname(os.path.realpath(__file__))
    # potential
    with open(os.path.join(dir_name, "bar", "V_bar.txt"), "r") as f:
        expr = f.read()
        with open(os.path.join(dir_name, "bar", "V_bar_magnet.py.raw"), "a+") as f_py:
            f_py.write(to_python(expr) + "\n")
    # x-component
    for fname in ("x_t1.1.txt", "x_t1.2.txt", "x_t2.txt", "x_t3.txt", "x_t4.txt"):
        with open(os.path.join(dir_name, "bar", f"{fname}"), "r") as f:
            expr = f.read()
            with open(os.path.join(dir_name, "bar", "bar_magnet_x.py.raw"), "a+") as f_py:
                f_py.write(to_python(expr) + "\n")
    # y-component
    for fname in ("y_t1.1.txt", "y_t1.2.txt", "y_t2.txt", "y_t3.txt", "y_t4.txt"):
        with open(os.path.join(dir_name, "bar", f"{fname}"), "r") as f:
            expr = f.read()
            with open(os.path.join(dir_name, "bar", "bar_magnet_y.py.raw"), "a+") as f_py:
                f_py.write(to_python(expr) + "\n")
    # z-component
    for fname in ("z_t1.txt", "z_t2.1.txt", "z_t2.2.txt", "z_t3.txt", "z_t4.txt"):
        with open(os.path.join(dir_name, "bar", f"{fname}"), "r") as f:
            expr = f.read()
            with open(os.path.join(dir_name, "bar", "bar_magnet_z.py.raw"), "a+") as f_py:
                f_py.write(to_python(expr) + "\n")


def magnet_segment_python():
    dir_name = os.path.dirname(os.path.realpath(__file__))
    for c in ("x", "y", "z"):
        open(os.path.join(dir_name, "cylinder_segment", f"cylinder_segment_{c}.py.raw"), "w").close()
    # x-component
    for fname in ("x_t1.1.txt", "x_t1.2.txt", "x_t2.1.txt", "x_t2.2.txt", "x_t3.txt"):
        with open(os.path.join(dir_name, "cylinder_segment", "f{fname}"), "r") as f:
            expr = f.read()
            with open(os.path.join(dir_name, "cylinder_segment", "cylinder_segment_x.py.raw"), "a+") as f_py:
                f_py.write(to_python(expr) + "\n")
    # y-component
    for fname in ("y_t1.1.txt", "y_t1.2.txt", "y_t2.1.txt", "y_t2.2.txt", "y_t3.txt"):
        with open(os.path.join(dir_name, "cylinder_segment", f"{fname}"), "r") as f:
            expr = f.read()
            with open(os.path.join(dir_name, "cylinder_segment", "cylinder_segment_y.py.raw"), "a+") as f_py:
                f_py.write(to_python(expr) + "\n")
    # z-component
    for fname in ("z_t1.txt", "z_t2.txt", "z_t3.1.txt", "z_t3.2.txt"):
        with open(os.path.join(dir_name, "cylinder_segment", f"{fname}"), "r") as f:
            expr = f.read()
            with open(os.path.join(dir_name, "cylinder_segment", "cylinder_segment_z.py.raw"), "a+") as f_py:
                f_py.write(to_python(expr) + "\n")


if __name__ == "__main__":
    bar_magnet_python()
    #magnet_segment_python()