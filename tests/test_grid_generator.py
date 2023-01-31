from source.grid_generator import bar_gear_mesh, ball_gear_mesh, segment_mesh
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear
import numpy as np
import subprocess
import os

def test_ball_gear_mesh():
    n = 5
    R = 10.
    r = 1.
    x_M = np.random.rand(3)
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)
    ball_gear = MagneticBallGear(n, R, r, x_M, axis)

    M0 = 10.  # magnetization strength
    ball_gear.create_magnets(M0)

    _ = ball_gear_mesh(ball_gear, mesh_size_space=2.0, mesh_size_magnets=0.3, padding=ball_gear.R / 10, \
        fname="testball", verbose=True)

def test_bar_gear_mesh():
    # create gear
    n = 5
    R = 10.
    h = 1.5
    w = .5
    d = .5
    x_M = np.random.rand(3)
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)

    bar_gear = MagneticBarGear(n, R, h, w, d, x_M, axis)
    M0 = 10.  # magnetization strength
    bar_gear.create_magnets(M0)

    _ = bar_gear_mesh(bar_gear, mesh_size_space=1.0, mesh_size_magnets=0.2, padding=bar_gear.R / 10, \
        fname="testbar", verbose=True)

def test_segment_mesh():
    x_M = np.array([0., 0., 0.])
    x_M_start =np.array([0., 20., 0.])
    axis = np.array([1., 0., 0.])
    Ri = 10.
    Ro = 15.
    thickness = 2.
    angle = np.pi
    mesh_size = 0.6
    fname = "segment"
    mesh, cell_marker, facet_marker = segment_mesh(x_M, x_M_start, axis, Ri, Ro, thickness, angle, mesh_size, fname, write_to_pvd=True, verbose=False)

if __name__ == "__main__":
    # create test directory
    test_dir = "test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    os.chdir(test_dir)

    #test_ball_gear_mesh()
    #test_bar_gear_mesh()
    test_segment_mesh()

    # remove test directory
    #os.chdir("..")
    #subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)