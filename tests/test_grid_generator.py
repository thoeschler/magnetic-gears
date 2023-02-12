from source.grid_generator import gear_mesh
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear, SegmentGear
import numpy as np
import subprocess
import os

def test_ball_gear_mesh():
    n = 5
    R = 10.
    r = 1.
    x_M = np.random.rand(3)

    ball_gear = MagneticBallGear(n, R, r, x_M)

    M0 = 10.  # magnetization strength
    ball_gear.create_magnets(M0)

    _ = gear_mesh(ball_gear, mesh_size_space=2.0, mesh_size_magnets=0.3, padding=ball_gear.r, \
        fname="testball", write_to_pvd=True, verbose=True)

def test_bar_gear_mesh():
    # create gear
    n = 5
    R = 10.
    h = 1.5
    w = 1.
    d = .5
    x_M = np.random.rand(3)

    bar_gear = MagneticBarGear(n, R, d, w, h, x_M)
    M0 = 10.  # magnetization strength
    bar_gear.create_magnets(M0)

    _ = gear_mesh(bar_gear, mesh_size_space=1.0, mesh_size_magnets=0.3, padding=bar_gear.h, \
        fname="testbar", write_to_pvd=True, verbose=True)

def test_segment_mesh():
    # create gear
    n = 4
    R = 7.
    w = 1.
    d = .5
    x_M = np.zeros(3)
    
    mesh_size_space = 3.0
    mesh_size_magnets = 0.4
    fname = "testsegment"

    segment_gear = SegmentGear(n, R, d, w, x_M)
    M0 = 10.  # magnetization strength
    segment_gear.create_magnets(M0)

    for mag in segment_gear.magnets:
        print(mag.x_M)

    _ = gear_mesh(segment_gear, mesh_size_space, mesh_size_magnets, fname, padding=w, \
                  write_to_pvd=True, verbose=True)

if __name__ == "__main__":
    # create test directory
    test_dir = "test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    os.chdir(test_dir)

    test_ball_gear_mesh()
    test_bar_gear_mesh()
    test_segment_mesh()

    # remove test directory
    os.chdir("..")
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)