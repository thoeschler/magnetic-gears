from source.magnet_classes import BallMagnet, BarMagnet
from source.magnetic_gear_classes import MagneticGearWithBallMagnets, MagneticGearWithBarMagnets
import numpy as np
import subprocess
import os


def test_ball_gear():
    # create gear
    n = 5
    R = 10.
    r = 1.
    x_M = np.random.rand(3)
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)

    ball_gear = MagneticGearWithBallMagnets(n, R, r, x_M, axis)

    assert ball_gear.n == n
    assert ball_gear.R == R
    assert ball_gear.r == r
    assert np.all(ball_gear.x_M == x_M)
    assert np.all(ball_gear.axis == axis)
    assert ball_gear.angle == 0.  # initial angle is always zero

    M0 = 10.  # magnetization strength
    ball_gear.create_magnets(M0)

    for magnet in ball_gear.magnets:
        assert isinstance(magnet, BallMagnet)
        assert np.isclose(np.linalg.norm(magnet.x_M - ball_gear.x_M), ball_gear.R) 
        assert np.allclose(magnet.Q.dot(ball_gear.axis), ball_gear.axis)

def test_bar_gear():
    # create gear
    n = 5
    R = 10.
    h = 1.
    w = 2.
    d = 3.
    x_M = np.random.rand(3)
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)

    bar_gear = MagneticGearWithBarMagnets(n, R, h, w, d, x_M, axis)
    M0 = 10.  # magnetization strength
    bar_gear.create_magnets(M0)

    for magnet in bar_gear.magnets:
        assert isinstance(magnet, BarMagnet)
        assert np.isclose(np.linalg.norm(magnet.x_M - bar_gear.x_M), bar_gear.R)
        assert np.allclose(magnet.Q.dot(bar_gear.axis), bar_gear.axis)

if __name__ == "__main__":
    # create test directory
    test_dir = "test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    os.chdir(test_dir)

    # run tests
    test_ball_gear()
    test_bar_gear()

    # remove test directory
    os.chdir("..")
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)