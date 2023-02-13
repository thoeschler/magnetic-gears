from source.magnet_classes import BallMagnet, BarMagnet, CylinderSegment
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear, SegmentGear
import numpy as np
import subprocess
import os


def test_ball_gear():
    # create gear
    n = 5
    R = 10.
    r = 1.
    x_M = np.random.rand(3)

    ball_gear = MagneticBallGear(n, R, r, x_M)

    assert ball_gear.n == n
    assert ball_gear.R == R
    assert ball_gear.r == r
    assert np.all(ball_gear.x_M == x_M)
    assert ball_gear.angle == 0.  # initial angle is always zero

    M0 = 10.  # magnetization strength
    ball_gear.create_magnets(M0)

    for magnet in ball_gear.magnets:
        assert isinstance(magnet, BallMagnet)
        assert np.isclose(np.linalg.norm(magnet.x_M - ball_gear.x_M), ball_gear.R) 
        assert np.allclose(magnet.Q.dot(np.array([1., 0., 0])), np.array([1., 0., 0]))

def test_bar_gear():
    # create gear
    n = 5
    R = 10.
    h = 1.
    w = 2.
    d = 3.
    x_M = np.random.rand(3)

    bar_gear = MagneticBarGear(n, R, d, w, h, x_M)

    assert bar_gear.n == n
    assert bar_gear.R == R
    assert bar_gear.d == d
    assert bar_gear.w == w
    assert bar_gear.h == h
    assert np.all(bar_gear.x_M == x_M)
    assert bar_gear.angle == 0.  # initial angle is always zero

    M0 = 10.  # magnetization strength
    bar_gear.create_magnets(M0)

    for magnet in bar_gear.magnets:
        assert isinstance(magnet, BarMagnet)
        assert np.isclose(np.linalg.norm(magnet.x_M - bar_gear.x_M), bar_gear.R)
        assert np.allclose(magnet.Q.dot(np.array([1., 0., 0])), np.array([1., 0., 0]))

def test_segment_gear():
    # create gear
    n = 5
    R = 10.
    w = 2.
    d = 3.
    x_M = np.random.rand(3)

    segment_gear = SegmentGear(n, R, d, w, x_M)

    assert segment_gear.n == n
    assert segment_gear.R == R
    assert segment_gear.d == d
    assert segment_gear.w == w
    assert np.all(segment_gear.x_M == x_M)
    assert segment_gear.angle == 0.  # initial angle is always zero

    M0 = 10.  # magnetization strength
    segment_gear.create_magnets(M0)

    for magnet in segment_gear.magnets:
        assert isinstance(magnet, MagnetSegment)
        assert np.isclose(np.linalg.norm(magnet.x_M - segment_gear.x_M), segment_gear.R)
        assert np.allclose(magnet.Q.dot(np.array([1., 0., 0])), np.array([1., 0., 0]))

if __name__ == "__main__":
    # create test directory
    test_dir = "test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    os.chdir(test_dir)

    # run tests
    test_ball_gear()
    test_bar_gear()
    test_segment_gear()

    # remove test directory
    os.chdir("..")
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)