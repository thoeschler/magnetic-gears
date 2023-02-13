import numpy as np
from source.magnet_classes import BallMagnet, BarMagnet, CylinderSegment
from scipy.spatial.transform import Rotation


def test_ball_magnet():
    # create ball magnet
    R = 5
    M0 = 20.
    Q = np.eye(3)
    x_M = np.random.rand(3)
    ball_magnet = BallMagnet(radius=R, magnetization_strength=M0, \
        position_vector=x_M, rotation_matrix=Q)

    assert ball_magnet.type == "Ball"
    assert ball_magnet.R == R
    assert np.all(ball_magnet.x_M == x_M)
    assert ball_magnet.M0 == M0
    assert np.all(ball_magnet.Q == Q)

    # check basic functionality
    assert callable(ball_magnet.Vm_eigen)
    Vm_test = ball_magnet.Vm_eigen(np.random.rand(3))
    assert isinstance(Vm_test, float)

    assert callable(ball_magnet.Vm)
    Vm_test = ball_magnet.Vm(np.random.rand(3))
    assert isinstance(Vm_test, float)

    assert callable(ball_magnet.B_eigen)
    B_test = ball_magnet.B_eigen(np.random.rand(3))
    assert isinstance(B_test, np.ndarray)
    assert len(B_test) == 3

    assert callable(ball_magnet.B)
    B_test = ball_magnet.B(np.random.rand(3))
    assert isinstance(B_test, np.ndarray)
    assert len(B_test) == 3

    # update magnet
    Q = Rotation.from_rotvec(np.random.rand(3)).as_matrix()
    assert np.allclose(Q.dot(Q.T), np.eye(3))

    ball_magnet.Q = Q  # set new rotation matrix
    assert np.all(Q == ball_magnet.Q)
    assert np.allclose(Q.dot(np.array([0., 0., 1.])) - ball_magnet.M, 0.)

    x_M = np.random.rand(3)  # set new center of mass
    ball_magnet.x_M = x_M
    assert np.all(x_M == ball_magnet.x_M)

    # update both at the same time
    Q = Rotation.from_rotvec(np.random.rand(3)).as_matrix()
    x_M = np.random.rand(3)
    ball_magnet.update_parameters(x_M, Q)
    assert np.all(x_M == ball_magnet.x_M)
    assert np.all(Q == ball_magnet.Q)
    assert np.allclose(Q.dot(np.array([0., 0., 1.])) - ball_magnet.M, 0.) 

    # delete ball magnet
    del ball_magnet


def test_bar_magnet():
    # create bar magnet
    w = 2.0
    d = 3.0
    h = 4.0
    M0 = 20.
    Q = np.eye(3)
    x_M = np.random.rand(3)
    bar_magnet = BarMagnet(height=h, width=w, depth=d, magnetization_strength=M0, \
        position_vector=x_M, rotation_matrix=Q)

    assert bar_magnet.type == "Bar"
    assert bar_magnet.w == w
    assert bar_magnet.d == d
    assert bar_magnet.h == h
    assert np.all(bar_magnet.x_M == x_M)
    assert bar_magnet.M0 == M0
    assert np.all(bar_magnet.Q == Q)

    # check basic functionality
    assert callable(bar_magnet.Vm_eigen)
    Vm_test = bar_magnet.Vm_eigen(np.random.rand(3))
    assert isinstance(Vm_test, float)

    assert callable(bar_magnet.Vm)
    Vm_test = bar_magnet.Vm(np.random.rand(3))
    assert isinstance(Vm_test, float)

    assert callable(bar_magnet.B_eigen)
    B_test = bar_magnet.B_eigen(np.random.rand(3))
    assert isinstance(B_test, np.ndarray)
    assert len(B_test) == 3

    assert callable(bar_magnet.B)
    B_test = bar_magnet.B(np.random.rand(3))
    assert isinstance(B_test, np.ndarray)
    assert len(B_test) == 3

    # update magnet
    Q = Rotation.from_rotvec(np.random.rand(3)).as_matrix()
    assert np.allclose(Q.dot(Q.T), np.eye(3))

    bar_magnet.Q = Q  # set new rotation matrix
    assert np.all(Q == bar_magnet.Q)
    assert np.allclose(Q.dot(np.array([0., 0., 1.])) - bar_magnet.M, 0.)

    x_M = np.random.rand(3)  # set new center of mass
    bar_magnet.x_M = x_M
    assert np.all(x_M == bar_magnet.x_M)

    # update both at the same time
    Q = Rotation.from_rotvec(np.random.rand(3)).as_matrix()
    x_M = np.random.rand(3)
    bar_magnet.update_parameters(x_M, Q)
    assert np.all(x_M == bar_magnet.x_M)
    assert np.all(Q == bar_magnet.Q)
    assert np.allclose(Q.dot(np.array([0., 0., 1.])) - bar_magnet.M, 0.) 

    # delete bar magnet
    del bar_magnet

def test_magnet_segment():
    # create segment
    w = 2.0
    d = 3.0
    Rm = 10.0
    M0 = 20.
    Q = np.eye(3)
    x_M = np.random.rand(3)
    magnet_segment = CylinderSegment(radius=Rm, width=w, depth=d, alpha=np.pi / 4, magnetization_strength=M0, \
        position_vector=x_M, rotation_matrix=Q)

    assert magnet_segment.type == "CylinderSegment"
    assert magnet_segment.w == w
    assert magnet_segment.d == d
    assert magnet_segment.Rm == Rm
    assert np.all(magnet_segment.x_M == x_M)
    assert magnet_segment.M0 == M0
    assert np.all(magnet_segment.Q == Q)

    # check basic functionality
    assert callable(magnet_segment.Vm_eigen)
    Vm_test = magnet_segment.Vm_eigen(np.random.rand(3))
    assert isinstance(Vm_test, float)

    assert callable(magnet_segment.Vm)
    Vm_test = magnet_segment.Vm(np.random.rand(3))
    assert isinstance(Vm_test, float)

    assert callable(magnet_segment.B_eigen)
    B_test = magnet_segment.B_eigen(np.random.rand(3))
    assert isinstance(B_test, np.ndarray)
    assert len(B_test) == 3

    assert callable(magnet_segment.B)
    B_test = magnet_segment.B(np.random.rand(3))
    assert isinstance(B_test, np.ndarray)
    assert len(B_test) == 3

    # update magnet
    Q = Rotation.from_rotvec(np.random.rand(3)).as_matrix()
    assert np.allclose(Q.dot(Q.T), np.eye(3))

    magnet_segment.Q = Q  # set new rotation matrix
    assert np.all(Q == magnet_segment.Q)

    x_M = np.random.rand(3)  # set new center of mass
    magnet_segment.x_M = x_M
    assert np.all(x_M == magnet_segment.x_M)

    # update both at the same time
    Q = Rotation.from_rotvec(np.random.rand(3)).as_matrix()
    x_M = np.random.rand(3)
    magnet_segment.update_parameters(x_M, Q)
    assert np.all(x_M == magnet_segment.x_M)
    assert np.all(Q == magnet_segment.Q)

    # delete bar magnet
    del magnet_segment


if __name__ == "__main__":
    test_ball_magnet()
    test_bar_magnet()
    test_magnet_segment()