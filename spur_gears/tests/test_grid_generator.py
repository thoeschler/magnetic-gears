from spur_gears.grid_generator import gear_mesh, segment_mesh
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear, SegmentGear
import numpy as np
import os
import subprocess


def test_ball_gear_mesh():
    n = 4
    R = 10.
    r = 1.
    x_M = np.zeros(3)

    ball_gear = MagneticBallGear(n, R, r, x_M)
    ball_gear.create_magnets(magnetization_strength=1.0)

    mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags = gear_mesh(
        ball_gear, x_M_ref=np.array([0., 20., 0.]), mesh_size_magnets=0.4, \
            fname="ballgeartest", write_to_pvd=True, verbose=True)

def test_bar_gear_mesh():
    n = 4
    R = 10.
    d = 1.
    w = 1.
    h = 3.
    x_M = np.zeros(3)

    bar_gear = MagneticBarGear(n, R, d, w, h, x_M)
    bar_gear.create_magnets(magnetization_strength=1.0)

    mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags = gear_mesh(
        bar_gear, x_M_ref=np.array([0., 20., 0.]), mesh_size_magnets=0.4, \
            fname="bargeartest", write_to_pvd=True, verbose=True)

def test_segment_gear_mesh():
    n = 4
    R = 10.
    d = 1.
    w = 1.
    x_M = np.zeros(3)

    segment_gear = SegmentGear(n, R, d, w, x_M)
    segment_gear.create_magnets(magnetization_strength=1.0)

    mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags = gear_mesh(
        segment_gear, x_M_ref=np.array([0., 20., 0.]), mesh_size_magnets=0.4, \
            fname="segmentgeartest", write_to_pvd=True, verbose=True)

def test_segment_mesh():
    _ = segment_mesh(10, 20, 4, np.pi / 4, np.zeros(3), np.array([0., 1., 1.]), \
                        fname="segmenttest", padding=1., mesh_size=0.8, write_to_pvd=True)

if __name__ == "__main__":
    test_dir = "test_grid_generator"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    os.chdir(test_dir)

    test_ball_gear_mesh()
    test_bar_gear_mesh()
    test_segment_gear_mesh()
    test_segment_mesh()

    os.chdir("..")
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)