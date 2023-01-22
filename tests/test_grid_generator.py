from source.grid_generator import bar_gear_mesh, ball_gear_mesh
from source.magnetic_gear_classes import MagneticGearWithBallMagnets, MagneticGearWithBarMagnets
import numpy as np
import subprocess
import os

# create test directory
test_dir = "test_dir"
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
os.chdir(test_dir)

out_path = "data/gear/meshes/"
if not os.path.exists(out_path):
    os.makedirs(out_path)

ball_gear = MagneticGearWithBallMagnets(4, 1, 10, np.zeros(3), 1., 0., 1)
out = ball_gear_mesh(ball_gear, mesh_size_space=2.0, mesh_size_magnets=0.3, padding=ball_gear.R / 10, \
    fname=out_path + "testball", verbose=True)


bar_gear = MagneticGearWithBarMagnets(4, 1, 1, 1, 10, np.zeros(3), \
    1., 0., 0)
out = bar_gear_mesh(bar_gear, mesh_size_space=2.0, mesh_size_magnets=1.0, padding=bar_gear.R / 10, \
    fname=out_path + "testbar", verbose=True)

# remove test directory
os.chdir("..")
subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)
