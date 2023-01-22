from source.magnetic_gear_classes import MagneticGearWithBallMagnets, MagneticGearWithBarMagnets
import numpy as np
import subprocess
import os

# create test directory
test_dir = "test_dir"
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
os.chdir(test_dir)

out_dir = "data/gears/meshes/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

ball_gear = MagneticGearWithBallMagnets(4, 1, 10, np.zeros(3), 1., 0., 1)
ball_gear.create_mesh(1.0, 0.3, fname=out_dir + "balltest", padding=ball_gear.R / 10)

bar_gear = MagneticGearWithBarMagnets(4, 1, 1, 1, 10, np.zeros(3), 1., 0., 2)
bar_gear.create_mesh(1.0, 0.3, fname=out_dir + "bartest", padding=bar_gear.R / 10)

# remove test directory
os.chdir("..")
subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)