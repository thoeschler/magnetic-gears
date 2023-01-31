import numpy as np
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear
from coaxial.coaxial_gears import CoaxialGearsProblem
from coaxial.grid_generator import ball_gear_mesh, bar_gear_mesh
import subprocess
import os


class CoaxialGearsTest(CoaxialGearsProblem):
    def __init__(self, first_gear, second_gear, D, main_dir=None):
        super().__init__(first_gear, second_gear, D, main_dir)
        self.align_gears()
        self.set_mesh_functions()

    def set_mesh_functions(self):
        """Set mesh function for both gears."""
        for gear in self.gears:
            if isinstance(gear, MagneticBallGear):
                gear.set_mesh_function(ball_gear_mesh)
            elif isinstance(gear, MagneticBarGear):
                gear.set_mesh_function(bar_gear_mesh)
            else:
                raise RuntimeError()

    def mesh_gear(self, gear, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd=True):
        other_gear = self.gear_2 if gear is self.gear_1 else self.gear_1
        self.create_gear_mesh(gear, x_M_ref=other_gear.x_M, mesh_size_space=mesh_size_space, \
            mesh_size_magnets=mesh_size_magnets, fname=fname, padding=padding, \
                write_to_pvd=write_to_pvd)

    def simulate(self, n_iterations=10):
        for _ in range(n_iterations):
            d_alpha = 2. * np.pi / self.gear_1.n / n_iterations
            B1 = self.interpolate_field_gear(self.gear_1, self.gear_2, "B", "CG", 1, 0.6, 8.0)
            B2 = self.interpolate_field_gear(self.gear_2, self.gear_1, "B", "CG", 1, 0.6, 8.0)

            tau1 = self.compute_torque_on_gear(self.gear_1, B2)
            tau2 = self.compute_torque_on_gear(self.gear_2, B1)

            with open(test_dir + "/test_coaxial_gears.csv", "a+") as f:
                f.write(f"{self.gear_1.angle} {tau1} {tau2}\n")
            self.gear_1.update_parameters(d_alpha)

def test_ball_gear_problem(test_dir):
    # create test directory
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    gear_1_ball = MagneticBallGear(n=5, r=1.5, R=10, x_M=np.zeros(3), axis=np.array([1., 0., 0.]))
    gear_2_ball = MagneticBallGear(n=7, r=2.0, R=15, x_M=np.array([0., 3., 0.]), axis=np.array([1., 0., 0.]))
    gear_1_ball.create_magnets(magnetization_strength=1.0)
    gear_2_ball.create_magnets(magnetization_strength=2.0)

    bg = CoaxialGearsTest(gear_1_ball, gear_2_ball, D=30, main_dir=test_dir)
    bg.mesh_gear(gear_1_ball, mesh_size_space=3.0, mesh_size_magnets=0.5, \
        fname="gear_1", padding=gear_1_ball.R / 10)
    bg.mesh_gear(gear_2_ball, mesh_size_space=2.0, mesh_size_magnets=0.3, \
        fname="gear_2", padding=gear_2_ball.R / 10)

    bg.simulate(n_iterations=2)

    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)

def test_bar_gear_problem(test_dir):
    # create test directory
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    gear_1_bar = MagneticBarGear(n=4, h=1.5, w=1.0, d=1.0, R=10, x_M=np.zeros(3), axis=np.array([1., 0., 0.]))
    gear_2_bar = MagneticBarGear(n=5, h=1.5, w=1.0, d=1.0, R=15, x_M=np.array([0., 3., 0.]), axis=np.array([1., 0., 0.]))
    gear_1_bar.create_magnets(magnetization_strength=1.0)
    gear_2_bar.create_magnets(magnetization_strength=2.0)

    bg = CoaxialGearsTest(gear_1_bar, gear_2_bar, D=30, main_dir=test_dir)
    bg.mesh_gear(gear_1_bar, mesh_size_space=3.0, mesh_size_magnets=0.5, \
        fname="gear_1", padding=gear_1_bar.R / 10)
    bg.mesh_gear(gear_2_bar, mesh_size_space=2.0, mesh_size_magnets=0.3, \
        fname="gear_2", padding=gear_2_bar.R / 10)

    bg.simulate(n_iterations=2)

    # remove test directory
    #subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)

if __name__ == "__main__":
    test_dir = os.getcwd() + "/test_dir"
    #test_ball_gear_problem(test_dir)
    test_bar_gear_problem(test_dir)