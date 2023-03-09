import numpy as np
from scipy.spatial.transform import Rotation
import dolfin as dlf
from dolfin import LagrangeInterpolator
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear
from spur_gears.spur_gears_problem import CoaxialGearsProblem
from spur_gears.grid_generator import gear_mesh, segment_mesh
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
            gear.set_mesh_function(gear_mesh)

    def mesh_gear(self, gear, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd=True):
        if gear is self.gear_1:
            other_gear = self.gear_2
        elif gear is self.gear_2:
            other_gear = self.gear_1
        else:
            raise RuntimeError()
        self.create_gear_mesh(gear, x_M_ref=other_gear.x_M, mesh_size_space=mesh_size_space, \
            mesh_size_magnets=mesh_size_magnets, fname=fname, padding=padding, \
                write_to_pvd=write_to_pvd)

    def interpolate_to_segment(self):
        # choose smaller gear ("sg")
        if self.gear_1.R < self.gear_2.R:
            self.sg = self.gear_1  # smaller gear
            self.lg = self.gear_2  # larger gear
        else:
            self.sg = self.gear_2  # smaller gear
            self.lg = self.gear_2  # other gear

        # set geometrical paramters        
        if isinstance(self.sg, MagneticBallGear):
            pad = self.sg.r * (1. + 1e-1)
            t = 2 * self.sg.r
        elif isinstance(self.sg, MagneticBarGear):
            pad = max(self.sg.h, self.sg.w)
            t = 2 * self.sg.d
        else:
            raise RuntimeError()

        # angle to contain smaller gear
        angle = np.abs(2 * np.arccos(1 - 0.5 * (self.sg.outer_radius / self.D) ** 2))
        angle += 2 * np.pi / self.lg.n  # allow rotation by +- pi / n

        # create segment mesh
        Ri = self.D - self.sg.outer_radius
        alpha_r = np.pi / self.sg.n * (self.sg.n - len(self.sg.magnets))
        assert alpha_r > 0
        x_axis = np.array([0., 1., 0.])
        if self.sg.x_M[1] < self.lg.x_M[1]:
            x_axis *= -1
        Ro = np.sqrt((self.D + self.sg.R * np.cos(alpha_r)) ** 2 + (self.sg.R * np.sin(alpha_r)) ** 2)
        self.seg_mesh, _, _ = segment_mesh(Ri=Ri, Ro=Ro, t=t, angle=angle, x_M_ref=self.lg.x_M, \
                                           x_axis=x_axis, fname="reference_segment", padding=pad, \
                                            write_to_pvd=True)
        # interpolate field of other gear on segment
        self.B_ref = self.interpolate_field_gear(self.lg, self.sg, self.seg_mesh, "B", "CG", 1, 0.6, 8.0)

    def simulate(self, n_iterations=10):
        d_alpha = np.pi / self.gear_1.n / n_iterations
        for _ in range(n_iterations):
            # interpolate reference field from segment to gear
            V = dlf.VectorFunctionSpace(self.sg.mesh, "CG", 1)
            B_lg = dlf.Function(V)
            LagrangeInterpolator.interpolate(B_lg, self.B_ref)

            # compute force
            F = self.compute_force_on_gear(self.sg, B_lg)

            # compute torque
            tau_sg = self.compute_torque_on_gear(self.sg, B_lg)
            F_pad = np.array([F[0], 0., F[1]])
            tau_bg = (np.cross((self.sg.x_M - self.lg.x_M), F_pad) - tau_sg)[0]

            with open(test_dir + "/test_coaxial_gears.csv", "a+") as f:
                f.write(f"{self.sg.angle} {self.lg.angle} {tau_sg} {tau_bg} \n")
            self.sg.update_parameters(d_alpha)

def test_ball_gear_problem(test_dir):
    # create test directory
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    gear_1_ball = MagneticBallGear(n=12, r=1.5, R=10, x_M=np.zeros(3))
    gear_2_ball = MagneticBallGear(n=14, r=2.0, R=15, x_M=np.array([0., 3., 0.]))
    gear_1_ball.create_magnets(magnetization_strength=1.0)
    gear_2_ball.create_magnets(magnetization_strength=1.0)

    bg = CoaxialGearsTest(gear_1_ball, gear_2_ball, D=35, main_dir=test_dir)
    bg.mesh_gear(gear_1_ball, mesh_size_space=3.0, mesh_size_magnets=0.5, \
        fname="gear_1", padding=gear_1_ball.r)
    bg.mesh_gear(gear_2_ball, mesh_size_space=2.0, mesh_size_magnets=0.3, \
        fname="gear_2", padding=gear_2_ball.r)

    bg.interpolate_to_segment()
    bg.simulate(n_iterations=5)

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

    bg.simulate(n_iterations=5)

    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)

if __name__ == "__main__":
    test_dir = os.getcwd() + "/test_dir"
    test_ball_gear_problem(test_dir)
    test_bar_gear_problem(test_dir)