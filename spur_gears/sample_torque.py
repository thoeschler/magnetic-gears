import dolfin as dlf
import numpy as np
import pandas as pd
import json
import os
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear, SegmentGear, MagneticGear
from spur_gears.spur_gears_problem import SpurGearsProblem
from source.grid_generator import gear_mesh


class SpurGearsSampling(SpurGearsProblem):
    def __init__(self, first_gear, second_gear, D, main_dir=None):
        super().__init__(first_gear, second_gear, D, main_dir)
        self.align_gears()
        assert self.gear_1.angle == 0.
        assert self.gear_2.angle == 0.
        self.set_mesh_functions()

    def set_mesh_functions(self):
        """Set mesh function for both gears."""
        for gear in self.gears:
            gear.set_mesh_function(gear_mesh)

    def mesh_gear(self, gear, mesh_size, fname, write_to_pvd=True):
        self.create_gear_mesh(gear, mesh_size=mesh_size, fname=fname, \
                              write_to_pvd=write_to_pvd)

    def update_gear(self, gear: MagneticGear, d_angle):
        """Update gear given an angle increment.

        Args:
            gear (MagneticGear): A magnetic gear.
            d_angle (float): Angle increment.
            update_segment (bool, optional): If true update the segment as well. Defaults to False.
        """
        gear.update_parameters(d_angle)
        if gear is self.lg:
            # update segment
            self.segment_mesh.rotate(d_angle * 180. / np.pi, 0, dlf.Point(gear.x_M))

    def sample(self, n_iterations, p_deg=2):
        """
        Sample torque values.

        Args:
            n_iterations (int): Number of torque samples (per angle).
            p_deg (int, optional): Polynomial degree used for computation. Defaults to 2.
        """
        angles_1 = np.linspace(0., 2. * np.pi / self.gear_1.n, num=n_iterations, endpoint=False)
        angles_2 = np.linspace(0., 2. * np.pi / self.gear_2.n, num=n_iterations, endpoint=False)
        d_angle_1 = angles_1[1]- angles_1[0]
        d_angle_2 = angles_2[1]- angles_2[0]

        # assign angle increments
        if self.gear_1 is self.sg:
            angles_sg = angles_1
            angles_lg = angles_2
            d_angle_sg = d_angle_1
            d_angle_lg = d_angle_2
        elif self.gear_2 is self.sg:
            angles_sg = angles_2
            angles_lg = angles_1
            d_angle_sg = d_angle_2
            d_angle_lg = d_angle_1
        else:
            raise RuntimeError()

        tau_12_values = np.zeros((n_iterations, n_iterations))
        tau_21_values = np.zeros((n_iterations, n_iterations))
        for i in range(n_iterations):
            for j in range(n_iterations):
                # check angles
                assert np.isclose(self.sg.angle, angles_sg[j])
                assert np.isclose(self.lg.angle, angles_lg[i])
                F_sg, tau_sg = self.compute_force_torque(p_deg, use_Vm=True)
                tau_lg = (np.cross(self.lg.x_M - self.sg.x_M, F_sg) - tau_sg)[0]

                if self.gear_1 is self.sg:
                    tau_21 = tau_sg
                    tau_12 = tau_lg
                elif self.gear_2 is self.sg:
                    tau_21 = tau_lg
                    tau_12 = tau_sg
                else:
                    raise RuntimeError()

                tau_12_values[i, j] = tau_12
                tau_21_values[i, j] = tau_21

                # rotate smaller gear
                self.update_gear(self.sg, d_angle=d_angle_sg)

            # rotate back to zero angle
            self.update_gear(self.sg, d_angle=float(-self.sg.angle))
            assert np.isclose(self.sg.angle, 0.)

            # update csv files
            pd.DataFrame(tau_12_values, index=angles_1, columns=angles_2).to_csv(f"{self._main_dir}/data/tau_12.csv")
            pd.DataFrame(tau_21_values, index=angles_1, columns=angles_2).to_csv(f"{self._main_dir}/data/tau_21.csv")

            # rotate larger gear
            self.update_gear(self.lg, d_angle=d_angle_lg)

def sample_torque_ball(n_iterations, mesh_size, p_deg=2, interpolate="twice", sample_dir="sample_ball_gear"):
    # create directory
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # create two ball gears
    gear_1_ball = MagneticBallGear(n=12, r=1., R=6, x_M=np.zeros(3))
    gear_2_ball = MagneticBallGear(n=18, r=1.5, R=9., x_M=np.array([0., 1., 0.]))
    gear_1_ball.create_magnets(magnetization_strength=1.0)
    gear_2_ball.create_magnets(magnetization_strength=1.0)

    # create coaxial gears problem
    D = gear_1_ball.outer_radius + gear_2_ball.outer_radius + 1.0
    bg = SpurGearsSampling(gear_1_ball, gear_2_ball, D, main_dir=sample_dir)

    # mesh smaller gear
    bg.mesh_gear(bg.sg, mesh_size=mesh_size, fname=f"gear_{1 if bg.sg is bg.gear_1 else 2}")

    # mesh the segment
    bg.mesh_reference_segment(mesh_size)
    bg.interpolate_to_reference_segment(mesh_size, p_deg=p_deg, interpolate=interpolate)

    write_paramter_file(bg, sample_dir)

    bg.sample(n_iterations, p_deg=p_deg)


def sample_torque_bar(n_iterations, mesh_size, p_deg=2, interpolate="twice", sample_dir="sample_bar_gear"):
    # create test directory
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    gear_1_bar = MagneticBarGear(n=2, h=2.0, w=0.75, d=0.5, R=12, x_M=np.zeros(3))
    gear_2_bar = MagneticBarGear(n=4, h=2.0, w=0.75, d=0.5, R=18, x_M=np.array([0., 1., 0.]))
    gear_1_bar.create_magnets(magnetization_strength=1.0)
    gear_2_bar.create_magnets(magnetization_strength=1.0)

    # create coaxial gears problem
    D = gear_1_bar.outer_radius + gear_2_bar.outer_radius + 1.0
    bg = SpurGearsSampling(gear_1_bar, gear_2_bar, D, main_dir=sample_dir)

    # mesh smaller gear
    bg.mesh_gear(bg.sg, mesh_size=mesh_size, fname=f"gear_{1 if bg.sg is bg.gear_1 else 2}")

    # mesh the segment
    bg.mesh_reference_segment(mesh_size)
    bg.interpolate_to_reference_segment(mesh_size, p_deg=p_deg, interpolate=interpolate)

    write_paramter_file(bg, sample_dir)

    bg.sample(n_iterations, p_deg=p_deg)

def sample_torque_segment(n_iterations, mesh_size, p_deg=2, interpolate="twice", sample_dir="sample_segment_gear"):
    sample_dir = "sample_segment_gear"
    # create test directory
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    gear_1_segment = SegmentGear(n=12, R=6.0, w=0.75, t=1., x_M=np.zeros(3))
    gear_2_segment = SegmentGear(n=4, R=6.0, w=0.75, t=1., x_M=np.array([0., 1., 0.]))
    gear_1_segment.create_magnets(magnetization_strength=1.0)
    gear_2_segment.create_magnets(magnetization_strength=1.0)

    # create coaxial gears problem
    D = gear_1_segment.outer_radius + gear_2_segment.outer_radius + 1.0
    sg = SpurGearsSampling(gear_1_segment, gear_2_segment, D, main_dir=sample_dir)

    # mesh smaller gear
    sg.mesh_gear(sg.sg, mesh_size=mesh_size, fname=f"gear_{1 if sg.sg is sg.gear_1 else 2}")

    # mesh the segment
    sg.mesh_reference_segment(mesh_size)
    sg.interpolate_to_reference_segment(mesh_size, p_deg=p_deg, interpolate=interpolate)

    write_paramter_file(sg, sample_dir)

    sg.sample(n_iterations, p_deg=p_deg)

def write_paramter_file(problem, dir_):
    with open(dir_ + "/par.json", "w") as f:
        par = {
            "gear_1": problem.gear_1.parameters,
            "gear_2": problem.gear_2.parameters,
            "D": problem.D
        }
        f.write(json.dumps(par))

if __name__ == "__main__":
    #sample_torque_ball(n_iterations=7, mesh_size=0.4, p_deg=2, interpolate="twice")
    #sample_torque_bar(n_iterations=20, mesh_size=0.1, p_deg=2, interpolate="twice")
    sample_torque_segment(n_iterations=5, mesh_size=0.2, p_deg=2, interpolate="twice")
