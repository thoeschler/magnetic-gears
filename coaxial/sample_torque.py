import numpy as np
import dolfin as dlf
from dolfin import LagrangeInterpolator
from coaxial.coaxial_gears import CoaxialGearsProblem
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear, SegmentGear
from coaxial.grid_generator import gear_mesh, segment_mesh
import json
import os


class CoaxialGears(CoaxialGearsProblem):
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

    def interpolate_to_segment(self, mesh_size_magnets):
        # choose smaller gear ("sg")
        if self.gear_1.R < self.gear_2.R:
            self.sg = self.gear_1  # smaller gear
            self.lg = self.gear_2  # larger gear
        else:
            self.sg = self.gear_2  # smaller gear
            self.lg = self.gear_1  # larger gear

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
        if self.sg.n > len(self.sg.magnets):
            # "-1" to allow rotation by +- pi / n
            # => assume one magnet less has been removed
            alpha_r = np.pi / self.sg.n * (self.sg.n - len(self.sg.magnets) - 1)
        else:
            alpha_r = 0.
        assert alpha_r > 0
        x_axis = np.array([0., 1., 0.])
        if self.sg.x_M[1] < self.lg.x_M[1]:
            x_axis *= -1
        Ro = np.sqrt((self.D + self.sg.R * np.cos(alpha_r)) ** 2 + (self.sg.R * np.sin(alpha_r)) ** 2)

        ref_path = self._main_dir + "/data/reference"
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        self.seg_mesh, _, _ = segment_mesh(Ri=Ri, Ro=Ro, t=t, angle=angle, x_M_ref=self.lg.x_M, \
                                           x_axis=x_axis, fname=ref_path + "/reference_segment", padding=pad, \
                                           mesh_size=mesh_size_magnets, write_to_pvd=True)
        # interpolate field of other gear on segment
        # size of basic reference field
        self.set_domain_size(self.D + self.gear_1.R + self.gear_2.R + pad)
        self.B_ref = self.interpolate_field_gear(self.lg, self.seg_mesh, "B", "CG", 1, \
                                                 mesh_size_magnets / max(self.gear_1.scale_parameter, \
                                                                         self.gear_2.scale_parameter), 8.0)

    def update_gear(self, gear, d_angle, segment_mesh=None):
        gear.update_parameters(d_angle)
        if segment_mesh is not None:
            assert segment_mesh is self.seg_mesh
            segment_mesh.rotate(d_angle * 180. / np.pi, 0, dlf.Point(gear.x_M))
        if np.abs(gear.angle) > np.pi / gear.n:
            # first rotate segment
            if segment_mesh is not None:
                segment_mesh.rotate(- np.sign(gear.angle) * 360. / gear.n, 0, dlf.Point(gear.x_M))
            # then update parameters
            gear.update_parameters(- np.sign(gear.angle) * 2. * np.pi / gear.n)

    def simulate(self, n_iterations, data_fname):
        if self.gear_1 is self.sg:
            d_angle_sg = 2. * np.pi / self.gear_1.n / n_iterations
            d_angle_lg = 2. * np.pi / self.gear_2.n / n_iterations
        else:
            d_angle_sg = 2. * np.pi / self.gear_2.n / n_iterations
            d_angle_lg = 2. * np.pi / self.gear_1.n / n_iterations
        for i in range(n_iterations):
            for _ in range(n_iterations):
                # interpolate reference field from segment to gear
                V = dlf.VectorFunctionSpace(self.sg.mesh, "CG", 1)
                B_lg = dlf.Function(V)

                # copy reference_field
                segment_mesh_copy = dlf.Mesh(self.seg_mesh)
                V_ref = dlf.VectorFunctionSpace(segment_mesh_copy, "CG", 1)
                B_ref = dlf.Function(V_ref, self.B_ref._cpp_object.vector())
                LagrangeInterpolator.interpolate(B_lg, B_ref)

                # compute force
                F = self.compute_force_on_gear(self.sg, B_lg)

                # compute torque
                tau_sg = self.compute_torque_on_gear(self.sg, B_lg)
                F_pad = np.array([0., F[0], F[1]])
                tau_lg = (np.cross((self.sg.x_M - self.lg.x_M), F_pad) - tau_sg)[0]

                if self.gear_1 is self.sg:
                    tau_1 = tau_sg
                    tau_2 = tau_lg
                else:
                    tau_1 = tau_lg
                    tau_2 = tau_sg

                with open(self._main_dir + f"/{data_fname}.csv", "a+") as f:
                    f.write(f"{self.gear_1.angle} {self.gear_2.angle} {tau_1} {tau_2} \n")
                # rotate smaller gear
                self.update_gear(self.sg, d_angle=d_angle_sg)

            # rotate smaller gear back
            self.update_gear(self.sg, d_angle=-n_iterations * d_angle_sg)
            # rotate larger gear
            self.update_gear(self.lg, segment_mesh=self.seg_mesh, d_angle=d_angle_lg)

            dlf.File(f"segment_mesh_test_{i}.pvd") << self.B_ref

def sample_torque_ball(n_iterations):
    # create directory
    sample_dir = "sample_ball_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    gear_1_ball = MagneticBallGear(n=15, r=1.5, R=10, x_M=np.zeros(3))
    gear_2_ball = MagneticBallGear(n=17, r=2.0, R=15, x_M=np.array([0., 1., 0.]))
    gear_1_ball.create_magnets(magnetization_strength=1.0)
    gear_2_ball.create_magnets(magnetization_strength=1.0)

    mesh_size_magnets = 0.6
    bg = CoaxialGears(gear_1_ball, gear_2_ball, D=33, main_dir=sample_dir)
    bg.mesh_gear(gear_1_ball, mesh_size_space=2.0, mesh_size_magnets=mesh_size_magnets, \
        fname="gear_1", padding=gear_1_ball.r / 2)
    bg.mesh_gear(gear_2_ball, mesh_size_space=2.0, mesh_size_magnets=mesh_size_magnets, \
        fname="gear_2", padding=gear_2_ball.r / 2)

    write_paramter_file(bg, sample_dir)

    bg.interpolate_to_segment(mesh_size_magnets)
    bg.simulate(n_iterations, data_fname="torque_ball_gears")


def sample_torque_bar(n_iterations):
    sample_dir = "sample_bar_gear"
    # create test directory
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    gear_1_bar = MagneticBarGear(n=12, h=2.0, w=0.75, d=0.5, R=12, x_M=np.zeros(3))
    gear_2_bar = MagneticBarGear(n=15, h=2.0, w=0.75, d=0.5, R=18, x_M=np.array([0., 1., 0.]))
    gear_1_bar.create_magnets(magnetization_strength=1.0)
    gear_2_bar.create_magnets(magnetization_strength=1.0)

    mesh_size_magnets = 0.1
    bg = CoaxialGears(gear_1_bar, gear_2_bar, D=35, main_dir=sample_dir)
    bg.mesh_gear(gear_1_bar, mesh_size_space=2.0, mesh_size_magnets=mesh_size_magnets, \
        fname="gear_1", padding=gear_1_bar.h / 2)
    bg.mesh_gear(gear_2_bar, mesh_size_space=2.0, mesh_size_magnets=mesh_size_magnets, \
        fname="gear_2", padding=gear_2_bar.h / 2)

    write_paramter_file(bg, sample_dir)

    bg.interpolate_to_segment()
    bg.simulate(n_iterations, data_fname="torque_bar_gears")

def sample_torque_segment(n_iterations):
    sample_dir = "sample_segment_gear"
    # create test directory
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    gear_1_bar = SegmentGear(n=12, R=10.0, w=0.75, d=0.5, x_M=np.zeros(3))
    gear_2_bar = SegmentGear(n=15, R=12.0, w=0.75, d=0.5, x_M=np.array([0., 1., 0.]))
    gear_1_bar.create_magnets(magnetization_strength=1.0)
    gear_2_bar.create_magnets(magnetization_strength=1.0)

    mesh_size_magnets = 0.1
    sg = CoaxialGears(gear_1_bar, gear_2_bar, D=30, main_dir=sample_dir)
    sg.mesh_gear(gear_1_bar, mesh_size_space=2.0, mesh_size_magnets=mesh_size_magnets, \
        fname="gear_1", padding=gear_1_bar.w / 2)
    sg.mesh_gear(gear_2_bar, mesh_size_space=2.0, mesh_size_magnets=mesh_size_magnets, \
        fname="gear_2", padding=gear_2_bar.w / 2)

    write_paramter_file(sg, sample_dir)

    sg.interpolate_to_segment()
    sg.simulate(n_iterations, data_fname="torque_bar_gears")

def write_paramter_file(problem, dir_):
    with open(dir_ + "/par.json", "w") as f:
        f.write(json.dumps(problem.gear_1.parameters))
        f.write(json.dumps(problem.gear_2.parameters))
        f.write(json.dumps(problem.D))


if __name__ == "__main__":
    sample_torque_ball(n_iterations=5)
    #sample_torque_bar(n_iterations=2)
    #sample_torque_segment(n_iterations=2)