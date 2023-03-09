import numpy as np
import dolfin as dlf
import json
import os
from dolfin import LagrangeInterpolator
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear, SegmentGear, MagneticGear
from source.tools.tools import interpolate_field
from source.tools.fenics_tools import compute_magnetic_field
from spur_gears.spur_gears_problem import SpurGearsProblem
from spur_gears.grid_generator import gear_mesh, segment_mesh


class SpurGearsSampling(SpurGearsProblem):
    def __init__(self, first_gear, second_gear, D, main_dir=None):
        super().__init__(first_gear, second_gear, D, main_dir)
        self.align_gears()
        self.set_mesh_functions()

    def assign_gears(self):
        if self.gear_1.R < self.gear_2.R:
            self.sg = self.gear_1  # smaller gear
            self.lg = self.gear_2  # larger gear
        else:
            self.sg = self.gear_2  # smaller gear
            self.lg = self.gear_1  # larger gear

    def set_mesh_functions(self):
        """Set mesh function for both gears."""
        for gear in self.gears:
            gear.set_mesh_function(gear_mesh)

    def mesh_gear(self, gear, mesh_size_magnets, fname, write_to_pvd=True):
        if gear is self.gear_1:
            other_gear = self.gear_2
        elif gear is self.gear_2:
            other_gear = self.gear_1
        else:
            raise RuntimeError()
        self.create_gear_mesh(gear, x_M_ref=other_gear.x_M, \
            mesh_size_magnets=mesh_size_magnets, fname=fname, write_to_pvd=write_to_pvd)

    def interpolate_to_segment(self, mesh_size_magnets, p_deg=1, interpolate="once"):
        ################################################
        ############# create segment mesh ##############
        ################################################

        # set geometrical paramters
        assert isinstance(self.sg, MagneticBallGear)
        assert isinstance(self.lg, MagneticBallGear)
        pad = self.sg.r
        t = 2 * self.sg.r

        # angle to contain smaller gear
        angle = np.abs(2 * np.arccos(1 - 0.5 * (self.sg.outer_radius / self.D) ** 2))
        angle += 2 * np.pi / self.lg.n  # allow rotation by +- pi / n
        if angle > 2 * np.pi:  # make sure angle is at most 2 pi
            angle = 2 * np.pi

        # inner segment radius
        Ri = self.D - self.sg.outer_radius

        # set angle of segment
        # "-1" to allow rotation by +- pi / n
        # => assume one magnet less has been removed
        if self.sg.n > len(self.sg.magnets):
            alpha_r = np.pi / self.sg.n * (self.sg.n - len(self.sg.magnets) - 1)
        else:
            alpha_r = 0.

        # x_axis of segment
        if self.sg.x_M[1] > self.lg.x_M[1]:
            x_axis = np.array([0., 1., 0.])
        else:
            x_axis = np.array([0., -1., 0.])
        
        # outer segment radius
        Ro = np.sqrt((self.D + self.sg.R * np.cos(alpha_r)) ** 2 + (self.sg.R * np.sin(alpha_r)) ** 2)

        ref_path = self._main_dir + "/data/reference"
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)

        self.seg_mesh, _, _ = segment_mesh(Ri=Ri, Ro=Ro, t=t, angle=angle, x_M_ref=self.lg.x_M, \
                                        x_axis=x_axis, fname=ref_path + "/reference_segment", padding=pad, \
                                        mesh_size=mesh_size_magnets, write_to_pvd=True)

        ################################################
        ######### interpolate to segment mesh ##########
        ################################################
        if interpolate == "once":
            V = dlf.FunctionSpace(self.seg_mesh, "CG", p_deg)
            Vm_ref_vec = 0.
            for mag in self.lg.magnets:
                Vm_mag = interpolate_field(mag.Vm_as_expression(), self.seg_mesh, "CG", p_deg)
                Vm_ref_vec += Vm_mag.vector()
            self.Vm_ref = dlf.Function(V, Vm_ref_vec)
        elif interpolate == "twice":
            # interpolate fields of other gear on segment
            self.Vm_ref = self.interpolate_field_gear(self.lg, self.seg_mesh, "Vm", "DG", p_deg, \
                                                    mesh_size_magnets / max(self.gear_1.scale_parameter, \
                                                                            self.gear_2.scale_parameter), 3 * mesh_size_magnets)

    def compute_force_torque_numerically(self, p_deg=1):
        assert hasattr(self, "seg_mesh")
        assert hasattr(self, "Vm_ref")
        # create function on smaller gear
        V = dlf.FunctionSpace(self.sg.mesh, "CG", p_deg)
        Vm_lg = dlf.Function(V)

        LagrangeInterpolator.interpolate(Vm_lg, self.Vm_ref)
        B_lg = compute_magnetic_field(Vm_lg, p_deg)
        dlf.File("Vm_ref.pvd") << self.Vm_ref
        dlf.File("Vm_lg.pvd") << Vm_lg
        dlf.File("B_lg.pvd") << B_lg

        # compute force
        F = self.compute_force_on_gear(self.sg, B_lg)

        # compute torque
        tau_sg = self.compute_torque_on_gear(self.sg, B_lg)
        F_pad = np.array([0., F[0], F[1]])  # pad force (insert x-component)

        return F_pad, tau_sg

    def update_gear(self, gear: MagneticGear, d_angle, update_segment=False):
        """Update gear given an angle increment.

        Args:
            gear (MagneticGear): A magnetic gear.
            d_angle (float): Angle increment.
            update_segment (bool, optional): If true update the segment as well. Defaults to False.
        """
        gear.update_parameters(d_angle)
        if update_segment:
            self.seg_mesh.rotate(d_angle * 180. / np.pi, 0, dlf.Point(gear.x_M))

        while np.abs(gear.angle) > np.pi / gear.n:
            # first rotate segment
            if update_segment:
                # rotate back: 
                self.seg_mesh.rotate(- np.sign(gear.angle) * 360. / gear.n, 0, dlf.Point(gear.x_M))

            # then update parameters
            gear.update_parameters(- np.sign(gear.angle) * 2. * np.pi / gear.n)

    def sample(self, n_iterations, p_deg, data_fname, interpolate="once"):
        if self.gear_1 is self.sg:
            d_angle_sg = 2. * np.pi / self.gear_1.n / n_iterations
            d_angle_lg = 2. * np.pi / self.gear_2.n / n_iterations
        else:
            d_angle_sg = 2. * np.pi / self.gear_2.n / n_iterations
            d_angle_lg = 2. * np.pi / self.gear_1.n / n_iterations
        for i in range(n_iterations):
            for j in range(n_iterations):
                if i == 1 and j == 1:
                    dlf.File("Vm_ref.pvd") << self.Vm_ref
                    dlf.File("segm_mesh.pvd") << self.seg_mesh
                    dlf.File("gear_1_mesh.pvd") << self.gear_1._cell_marker
                    dlf.File("gear_2_mesh.pvd") << self.gear_2._cell_marker
                F_pad, tau_sg = self.compute_force_torque_numerically(p_deg)
                if i == 1 and j == 1:
                    exit()
                tau_lg = (np.cross(self.lg.x_M - self.sg.x_M, F_pad) - tau_sg)[0]

                if self.gear_1 is self.sg:
                    tau_21 = tau_sg
                    tau_12 = tau_lg
                elif self.gear_2 is self.sg:
                    tau_21 = tau_lg
                    tau_12 = tau_sg
                else:
                    raise RuntimeError()

                with open(self._main_dir + f"/{data_fname}.csv", "a+") as f:
                    f.write(f"{self.gear_1.angle} {self.gear_2.angle} {tau_21} {tau_12} \n")
                # rotate smaller gear
                self.update_gear(self.sg, d_angle=d_angle_sg)

            # rotate larger gear
            self.update_gear(self.lg, d_angle=d_angle_lg, update_segment=True)

def sample_torque_ball(n_iterations, mesh_size, p_deg=1, interpolate="once", dir="sample_ball_gear"):
    # create directory
    if not os.path.exists(dir):
        os.mkdir(dir)

    # create two ball gears
    gear_1_ball = MagneticBallGear(n=15, r=1.5, R=10, x_M=np.zeros(3))
    gear_2_ball = MagneticBallGear(n=17, r=2.0, R=15, x_M=np.array([0., 1., 0.]))
    gear_1_ball.create_magnets(magnetization_strength=1.0)
    gear_2_ball.create_magnets(magnetization_strength=1.0)

    # create coaxial gears problem
    D = gear_1_ball.R + gear_2_ball.R + gear_1_ball.r + gear_2_ball.r + 1.0
    bg = SpurGearsSampling(gear_1_ball, gear_2_ball, D, main_dir=dir)

    # mesh both gears
    bg.mesh_gear(bg.gear_1, mesh_size_magnets=mesh_size, fname="gear_1")
    bg.mesh_gear(bg.gear_2, mesh_size_magnets=mesh_size, fname="gear_2")

    # choose smaller gear ("sg")
    bg.assign_gears()

    # mesh the segment
    bg.interpolate_to_segment(mesh_size, p_deg, interpolate=interpolate)

    write_paramter_file(bg, dir)

    bg.sample(n_iterations, p_deg, data_fname="torque_ball_gears", interpolate="once")


def sample_torque_bar(n_iterations, mesh_size, p_deg=1, interpolate="once"):
    sample_dir = "sample_bar_gear"
    # create test directory
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    gear_1_bar = MagneticBarGear(n=12, h=2.0, w=0.75, d=0.5, R=12, x_M=np.zeros(3))
    gear_2_bar = MagneticBarGear(n=15, h=2.0, w=0.75, d=0.5, R=18, x_M=np.array([0., 1., 0.]))
    gear_1_bar.create_magnets(magnetization_strength=1.0)
    gear_2_bar.create_magnets(magnetization_strength=1.0)

    # create coaxial gears problem
    D = gear_1_bar.R + gear_2_bar.R + gear_1_bar.r + gear_2_bar.r + 1.0
    bg = SpurGearsSampling(gear_1_bar, gear_2_bar, D)

    # mesh both gears
    bg.mesh_gear(bg.gear_1, mesh_size_magnets=mesh_size, fname="gear_1")
    bg.mesh_gear(bg.gear_2, mesh_size_magnets=mesh_size, fname="gear_2")

    # choose smaller gear ("sg")
    bg.assign_gears()

    # mesh the segment
    bg.interpolate_to_segment(mesh_size, p_deg, interpolate=interpolate)

    write_paramter_file(bg, sample_dir)

    bg.sample(n_iterations, p_deg, data_fname="torque_bar_gears", interpolate="once")

"""def sample_torque_segment(n_iterations):
    sample_dir = "sample_segment_gear"
    # create test directory
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    gear_1_bar = SegmentGear(n=12, R=10.0, w=0.75, d=0.5, x_M=np.zeros(3))
    gear_2_bar = SegmentGear(n=15, R=12.0, w=0.75, d=0.5, x_M=np.array([0., 1., 0.]))
    gear_1_bar.create_magnets(magnetization_strength=1.0)
    gear_2_bar.create_magnets(magnetization_strength=1.0)

    mesh_size_magnets = 0.1
    sg = SpurGearsSampling(gear_1_bar, gear_2_bar, D=30, main_dir=sample_dir)
    sg.mesh_gear(gear_1_bar, mesh_size_space=2.0, mesh_size_magnets=mesh_size_magnets, \
        fname="gear_1", padding=gear_1_bar.w / 2)
    sg.mesh_gear(gear_2_bar, mesh_size_space=2.0, mesh_size_magnets=mesh_size_magnets, \
        fname="gear_2", padding=gear_2_bar.w / 2)

    write_paramter_file(sg, sample_dir)

    sg.interpolate_to_segment()
    sg.sample(n_iterations, data_fname="torque_bar_gears")"""

def write_paramter_file(problem, dir_):
    with open(dir_ + "/par.json", "w") as f:
        f.write(json.dumps(problem.gear_1.parameters))
        f.write(json.dumps(problem.gear_2.parameters))
        f.write(json.dumps(problem.D))


if __name__ == "__main__":
    sample_torque_ball(n_iterations=3, mesh_size=0.6, p_deg=1, interpolate="once")
    #sample_torque_bar(n_iterations=2)
    #sample_torque_segment(n_iterations=2)