import dolfin as dlf
import numpy as np
import pandas as pd
import json
import os
from math import ceil
import itertools as it
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear, SegmentGear, MagneticGear
from spur_gears.spur_gears_problem import SpurGearsProblem
from source.grid_generator import gear_mesh


class SpurGearsSampling(SpurGearsProblem):
    def __init__(self, first_gear, second_gear, D, main_dir=None):
        super().__init__(first_gear, second_gear, D, main_dir)
        self.align_gears()
        self.remove_magnets(self.gear_1, D_ref=D)
        self.remove_magnets(self.gear_2, D_ref=D)
        assert self.gear_1.angle == 0.
        assert self.gear_2.angle == 0.
        self.set_mesh_functions()

    def set_mesh_functions(self):
        """Set mesh function for both gears."""
        for gear in self.gears:
            gear.set_mesh_function(gear_mesh)

    def mesh_gear(self, gear, mesh_size, fname, write_to_pvd=True):
        """
        Mesh a magnetic gear.

        Args:
            gear (MagneticGear): Magnetic gear.
            mesh_size (float): Mesh size.
            fname (str): File name.
            write_to_pvd (bool, optional): If true write mesh to pvd file. Defaults to True.
        """
        self.create_gear_mesh(gear, mesh_size=mesh_size, fname=fname, write_to_pvd=write_to_pvd)

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

    def sample(self, n_iterations, data_dir, p_deg=2):
        """
        Sample torque values.

        Args:
            n_iterations (int): Number of torque samples (per angle).
            data_dir (str): Directory where sampling data shall be saved.
            p_deg (int, optional): Polynomial degree used for computation. Defaults to 2.
        """
        angle_1_lim = [- np.pi / self.gear_1.p, np.pi / self.gear_1.p]
        angle_2_lim = [- np.pi / self.gear_2.p, np.pi / self.gear_2.p]
        if self.gear_1 is self.lg:
            angle_1_lim[0] = 0.
            n1 = ceil(n_iterations / 2)
            n2 = n_iterations
        elif self.gear_2 is self.lg:
            angle_2_lim[0] = 0.
            n1 = n_iterations
            n2 = ceil(n_iterations / 2)
        angles_1 = np.linspace(*angle_1_lim, num=n1, endpoint=True)
        angles_2 = np.linspace(*angle_2_lim, num=n2, endpoint=True)
        d_angle_1 = angles_1[1] - angles_1[0]
        d_angle_2 = angles_2[1] - angles_2[0]

        assert np.isclose(self.gear_1.angle, 0.)
        assert np.isclose(self.gear_2.angle, 0.)
        # move gears to inital position
        self.update_gear(self.gear_1, angles_1[0])
        self.update_gear(self.gear_2, angles_2[0])
        assert np.isclose(self.gear_1.angle, angles_1[0])
        assert np.isclose(self.gear_2.angle, angles_2[0])

        tau_12_values = np.zeros((n1, n2))
        tau_21_values = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                # check angles
                assert np.isclose(self.gear_1.angle, angles_1[i])
                assert np.isclose(self.gear_2.angle, angles_2[j])
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

                # save torque values
                tau_12_values[i, j] = tau_12
                tau_21_values[i, j] = tau_21

                # rotate first gear
                self.update_gear(self.gear_2, d_angle=d_angle_2)

            # rotate first gear back to initial angle
            self.update_gear(self.gear_2, d_angle=float(-self.gear_2.angle) + angles_2[0])
            assert np.isclose(self.gear_2.angle, angles_2[0])

            # update csv files
            pd.DataFrame(tau_12_values, index=angles_1, columns=angles_2).to_csv(f"{self._main_dir}/{data_dir}/tau_12.csv")
            pd.DataFrame(tau_21_values, index=angles_1, columns=angles_2).to_csv(f"{self._main_dir}/{data_dir}/tau_21.csv")

            # rotate second gear
            self.update_gear(self.gear_1, d_angle=d_angle_1)


def write_parameter_file(problem, sample_dir):
    with open(sample_dir + "/par.json", "w") as f:
        par = {
            "gear_1": problem.gear_1.parameters,
            "gear_2": problem.gear_2.parameters,
            "D": problem.D
        }
        f.write(json.dumps(par))

def sample_ball(n_iterations, par_number):
    # create sample directory
    sample_dir = "sample_ball_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    R1 = 10.
    mesh_size = 0.6
    p_deg = 2
    d = 0.1

    # parameters
    gear_ratio_values = np.array([1.0, 1.4, 2.0, 2.4])
    p1_values = list(range(4, 40, 2))
    par_list = list(it.product(gear_ratio_values, p1_values))
    par = par_list[par_number]
    gear_ratio, p1 = par

    # compute rest of values
    R2 = R1 * gear_ratio
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()
    p2 = int(p2)

    r1 = R1 * np.sin(np.pi / p1)
    r2 = R2 * np.sin(np.pi / p2)

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = f"{gear_ratio_dir}/{pole_nb_dir}"
    if not os.path.exists(f"{sample_dir}/{data_dir}"):
        os.makedirs(f"{sample_dir}/{data_dir}")

    gear_1 = MagneticBallGear(p1, R1, r1, np.zeros(3))
    gear_2 = MagneticBallGear(p2, R2, r2, np.array([0., 1., 0.]))
    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    D = gear_1.outer_radius + gear_2.outer_radius + d

    # use same main dir for all parameter combination
    # this way all reference fields can be used in each iteration
    sampling = SpurGearsSampling(gear_1, gear_2, D, main_dir=sample_dir)

    # mesh smaller gear
    sampling.mesh_gear(sampling.sg, mesh_size=mesh_size, fname=f"gear_{1 if sampling.sg is sampling.gear_1 else 2}", \
                       write_to_pvd=False)

    # mesh the segment
    sampling.load_reference_field(sampling.lg, "Vm", "CG", p_deg=p_deg, mesh_size_min=mesh_size, \
                                    mesh_size_max=mesh_size, domain_size=sampling.domain_size, \
                                    analytical_solution=True, write_to_pvd=False)
    sampling.mesh_reference_segment(mesh_size, write_to_pvd=False)
    sampling.interpolate_to_reference_segment(p_deg=p_deg, interpolate="twice", use_Vm=True)

    write_parameter_file(sampling, f"{sample_dir}/{data_dir}")

    sampling.sample(n_iterations, data_dir=data_dir, p_deg=p_deg)


def sample_bar(n_iterations, par_number):
    t1 = 1.0
    t2 = 1.0
    d_ref = 0.2
    R1 = 10.
    # create sample directory
    sample_dir = "sample_bar_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    mesh_size = 0.1
    p_deg = 2

    # parameters
    gear_ratio_values = np.array([1.0, 1.4, 2.0, 2.4])
    w_ref = 3.0
    R_ref_values = np.array([6., 10., 20.])
    w1_values = w_ref * R1 / R_ref_values
    p1_values = list(range(4, 60, 2))
    par_list = list(it.product(gear_ratio_values, w1_values, p1_values))
    par = par_list[par_number]
    gear_ratio, w1, p1 = par
    w2 = w1

    # compute rest of values
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()
    d = d_ref * w1 / w_ref

    p2 = int(p2)
    d1 = np.tan(np.pi / p1) * (2 * R1 - t1)
    d2 = d1
    R2 = 1 / 2 * (np.tan(np.pi / p1) / np.tan(np.pi / p2) * (2 * R1 - t1) + t2)

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    width_dir = f"w1_{str(w1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = f"{gear_ratio_dir}/{width_dir}/{pole_nb_dir}"
    if not os.path.exists(f"{sample_dir}/{data_dir}"):
        os.makedirs(f"{sample_dir}/{data_dir}")

    gear_1 = MagneticBarGear(p1, R1, w1, t1, d1, np.zeros(3))
    gear_2 = MagneticBarGear(p2, R2, w2, t2, d2, np.array([0., 1., 0.]))
    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    D = gear_1.outer_radius + gear_2.outer_radius + d

    # use same main dir for all parameter combination
    # this way all reference fields can be used in each iteration
    sampling = SpurGearsSampling(gear_1, gear_2, D, main_dir=sample_dir)

    # mesh smaller gear
    sampling.mesh_gear(sampling.sg, mesh_size=mesh_size, \
                       fname=f"gear_{1 if sampling.sg is sampling.gear_1 else 2}_{id(sampling)}", \
                        write_to_pvd=False)

    # mesh the segment
    sampling.load_reference_field(sampling.lg, "Vm", "CG", p_deg=p_deg, mesh_size_min=mesh_size, \
                                    mesh_size_max=mesh_size, domain_size=sampling.domain_size, \
                                    analytical_solution=True, write_to_pvd=False)

    sampling.mesh_reference_segment(mesh_size, write_to_pvd=False)
    sampling.interpolate_to_reference_segment(p_deg=p_deg, interpolate="twice", use_Vm=True)

    write_parameter_file(sampling, f"{sample_dir}/{data_dir}")

    sampling.sample(n_iterations, data_dir=data_dir, p_deg=p_deg)


def sample_segment(n_iterations, par_number):
    t1 = 1.0
    t2 = 1.0
    d_ref = 0.2
    R1 = 10.
    # create sample directory
    sample_dir = "sample_cylinder_segment_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    mesh_size = 0.1
    p_deg = 2

    # parameters
    gear_ratio_values = np.array([1.0, 1.4, 2.0, 2.4])
    w_ref = 3.0
    R_ref_values = np.array([6., 10., 20.])
    w1_values = w_ref * R1 / R_ref_values
    p1_values = list(range(4, 60, 2))
    par_list = list(it.product(gear_ratio_values, w1_values, p1_values))
    par = par_list[par_number]
    gear_ratio, w1, p1 = par
    w2 = w1

    # compute rest of values
    R2 = R1 * gear_ratio
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()
    p2 = int(p2)
    d = d_ref * w1 / w_ref

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    width_dir = f"w1_{str(w1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = f"{gear_ratio_dir}/{width_dir}/{pole_nb_dir}"
    if not os.path.exists(f"{sample_dir}/{data_dir}"):
        os.makedirs(f"{sample_dir}/{data_dir}")

    gear_1 = SegmentGear(p1, R1, w1, t1, np.zeros(3))
    gear_2 = SegmentGear(p2, R2, w2, t2, np.array([0., 1., 0.]))
    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    D = gear_1.outer_radius + gear_2.outer_radius + d

    # use same main dir for all parameter combination
    # this way all reference fields can be used in each iteration
    sampling = SpurGearsSampling(gear_1, gear_2, D, main_dir=sample_dir)

    # mesh smaller gear
    sampling.mesh_gear(sampling.sg, mesh_size=mesh_size, 
                       fname=f"gear_{1 if sampling.sg is sampling.gear_1 else 2}_{id(sampling)}", \
                       write_to_pvd=False)

    # mesh the segment
    sampling.load_reference_field(sampling.lg, "Vm", "CG", p_deg=p_deg, mesh_size_min=mesh_size, \
                                    mesh_size_max=mesh_size, domain_size=sampling.domain_size, \
                                    analytical_solution=False, write_to_pvd=False)
    sampling.mesh_reference_segment(mesh_size, write_to_pvd=False)
    sampling.interpolate_to_reference_segment(p_deg=p_deg, interpolate="twice", use_Vm=True)

    write_parameter_file(sampling, f"{sample_dir}/{data_dir}")

    sampling.sample(n_iterations, data_dir=data_dir, p_deg=p_deg)


if __name__ == "__main__":
    import sys
    magnet_type = sys.argv[1]
    it_nb = int(sys.argv[2])

    if magnet_type == "ball":
        sample_ball(25, it_nb)
    elif magnet_type == "bar":
        sample_bar(5, it_nb)
    elif magnet_type == "cylinder_segment":
        sample_segment(5, it_nb)
    else:
        raise RuntimeError()
