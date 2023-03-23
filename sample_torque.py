import dolfin as dlf
import numpy as np
import pandas as pd
import json
import os
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
        angles_1 = np.linspace(- np.pi / self.gear_1.n, np.pi / self.gear_1.n, num=n_iterations, endpoint=True)
        angles_2 = np.linspace(- np.pi / self.gear_2.n, np.pi / self.gear_2.n, num=n_iterations, endpoint=True)
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
        assert np.isclose(self.gear_1.angle, 0.)
        assert np.isclose(self.gear_2.angle, 0.)
        # rotate segment back by - pi / n
        self.update_gear(self.lg, - np.pi / self.lg.n)
        self.update_gear(self.sg, - np.pi / self.sg.n)
        assert np.isclose(self.lg.angle, angles_lg[0])
        assert np.isclose(self.sg.angle, angles_sg[0])

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
                    tau_12_values[j, i] = tau_12
                    tau_21_values[j, i] = tau_21
                elif self.gear_2 is self.sg:
                    tau_21 = tau_lg
                    tau_12 = tau_sg
                    tau_12_values[i, j] = tau_12
                    tau_21_values[i, j] = tau_21
                else:
                    raise RuntimeError()

                # rotate smaller gear
                self.update_gear(self.sg, d_angle=d_angle_sg)

            # rotate back to zero angle
            self.update_gear(self.sg, d_angle=float(-self.sg.angle) + angles_sg[0])
            assert np.isclose(self.sg.angle, angles_sg[0])

            # update csv files
            pd.DataFrame(tau_12_values, index=angles_1, columns=angles_2).to_csv(f"{self._main_dir}/{data_dir}/tau_12.csv")
            pd.DataFrame(tau_21_values, index=angles_1, columns=angles_2).to_csv(f"{self._main_dir}/{data_dir}/tau_21.csv")

            # rotate larger gear
            self.update_gear(self.lg, d_angle=d_angle_lg)


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
    mesh_size = 0.2
    p_deg = 2
    r1 = 1.0
    r2 = 1.0
    d = 0.1

    # parameters
    gear_ratio_values = np.array([1.0, 1.5, 2.0, 4.0])
    R1_values = np.array([6.0, 10.0, 20.0])
    p1_values = list(range(4, 40, 2))
    par_list = list(it.product(gear_ratio_values, R1_values, p1_values))
    par = par_list[par_number]
    gear_ratio, R1, p1 = par

    # compute rest of values
    R2 = R1 * gear_ratio
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()
    p2 = int(p2)

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    radius_dir = f"R1_{str(R1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = f"{gear_ratio_dir}/{radius_dir}/{pole_nb_dir}"
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
    w1 = 3.0
    w2 = w1
    t1 = 1.0
    t2 = 1.0
    d = 0.5
    # create sample directory
    sample_dir = "sample_bar_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    mesh_size = 0.2
    p_deg = 2
    d = 0.1

    # parameters
    gear_ratio_values = np.array([1.0, 1.5, 2.0, 4.0])
    R1_values = np.array([6.0, 10.0, 20.0])
    p1_values = list(range(4, 40, 2))
    par_list = list(it.product(gear_ratio_values, R1_values, p1_values))
    par = par_list[par_number]
    gear_ratio, R1, p1 = par

    # compute rest of values
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()
    p2 = int(p2)
    d1 = np.tan(np.pi / p1) * (2 * R1 - t1)
    d2 = d1
    R2 = 1 / 2 * (np.tan(np.pi / p1) / np.tan(np.pi / p2) * (2 * R1 - t1) + t2)

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    radius_dir = f"R1_{str(R1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = f"{gear_ratio_dir}/{radius_dir}/{pole_nb_dir}"
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
    print(f"gear_{1 if sampling.sg is sampling.gear_1 else 2}_{id(sampling)}")
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
    w1 = 3.0
    w2 = w1
    t1 = 1.0
    t2 = 1.0
    d = 0.1
    # create sample directory
    sample_dir = "sample_cylinder_segment_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    mesh_size = 0.2
    p_deg = 2
    d = 0.1

    # parameters
    gear_ratio_values = np.array([1.0, 1.5, 2.0, 4.0])
    R1_values = np.array([6.0, 10.0, 20.0])
    p1_values = list(range(4, 40, 2))
    par_list = list(it.product(gear_ratio_values, R1_values, p1_values))
    par = par_list[par_number]
    gear_ratio, R1, p1 = par

    # compute rest of values
    R2 = R1 * gear_ratio
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()
    p2 = int(p2)

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    radius_dir = f"R1_{str(R1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = f"{gear_ratio_dir}/{radius_dir}/{pole_nb_dir}"
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
        sample_ball(10, it_nb)
    elif magnet_type == "bar":
        sample_bar(10, it_nb)
    elif magnet_type == "cylinder_segment":
        sample_segment(10, it_nb)
    else:
        raise RuntimeError()
