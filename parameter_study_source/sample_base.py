import dolfin as dlf
import numpy as np
import pandas as pd
import json
import os
from source.magnetic_gear_classes import MagneticGear
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

    def sample(self, angles_1, angles_2, data_dir, p_deg=2):
        """
        Sample torque values.

        Args:
            n_iterations (int): Number of torque samples (per angle).
            data_dir (str): Directory where sampling data shall be saved.
            p_deg (int, optional): Polynomial degree used for computation. Defaults to 2.
        """
        d_angle_1 = angles_1[1] - angles_1[0]
        d_angle_2 = angles_2[1] - angles_2[0]
        n1 = angles_1.size
        n2 = angles_2.size

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
            pd.DataFrame(tau_12_values, index=angles_1, columns=angles_2).to_csv(
                os.path.join(self._main_dir, data_dir, "tau_12.csv")
                )
            pd.DataFrame(tau_21_values, index=angles_1, columns=angles_2).to_csv(
                os.path.join(self._main_dir, data_dir, "tau_21.csv")
                )

            # rotate second gear
            dlf.File("mesh.pvd") << self.sg.mesh
            self.update_gear(self.gear_1, d_angle=d_angle_1)


def write_parameter_file(problem, sample_dir):
    with open(os.path.join(sample_dir, "par.json"), "w") as f:
        par = {
            "gear_1": problem.gear_1.parameters,
            "gear_2": problem.gear_2.parameters,
            "D": problem.D
        }
        f.write(json.dumps(par, cls=NpEncoder))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)