import numpy as np
import pandas as pd
from coaxial.coaxial_gears import CoaxialGearsProblem
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear
from coaxial.grid_generator import ball_gear_mesh, bar_gear_mesh
import os


class CoaxialGears(CoaxialGearsProblem):
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
        d_alpha = 2. * np.pi / self.gear_1.n / n_iterations
        with open(self._main_dir + f"/sample_torque.csv", "w") as f:
            f.write("phi_1 \t phi_2 \t tau_1 \t tau_2\n")
        for _ in range(n_iterations):
            df = pd.DataFrame(np.empty((n_iterations, 4)), columns=("phi_1", "phi_2", "tau_1", "tau_2"))
            for j in range(n_iterations):
                B1 = self.interpolate_field_gear(self.gear_1, self.gear_2, "B", "CG", 1, 0.1, 5.0)
                B2 = self.interpolate_field_gear(self.gear_2, self.gear_1, "B", "CG", 1, 0.1, 5.0)

                tau1 = self.compute_torque_on_gear(self.gear_1, B2)
                tau2 = self.compute_torque_on_gear(self.gear_2, B1)

                df.loc[j] = (self.gear_1.angle, self.gear_2.angle, tau1, tau2)
                self.gear_2.update_parameters(d_alpha)
            df.to_csv(self._main_dir + f"/sample_torque.csv", mode="a+", index=False, header=False, sep="\t")
            self.gear_2.update_parameters(- 2 * np.pi / self.gear_1.n)
            assert np.isclose(self.gear_2.angle, 0.)
            self.gear_1.update_parameters(d_alpha)
        


if __name__ == "__main__":
    sample = os.getcwd() + "/sample"
    # create test directory
    if not os.path.exists(sample):
        os.mkdir(sample)

    gear_1_ball = MagneticBallGear(n=15, r=1.5, R=10, x_M=np.zeros(3), axis=np.array([1., 0., 0.]))
    gear_2_ball = MagneticBallGear(n=17, r=2.0, R=15, x_M=np.array([0., 1., 0.]), axis=np.array([1., 0., 0.]))
    gear_1_ball.create_magnets(magnetization_strength=1.0)
    gear_2_ball.create_magnets(magnetization_strength=1.0)

    bg = CoaxialGears(gear_1_ball, gear_2_ball, D=30, main_dir=sample)
    bg.mesh_gear(gear_1_ball, mesh_size_space=3.0, mesh_size_magnets=0.2, \
        fname="gear_1", padding=gear_1_ball.r / 2)
    bg.mesh_gear(gear_2_ball, mesh_size_space=2.0, mesh_size_magnets=0.3, \
        fname="gear_2", padding=gear_2_ball.r / 2)

    bg.simulate(n_iterations=30)