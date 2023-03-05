import dolfin as dlf
from dolfin import LagrangeInterpolator
import numpy as np
import os
from source.magnetic_gear_classes import MagneticBallGear, MagneticGear
from source.tools.tools import interpolate_field
from source.tools.fenics_tools import compute_magnetic_field
from spur_gears.spur_gears import CoaxialGearsProblem
from spur_gears.grid_generator import segment_mesh, gear_mesh
#from source.grid_generator import gear_mesh
from tests.convergence_tests.ball_force_torque_convergence import compute_force_analytically


class CoaxialGearsConvergenceTest(CoaxialGearsProblem):
    def __init__(self, first_gear, second_gear, D, main_dir=None):
        super().__init__(first_gear, second_gear, D, main_dir)
        self.align_gears()
        self.set_mesh_functions()

    def set_mesh_functions(self):
        """Set mesh function for both gears."""
        for gear in self.gears:
            gear.set_mesh_function(gear_mesh)

    def assign_gears(self):
        if self.gear_1.R < self.gear_2.R:
            self.sg = self.gear_1  # smaller gear
            self.lg = self.gear_2  # larger gear
        else:
            self.sg = self.gear_2  # smaller gear
            self.lg = self.gear_1  # larger gear

    def mesh_gear(self, gear, mesh_size_magnets, fname, write_to_pvd=True):
        if gear is self.gear_1:
            other_gear = self.gear_2
        elif gear is self.gear_2:
            other_gear = self.gear_1
        else:
            raise RuntimeError()
        self.create_gear_mesh(gear, x_M_ref=other_gear.x_M, \
            mesh_size_magnets=mesh_size_magnets, fname=fname, write_to_pvd=write_to_pvd)
        """self.create_gear_mesh(gear, mesh_size_space=mesh_size_space, \
            mesh_size_magnets=mesh_size_magnets, fname=fname, padding=padding, \
                write_to_pvd=write_to_pvd)"""

    def interpolate_to_segment(self, mesh_size_magnets, interpolate="once", use_Vm=False):
        assert interpolate in ("once", "twice")

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
            if use_Vm:
                V = dlf.FunctionSpace(self.seg_mesh, "CG", 1)
                Vm_ref_vec = 0.
                for mag in self.lg.magnets:
                    Vm_mag = interpolate_field(mag.Vm_as_expression(), self.seg_mesh, "CG", 1)
                    Vm_ref_vec+= Vm_mag._cpp_object.vector()
                self.Vm_ref = dlf.Function(V, Vm_ref_vec)
            else:
                V = dlf.VectorFunctionSpace(self.seg_mesh, "CG", 1)
                B_ref_vec = 0.
                for mag in self.lg.magnets:
                    B_mag = interpolate_field(mag.B_as_expression(), self.seg_mesh, "CG", 1)
                    B_ref_vec += B_mag._cpp_object.vector()
                self.B_ref = dlf.Function(V, B_ref_vec)
        elif interpolate == "twice":
            if use_Vm:
                # interpolate fields of other gear on segment
                self.Vm_ref = self.interpolate_field_gear(self.lg, self.seg_mesh, "Vm", "DG", 1, \
                                                        mesh_size_magnets / max(self.gear_1.scale_parameter, \
                                                                                self.gear_2.scale_parameter), 3 * mesh_size_magnets)
            else:
                self.B_ref = self.interpolate_field_gear(self.lg, self.seg_mesh, "B", "CG", 1, \
                                                    mesh_size_magnets / max(self.gear_1.scale_parameter, \
                                                                            self.gear_2.scale_parameter), 3 * mesh_size_magnets)


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
                self.seg_mesh.rotate(- np.sign(gear.angle) * 360. / gear.n, 0, dlf.Point(gear.x_M))
            # then update parameters
            gear.update_parameters(- np.sign(gear.angle) * 2. * np.pi / gear.n)


def compute_torque_analytically(magnet_1, magnet_2, force, x_M_gear):
    """Compute torque on magnet_2 caused by magnetic field of magnet_1.

    Args:
        magnet_1 (BallMagnet): First magnet.
        magnet_2 (BallMagnet): Second magnet.
        force (np.ndarray): The force on magnet_2 caused by magnet_1.
        x_M_gear_2 (np.ndarray): The gear's center that magnet_2 belongs to.
    Returns:
        np.ndarray: The torque in the specified coordinated system.
    """

    # all quantities are represented in the cartesian eigenbasis of magnet 1 
    x_M_2 = magnet_1.Q.T.dot(magnet_2.x_M - magnet_1.x_M)
    M_2 = magnet_1.Q.T.dot(magnet_2.M)

    # compute torque 
    tau = np.zeros(3)  # initialize
    B = magnet_1.B_eigen_plus(x_M_2)
    vol_magnet_2 = 4. / 3. * np.pi * magnet_2.R ** 3
    tau += magnet_1.Q.dot(vol_magnet_2 * np.cross(M_2, B))

    # this is the torque w.r.t. the second magnet's center
    # now, compute torque w.r.t. gear's center (x_M_gear)
    tau += np.cross(magnet_2.x_M - x_M_gear, force)
    return tau


def main(mesh_sizes, p_deg=1, interpolate="never", use_Vm=False, dir=None):
    if dir is not None:
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.chdir(dir)
    assert interpolate in ("never", "once", "twice")

    f_12_error = []
    f_21_error = []
    tau_12_error = []
    tau_21_error = []

    for ms in mesh_sizes[::-1]:
        # create two ball gears
        gear_1 = MagneticBallGear(5, 6, 1, np.zeros(3))
        gear_1.create_magnets(magnetization_strength=1.0)
        gear_2 = MagneticBallGear(5, 5, 1, np.array([0., 1., 0.]))
        gear_2.create_magnets(magnetization_strength=1.0)

        # create coaxial gears problem
        D = gear_1.R + gear_2.R + gear_1.r + gear_2.r + 3.0
        cg = CoaxialGearsConvergenceTest(gear_1, gear_2, D)

        # mesh both gears
        cg.mesh_gear(cg.gear_1, mesh_size_magnets=ms, fname="gear_1")
        cg.mesh_gear(cg.gear_2, mesh_size_magnets=ms, fname="gear_2")

        # choose smaller gear ("sg")
        cg.assign_gears()

        # mesh the segment if interpolation is used (once or twice)
        if interpolate in ("once", "twice"):
            cg.interpolate_to_segment(ms, interpolate=interpolate, use_Vm=use_Vm)

        # introduce some randomness: rotate gears and segment by arbitrary angles
        cg.update_gear(cg.sg, d_angle=np.pi / 4 / cg.sg.n) #((2 * np.random.rand(1) - 1) * np.pi / cg.sg.n).item())
        cg.update_gear(cg.lg, d_angle=np.pi / 4 / cg.lg.n, update_segment=(interpolate != "never"))
        #cg.update_gear(cg.lg, d_angle=((2 * np.random.rand(1) - 1) * np.pi / cg.lg.n).item(), update_segment=(interpolate != "never"))

        ####################################
        ### 1. compute torque numerically ##
        ####################################

        if interpolate in ("once", "twice"):
            assert hasattr(cg, "seg_mesh")
            # if interpolation is used, use the field in on the segment
            if use_Vm:
                # create function on smaller gear
                V = dlf.FunctionSpace(cg.sg.mesh, "CG", p_deg)
                Vm_lg = dlf.Function(V)

                # copy reference_field
                segment_mesh_copy = dlf.Mesh(cg.seg_mesh)
                V_ref = dlf.FunctionSpace(segment_mesh_copy, "CG", 1)
                Vm_ref = dlf.Function(V_ref, cg.Vm_ref._cpp_object.vector())

                LagrangeInterpolator.interpolate(Vm_lg, Vm_ref)
                B_lg = compute_magnetic_field(Vm_lg)
            else:
                # create function on smaller gear
                V = dlf.VectorFunctionSpace(cg.sg.mesh, "CG", p_deg)
                B_lg = dlf.Function(V)

                # copy reference_field
                segment_mesh_copy = dlf.Mesh(cg.seg_mesh)
                V_ref = dlf.VectorFunctionSpace(segment_mesh_copy, "CG", p_deg)
                B_ref = dlf.Function(V_ref, cg.B_ref._cpp_object.vector())

                LagrangeInterpolator.interpolate(B_lg, B_ref)
        elif interpolate == "never":
            if use_Vm:
                V = dlf.FunctionSpace(cg.sg.mesh, "CG", p_deg)
                Vm_lg_vec = 0.
                for mag in cg.lg.magnets:
                    Vm_mag = interpolate_field(mag.Vm_as_expression(), cg.sg.mesh, "CG", p_deg)
                    Vm_lg_vec += Vm_mag._cpp_object.vector()
                Vm_lg = dlf.Function(V, Vm_lg_vec)
                B_lg = compute_magnetic_field(Vm_lg)
            else:
                V = dlf.VectorFunctionSpace(cg.sg.mesh, "CG", p_deg)
                B_lg_vec = 0.
                for mag in cg.lg.magnets:
                    B_mag = interpolate_field(mag.B_as_expression(), cg.sg.mesh, "CG", p_deg)
                    B_lg_vec += B_mag._cpp_object.vector()
                B_lg = dlf.Function(V, B_lg_vec)

        dlf.File("B_lg.pvd") << B_lg

        # compute force
        F = cg.compute_force_on_gear(cg.sg, B_lg)

        # compute torque
        tau_sg = cg.compute_torque_on_gear(cg.sg, B_lg)
        F_pad = np.array([0., F[0], F[1]])
        tau_lg = (np.cross(cg.lg.x_M - cg.sg.x_M, F_pad) - tau_sg)[0]

        if cg.gear_1 is cg.sg:
            tau_21_num = tau_sg
            tau_12_num = tau_lg
            f_21_num_vec = F_pad
            f_12_num_vec = - F_pad
        elif cg.gear_2 is cg.sg:
            tau_21_num = tau_lg
            tau_12_num = tau_sg
            f_12_num_vec = F_pad
            f_21_num_vec = - F_pad
        else:
            raise RuntimeError()

        ####################################
        ## 2. compute torque analytically ##
        ####################################

        # compute torque analytically
        f_12_ana_vec = np.zeros(3)
        f_21_ana_vec = np.zeros(3)
        tau_12_ana_vec = np.zeros(3)
        tau_21_ana_vec = np.zeros(3)
        for m1 in cg.gear_1.magnets:
            for m2 in cg.gear_2.magnets:
                f_ana = compute_force_analytically(m1, m2)
                f_12_ana_vec += f_ana
                tau_mag = compute_torque_analytically(m1, m2, f_ana, cg.gear_2.x_M)
                tau_12_ana_vec += tau_mag

        for m1 in cg.gear_2.magnets:
            for m2 in cg.gear_1.magnets:
                f_ana = compute_force_analytically(m1, m2)
                f_21_ana_vec += f_ana
                tau_mag = compute_torque_analytically(m1, m2, f_ana, cg.gear_1.x_M)
                tau_21_ana_vec += tau_mag
        tau_12_ana = tau_12_ana_vec[0]
        tau_21_ana = tau_21_ana_vec[0]


        f_12_e = np.linalg.norm(f_12_ana_vec[1:] - f_12_num_vec[1:]) / np.linalg.norm(f_12_ana_vec[1:])
        f_21_e = np.linalg.norm(f_21_ana_vec[1:] - f_21_num_vec[1:]) / np.linalg.norm(f_21_ana_vec[1:])
        tau_12_e = np.abs(tau_12_ana - tau_12_num) / np.abs(tau_12_ana)
        tau_21_e = np.abs(tau_21_ana - tau_21_num) / np.abs(tau_21_ana)
        # compare values
        print(f"""INFO: \n
              mesh size: {ms} \n
              f_12_num: {f_12_num_vec} \n
              f_12_ana: {f_12_ana_vec} \n
              f_12_error: {f_12_e} \n
              f_21_num: {f_21_num_vec} \n
              f_21_ana: {f_21_ana_vec} \n
              f_21_error: {f_21_e} \n
              tau_12_num: {tau_12_num} \n
              tau_12_ana: {tau_12_ana} \n
              tau_12_error: {tau_12_e} \n
              tau_21_num: {tau_21_num} \n
              tau_21_ana: {tau_21_ana} \n
              tau_21_error: {tau_21_e} \n
              """)
        f_12_error.append(f_12_e)
        f_21_error.append(f_21_e)
        tau_12_error.append(tau_12_e)
        tau_21_error.append(tau_21_e)

    errors = (f_12_error, f_21_error, tau_12_error, tau_21_error)
    names = ("f_12", "f_21", "tau_12", "tau_21")
    for error, name in zip(errors, names):
        with open(f"{name}_error.csv", "a+") as f:
            for e, ms in zip(error, mesh_sizes[::-1]):
                f.write(f"{ms} {e} \n")

    os.chdir("..")

if __name__ == "__main__":
    conv_dir = "convergence_test"
    if not os.path.exists(conv_dir):
        os.mkdir(conv_dir)
    os.chdir(conv_dir)
    mesh_sizes = np.geomspace(1e-1, 1.0, num=3)
    for p_deg in (1, 2):
        # 1. interpolate never, use Vm
        main(mesh_sizes, p_deg, interpolate="never", use_Vm=True, dir=f"Vm_interpol_never_pdeg_{p_deg}")
        # 2. interpolate once, use Vm
        main(mesh_sizes, p_deg, interpolate="once", use_Vm=True, dir=f"Vm_interpol_once_pdeg_{p_deg}")
        # 3. interpolate twice, use Vm
        main(mesh_sizes, p_deg, interpolate="twice", use_Vm=True, dir=f"Vm_interpol_twice_pdeg_{p_deg}")
        # 4. interpolate never, use B directly
        main(mesh_sizes, p_deg, interpolate="never", use_Vm=False, dir=f"B_interpol_never_pdeg_{p_deg}")
        # 6. interpolate once, use  directly
        main(mesh_sizes, p_deg, interpolate="once", use_Vm=False, dir=f"B_interpol_once_pdeg_{p_deg}")
        # 5. interpolate twice, use B directly
        main(mesh_sizes, p_deg, interpolate="twice", use_Vm=False, dir=f"B_interpol_twice_pdeg_{p_deg}")