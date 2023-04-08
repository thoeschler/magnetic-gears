import dolfin as dlf
import numpy as np
import os
from source.magnetic_gear_classes import MagneticGear
from source.tools.tools import interpolate_field
from source.tools.math_tools import get_rot
from source.tools.fenics_tools import compute_current_potential, rotate_vector_field
from spur_gears.spur_gears_problem import SpurGearsProblem
from source.grid_generator import gear_mesh
from tests.convergence_tests.ball_magnets_force_torque import compute_force_ana


class SpurGearsConvergenceTest(SpurGearsProblem):
    def __init__(self, first_gear, second_gear, D, main_dir=None):
        super().__init__(first_gear, second_gear, D, main_dir)
        self.align_gears()
        assert self.gear_1.angle == 0.
        assert self.gear_2.angle == 0.
        self.all_magnets_1 = list(self.gear_1.magnets)
        self.all_magnets_2 = list(self.gear_2.magnets)

    def set_mesh_functions(self):
        """Set mesh function for both gears."""
        for gear in self.gears:
                gear.set_mesh_function(gear_mesh)

    def mesh_gear(self, gear, mesh_size, fname, write_to_pvd=True):
        self.create_gear_mesh(gear, mesh_size=mesh_size, fname=fname, \
                                write_to_pvd=write_to_pvd)

    def compute_force_torque_num(self, p_deg=1, interpolate="never", use_Vm=False):
        if interpolate in ("once", "twice"):
            F_sg, tau_sg = self.compute_force_torque(
                p_deg=p_deg, use_Vm=use_Vm
                )
        elif interpolate == "never":
            if use_Vm:
                V = dlf.FunctionSpace(self.sg.mesh, "CG", p_deg)
                Vm_lg_vec = 0.
                for mag in self.lg.magnets:
                    Vm_mag = interpolate_field(mag.Vm, self.sg.mesh, "CG", p_deg)
                    Vm_lg_vec += Vm_mag.vector()
                Vm_lg = dlf.Function(V, Vm_lg_vec)
                B_lg = compute_current_potential(Vm_lg, project=True)
            else:
                V = dlf.VectorFunctionSpace(self.sg.mesh, "CG", p_deg)
                B_lg_vec = 0.
                for mag in self.lg.magnets:
                    B_mag = interpolate_field(mag.B, self.sg.mesh, "CG", p_deg)
                    B_lg_vec += B_mag.vector()
                B_lg = dlf.Function(V, B_lg_vec)

            # compute force
            F = self.compute_force_on_gear(self.sg, B_lg)

            # compute torque
            tau_sg = self.compute_torque_on_gear(self.sg, B_lg)
            F_sg = np.array([0., F[0], F[1]])  # pad force (insert x-component)
        else:
            raise ValueError()

        return F_sg, tau_sg

    def update_gear(self, gear: MagneticGear, d_angle, mesh_all_magnets=False, interpolate="never"):
        """Update gear given an angle increment.

        Args:
            gear (MagneticGear): A magnetic gear.
            d_angle (float): Angle increment.
            mesh_all_magnets(bool, optional): If True all magnets are meshed and none are deleted.
            interpolate (str): Interpolation type. Defaults to "never".
        """
        if not mesh_all_magnets:
            if not hasattr(self, "deleted_magnets_1"):
                # get deleted magnets
                self.deleted_magnets_1 = list()
                for mag in self.all_magnets_1:
                    if mag not in self.gear_1.magnets:
                        self.deleted_magnets_1.append(mag)
                assert len(self.deleted_magnets_1) == self.gear_1.p - len(self.gear_1.magnets)

            if not hasattr(self, "deleted_magnets_2"):
                # get deleted magnets
                self.deleted_magnets_2 = list()
                for mag in self.all_magnets_2:
                    if mag not in self.gear_2.magnets:
                        self.deleted_magnets_2.append(mag)
                assert len(self.deleted_magnets_2) == self.gear_2.p - len(self.gear_2.magnets)

        gear.update_parameters(d_angle)
        if not mesh_all_magnets:
            if gear is self.gear_1:
                gear.update_magnets(self.deleted_magnets_1, d_angle, np.zeros(3))
            elif gear is self.gear_2:
                gear.update_magnets(self.deleted_magnets_2, d_angle, np.zeros(3))
            else:
                raise RuntimeError()

        if gear is self.lg and interpolate != "never":
            self.segment_mesh.rotate(d_angle * 180. / np.pi, 0, dlf.Point(gear.x_M))
            if hasattr(self, "B_segment"):
                rotate_vector_field(self.B_segment, get_rot(d_angle))

def compute_torque_ana(magnet_1, magnet_2, force, x_M_gear):
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


def convergence_test(cg: SpurGearsConvergenceTest, mesh_size, p_deg=1, mesh_all_magnets=True, \
                     D_ref=None, interpolate="never", use_Vm=False, dir=None, \
                        analytical_solution=True, R_inf_mult=None):
    if dir is not None:
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.chdir(dir)
    assert interpolate in ("never", "once", "twice")
    # check input
    if not mesh_all_magnets:
        assert D_ref is not None
        cg.remove_magnets(cg.gear_1, D_ref=D_ref)
        cg.remove_magnets(cg.gear_2, D_ref=D_ref)
    if not analytical_solution:
        assert R_inf_mult is not None

    cg.set_mesh_functions()

    # mesh both gears
    cg.mesh_gear(cg.gear_1, mesh_size=mesh_size, fname="gear_1")
    cg.mesh_gear(cg.gear_2, mesh_size=mesh_size, fname="gear_2")

    # mesh the segment if interpolation is used (once or twice)
    if interpolate in ("once", "twice"):
        if R_inf_mult is None:
            R_inf = None
        else:
            R_inf = R_inf_mult * cg.sg.magnets[0].size
        cg.load_reference_field(cg.lg, "Vm" if use_Vm else "B", "CG", p_deg, \
                                mesh_size, mesh_size, cg.domain_size, analytical_solution=analytical_solution,
                                R_inf=R_inf)
        cg.mesh_reference_segment(mesh_size)
        cg.interpolate_to_reference_segment(p_deg, interpolate=interpolate, use_Vm=use_Vm)

    # introduce some randomness: rotate gears and segment by arbitrary angles
    cg.update_gear(cg.sg, d_angle=((2 * np.random.rand(1) - 1) * np.pi / cg.sg.p).item())
    # the segment contains the field of the larger gear. Rotate the segment as well
    # when rotating the larger gear. Updates B_segment automatically (if used). 
    cg.update_gear(cg.lg, d_angle=(np.random.rand(1) * np.pi / cg.lg.p).item(), \
                    mesh_all_magnets=mesh_all_magnets, interpolate=interpolate)

    ###############################################
    ##### 1. compute force/torque numerically #####
    ###############################################
    F_pad, tau_sg = cg.compute_force_torque_num(p_deg, interpolate=interpolate, use_Vm=use_Vm)
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

    ################################################
    ##### 2. compute force/torque analytically #####
    ################################################

    # compute torque analytically
    f_12_ana_vec = np.zeros(3)
    f_21_ana_vec = np.zeros(3)
    tau_12_ana_vec = np.zeros(3)
    tau_21_ana_vec = np.zeros(3)
    assert len(cg.all_magnets_1) == cg.gear_1.p
    assert len(cg.all_magnets_2) == cg.gear_2.p

    for m1 in cg.all_magnets_1:
        for m2 in cg.all_magnets_2:
            f_ana = compute_force_ana(m1, m2)
            f_12_ana_vec += f_ana
            tau_mag = compute_torque_ana(m1, m2, f_ana, cg.gear_2.x_M)
            tau_12_ana_vec += tau_mag

    for m1 in cg.all_magnets_2:
        for m2 in cg.all_magnets_1:
            f_ana = compute_force_ana(m1, m2)
            f_21_ana_vec += f_ana
            tau_mag = compute_torque_ana(m1, m2, f_ana, cg.gear_1.x_M)
            tau_21_ana_vec += tau_mag
    tau_12_ana = tau_12_ana_vec[0]
    tau_21_ana = tau_21_ana_vec[0]

    f_12_e = np.linalg.norm(f_12_ana_vec[1:] - f_12_num_vec[1:]) / np.linalg.norm(f_12_ana_vec[1:])
    f_21_e = np.linalg.norm(f_21_ana_vec[1:] - f_21_num_vec[1:]) / np.linalg.norm(f_21_ana_vec[1:])
    tau_12_e = np.abs(tau_12_ana - tau_12_num) / np.abs(tau_12_ana)
    tau_21_e = np.abs(tau_21_ana - tau_21_num) / np.abs(tau_21_ana)

    print(f"""INFO: \n
            mesh size: {mesh_size} \n
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

    errors = (f_12_e, f_21_e, tau_12_e, tau_21_e)
    names = ("f_12", "f_21", "tau_12", "tau_21")

    if dir is not None:
        os.chdir("..")

    return errors, names
