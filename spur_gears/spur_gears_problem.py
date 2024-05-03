from source.magnetic_gear_classes import MagneticGear, MagneticBarGear, SegmentGear, MagneticBallGear
from source.magnet_classes import PermanentMagnet, PermanentAxialMagnet
from source.tools.fenics_tools import rotate_vector_field, compute_current_potential
from source.tools.tools import create_reference_mesh, interpolate_field, write_hdf5_file, read_hd5f_file
from source.tools.mesh_tools import read_mesh
from source.tools.math_tools import get_rot
from source.fe import compute_magnetic_potential
from spur_gears.grid_generator import cylinder_segment_mesh

import dolfin as dlf
dlf.set_log_level(dlf.LogLevel.ERROR)
dlf.parameters["allow_extrapolation"] = True
import numpy as np
import os
import json
import logging
logging.basicConfig(level=logging.INFO)


class SpurGearsProblem:
    def __init__(self, first_gear: MagneticGear, second_gear: MagneticGear, D, main_dir=None):
        """Spur gears problem base.

        The rotation axis is chosen to be the x-axis. The two gears are expected to be
        seperated only in y-direction. The second gear's center point is reset according
        to the first gear's center point and the specified distance D.

        Args:
            first_gear (MagneticGear): First magnetic gear.
            second_gear (MagneticGear): Second magnetic gear.
            D (float): Distance between the gears' center points.
            main_dir (str, optional): Main directory. Defaults to None. Current working
                                      directory is used in that case.
        """
        # set write and read directory
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert os.path.exists(main_dir)
            self._main_dir = main_dir
        # make sure distance is large enough 
        assert hasattr(first_gear, "outer_radius")
        assert hasattr(second_gear, "outer_radius")
        assert D > first_gear.outer_radius + second_gear.outer_radius
        self._D = D

        # set gears
        self._gear_1 = first_gear
        self._gear_2 = second_gear
        self.assign_gear_roles()

        # set the domain size
        self.gear_2.x_M = self.gear_1.x_M + np.array([0., D, 0.])
        self.set_domain_size()
        self._magnet_type = self.gear_1.magnet_type

    @property
    def domain_size(self):
        assert hasattr(self, "_domain_size")
        return self._domain_size

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self):
        return self._D

    @property
    def gear_1(self):
        assert hasattr(self, "_gear_1")
        return self._gear_1

    @property
    def gear_2(self):
        assert hasattr(self, "_gear_2")
        return self._gear_2

    @property
    def gears(self):
        assert hasattr(self, "_gear_1")
        assert hasattr(self, "_gear_2")
        return (self._gear_1, self._gear_2)

    @property
    def magnet_type(self):
        return self._magnet_type

    def _reference_parameters_dict(self, gear, domain_radius):
        """Get reference parameters for reference files.

        Args:
            gear (MagneticGear): The magnetic gear the reference
                                 field should be used for.
            domain_radius (float): The problem's domain radius.

        Returns:
            dict: A dictionary containing all parameters.
        """
        par = {
            "magnet_type": gear.magnet_type,
            "domain_radius": domain_radius / gear.scale_parameter,
        }
        if isinstance(gear, MagneticBarGear):
            par.update({
                "w": gear.w / gear.scale_parameter,
                "d": gear.d / gear.scale_parameter
                })
        elif isinstance(gear, SegmentGear):
            par.update({
                "w": gear.w / gear.scale_parameter,
                "t": gear.t / gear.scale_parameter,
                "alpha": gear.alpha,
                })
        return par

    def align_gears(self):
        """Rotate both gears such that magnets align."""
        assert hasattr(self.gear_1, "_magnets")
        assert hasattr(self.gear_2, "_magnets")

        # align both gears
        self.align_gear(self.gear_1, self.gear_2)
        self.align_gear(self.gear_2, self.gear_1)

        # make sure north and south pole of the magnets are facing each other
        self.sg.update_parameters(2 * np.pi / self.sg.p)
        self.sg.reset_angle(0.)
        assert np.isclose(self.sg.angle, 0.)

    def assign_gear_roles(self):
        """Assign smaller and larger gear."""
        if self.gear_1.R < self.gear_2.R:
            self.sg = self.gear_1  # smaller gear
            self.lg = self.gear_2  # larger gear
        else:
            self.sg = self.gear_2  # smaller gear
            self.lg = self.gear_1  # larger gear

    def align_gear(self, gear, other_gear):
        """Align gear with another gear.

        The gear's angle is reset to zero in the aligned position.

        Args:
            gear (MagneticGear): Gear that shall be rotated such that it is aligned.
            other_gear (MagneticGear): Other gear that the first one shall be aligned with.
        """
        assert hasattr(gear, "_magnets")

        # goal position for magnet
        vec = (other_gear.x_M - gear.x_M)
        vec /= np.linalg.norm(vec)
        x_M_magnet = gear.x_M + gear.R * vec

        # rotate gear such that first magnet is aligned
        arg = np.dot(gear.magnets[0].x_M - gear.x_M, x_M_magnet - gear.x_M) / gear.R ** 2  # argument of arccos
        if np.isclose(abs(arg), 1.):  # the value may be slightly higher than 1, like 1.0000000000004 => set to 1
            arg = np.sign(arg) * 1

        angle = np.abs(np.arccos(arg))
        sign = np.sign(np.cross(gear.magnets[0].x_M - gear.x_M, x_M_magnet - gear.x_M).dot(np.array([1., 0., 0.])))
        if np.isclose(angle, np.pi):  # angle = pi leads to sign = 0, so set the sign manually
            sign = 1
        gear.update_parameters(sign * angle)
        assert np.allclose(x_M_magnet, gear.magnets[0].x_M)

        # reset angle to zero
        gear.reset_angle(0.)

    def _find_reference_files(self, gear, field_name, mesh_size_min, mesh_size_max, p_deg, rtol=1.0):
        """Find reference reference mesh and reference field files.

        Args:
            gear (MagneticGear): Magnetic gear.
            field_name (str): Name of the reference field.
            mesh_size_min (float): Minimum mesh size in reference field.
            mesh_size_max (float): Maximum mesh size in reference field.
            p_deg (int): Polynomial degree of finite element space.
            rtol (float, optional): Relative tolerance. If the domain size 
                                    of the reference file exceeds the problem's
                                    domain size by rtol the reference file will
                                    not be used. Defaults to 1.0 (allows for
                                    twice the problem's size).

        Returns:
            tuple[str, bool]: Directory name and a boolean indicating whether an
                              appropriate directory was found. If not the direc-
                              tory name will be None.
        """
        assert hasattr(self, "_domain_size")
        # some parameters
        found_dir = False
        dir_name = None
        # start with some domain radius that is too large
        domain_radius_file = float(np.inf)

        # search existing gear mesh directories for matching parameter file
        subdirs = [r[0] for r in os.walk(self._main_dir + "/data/reference/")]
        for subdir in subdirs:
            if "par.json" in os.listdir(subdir):
                with open(os.path.join(subdir, "par.json"), "r") as f:
                    # read paramters
                    par = json.loads(f.read())
                    # check if all paramters match
                    if par["domain_radius"] < domain_radius_file and \
                        self._match_reference_parameters(par, gear, field_name, mesh_size_min, mesh_size_max, p_deg):
                        found_dir = True
                        domain_radius_file = par["domain_radius"]
                        dir_name = subdir
        return dir_name, found_dir

    def _match_reference_parameters(self, par_file, gear, field_name, mesh_size_min, mesh_size_max, p_deg):
        """Check whether reference files are appropriate for the problem.

        Args:
            par_file (dict): A dictionary containing the reference file info.
            gear (MagneticGear): The magnetic gear the reference file should
                                 be used for.
            field_name (str): Name of the reference field.
            mesh_size_min (float): Minimum mesh size.
            mesh_size_max (float): Maximum mesh size.
            p_deg (int): Polynomial degree of finite element space.

        Returns:
            bool: True if parameters match (reference file can be used).
        """
        # check if reference mesh shape is the same
        if par_file["shape"] != "cylinder":
            return False

        # check if magnet type is the same
        if par_file["magnet_type"] != gear.magnet_type:
            return False

        # check if field name is the same
        if par_file["name"] != field_name:
            return False

        # check if polynomial degree is the same
        if par_file["p_deg"] != p_deg:
            return False

        # check if mesh size is correct
        if not np.isclose(par_file["mesh_size_min"], mesh_size_min):
            return False
        if not np.isclose(par_file["mesh_size_max"], mesh_size_max):
            return False

        # check if domain size and geometry matches
        if par_file["domain_radius"] * gear.scale_parameter < self._domain_size:
            return False

        if isinstance(gear, MagneticBarGear):
            if not np.allclose((gear.w, gear.d),
                               (par_file["w"] * gear.scale_parameter,
                                par_file["d"] * gear.scale_parameter)):
                return False
        elif isinstance(gear, SegmentGear):
            if not np.allclose((gear.w, gear.t, gear.alpha),
                               (par_file["w"] * gear.scale_parameter,
                                par_file["t"] * gear.scale_parameter,
                                par_file["alpha"])):
                return False

        return True

    def load_reference_field(self, gear: MagneticGear, field_name, cell_type, p_deg, mesh_size_min,
                              mesh_size_max, domain_size, analytical_solution=True, R_inf=None,
                                write_to_pvd=False):
        """Load reference field from hdf5 file. If no appropriate file is
           found the file will be written first.

        Args:
            gear (MagneticGear): Magnetic gear.
            field_name (str): Name of the field, e.g. "B".
            cell_type (str): Finite element cell type.
            p_deg (int): Polynomial degree of finite element.
            mesh_size_min (float): Minimum mesh size of reference field. This value
                                   will be used as the starting mesh size at the center
                                   of the reference mesh.
            mesh_size_max (float): Maximum mesh size of reference mesh. The reference
                                   mesh size is increased up to this value with increasing
                                   distance from the center of the reference mesh.
            domain_size (float): The maximum distance between two points of the two
                                 gear meshes. This value will be used as the radius of
                                 the reference mesh.
        """
        assert field_name in ("Vm", "B"), "What field is this?"
        vector_valued = (field_name == "B")  # check if field is vector valued

        # find directory with matching files
        ref_dir, found_dir = self._find_reference_files(gear, field_name, mesh_size_min, mesh_size_max, p_deg, rtol=1.0)

        # check if both mesh file and hdf5 file exist; if not, create both
        if not found_dir:
            # create directory
            subdirs = [r[0] for r in os.walk(os.path.join(self._main_dir, "data", "reference"))]
            while True:
                ref_dir = os.path.join(self._main_dir, "data", "reference", 
                                   f"{gear.magnet_type}_{field_name}_{int(domain_size)}_{np.random.randint(10_000, 50_000)}"
                                   )
                if not(ref_dir in subdirs):
                    break
            os.makedirs(ref_dir)

            # create reference mesh
            ref_mag = gear.reference_magnet()

            if gear is self.gear_1:
                thickness = self.gear_2.width
            elif gear is self.gear_2:
                thickness = self.gear_1.width
            else:
                raise RuntimeError()

            # create reference magnet and check if the field is implemented
            assert hasattr(ref_mag, field_name), f"{field_name} is not implemented for this magnet class"

            # interface size (only used if field is seemingly poorly defined near the boundary)
            d = self.D - self.gear_1.outer_radius - self.gear_2.outer_radius
            d /= gear.scale_parameter  # make sure d is still large enough after rescaling the reference field

            # for the segment gear no analytical solution is available: compute potential numerically
            if not analytical_solution:
                fname = os.path.join(self._main_dir, "data", f"Vm_{id(self)}")
                # radius of sphere has to be larger such that it contains the cylinder
                ref_radius = np.sqrt(domain_size ** 2 + thickness ** 2 / 4) + 1e-3
                field_num = compute_magnetic_potential(ref_mag, ref_radius, R_inf=R_inf, mesh_size_magnet=mesh_size_min,
                                                        mesh_size_domain_min=mesh_size_min,
                                                        mesh_size_domain_max=mesh_size_max, p_deg=p_deg,
                                                        cylinder_mesh_size_field=True, mesh_size_field_thickness=thickness,
                                                        fname=fname, write_to_pvd=write_to_pvd)
                if field_name == "B":
                    field_num = compute_current_potential(field_num, project=True)

                create_reference_mesh(ref_mag, domain_size / gear.scale_parameter, mesh_size_min, mesh_size_max,
                                        shape="cylinder", thickness=thickness, d=d, fname=os.path.join(ref_dir, "reference_mesh"))
                # read reference mesh
                reference_mesh = read_mesh(os.path.join(ref_dir, "reference_mesh.xdmf"))

                # interpolate reference field
                field_interpol = interpolate_field(field_num, reference_mesh, cell_type, p_deg,
                                                   fname=os.path.join(ref_dir, f"{field_name}_{id(self)}"),
                                                   write_pvd=True)
            else:
                create_reference_mesh(ref_mag, domain_size / gear.scale_parameter, mesh_size_min, 
                                      mesh_size_max, shape="cylinder", thickness=thickness, d=d,
                                      fname=os.path.join(ref_dir, "reference_mesh")
                                      )
                # read reference mesh
                reference_mesh = read_mesh(os.path.join(ref_dir, "reference_mesh.xdmf"))
                # interpolate reference field
                if field_name == "B":
                    field_interpol = interpolate_field(ref_mag.B, reference_mesh, cell_type, p_deg,
                                                       fname=os.path.join(ref_dir, field_name),
                                                       write_pvd=True)
                elif field_name == "Vm":
                    field_interpol = interpolate_field(ref_mag.Vm, reference_mesh, cell_type, p_deg,
                                                       fname=os.path.join(ref_dir, field_name), write_pvd=True)
                else:
                    raise RuntimeError()

            # write the field to hdf5 file
            write_hdf5_file(field_interpol, reference_mesh, fname=os.path.join(ref_dir, f"{field_name}.h5"), field_name=field_name)

            # write parameter file
            with open(os.path.join(ref_dir, "par.json"), "w") as f:
                ref_par = self._reference_parameters_dict(gear, domain_size)
                ref_par.update({
                    "shape": "cylinder",
                    "name": field_name,
                    "mesh_size_min": mesh_size_min, 
                    "mesh_size_max": mesh_size_max,
                    "p_deg": p_deg
                    })
                f.write(json.dumps(ref_par))
        else:
            # read the reference mesh
            reference_mesh = read_mesh(os.path.join(ref_dir, "reference_mesh.xdmf"))

        # read reference field from hd5f file
        reference_field = read_hd5f_file(os.path.join(ref_dir, f"{field_name}.h5"), field_name, 
                                         reference_mesh, cell_type, p_deg, vector_valued=vector_valued)

        # set reference field and mesh for gear
        gear.set_reference_mesh(reference_mesh, field_name)
        gear.set_reference_field(reference_field, field_name)

    def mesh_reference_segment(self, mesh_size, phi_sg_min, phi_sg_max, phi_lg_min,
                               phi_lg_max, write_to_pvd=False):
        """Create mesh of reference cylinder segment.

        Args:
            mesh_size (float): Mesh size.
            write_to_pvd (bool, optional): If True write pvd files. Defaults to False.
        """
        # set geometrical paramters
        t = self.lg.width  # segment thickness (width)

        # angle to contain smaller gear
        beta = np.abs(2 * np.arccos(np.sqrt(self.D ** 2 - self.lg.outer_radius ** 2) / self.D))

        # inner segment radius
        Ri = self.D - self.lg.outer_radius

        # set outer segment radius
        # alpha_r is the angle that is "removed" from the smaller gear if magnets
        # are removed
        if self.lg.p > len(self.lg.magnets):
            alpha_r = 2 * np.pi / self.lg.p * (self.lg.p - len(self.lg.magnets)) - 2 * np.max(np.abs((phi_lg_min, phi_lg_max)))
        else:
            alpha_r = 0.
        # outer segment radius
        Ro = np.sqrt((self.D + self.lg.outer_radius * np.cos(alpha_r / 2)) ** 2 + (self.lg.outer_radius * np.sin(alpha_r / 2)) ** 2)

        # x_axis of segment (for angle)
        if self.lg.x_M[1] > self.sg.x_M[1]:
            x_axis = np.array([0., 1., 0.])
        else:
            x_axis = np.array([0., -1., 0.])

        ref_path = os.path.join(self._main_dir, "data", "reference")
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)

        self.segment_mesh, _, _ = cylinder_segment_mesh(Ri=Ri, Ro=Ro, t=t, angle_start=- beta / 2 - phi_sg_max,
                                                        angle_stop=beta / 2 - phi_sg_min, x_M_ref=self.sg.x_M, x_axis=x_axis,
                                                        fname=os.path.join(ref_path, f"reference_segment_{id(self)}"),
                                                        pad=False, mesh_size=mesh_size, write_to_pvd=write_to_pvd)

    def interpolate_to_reference_segment(self, p_deg=2, interpolate="twice", use_Vm=True):
        """
        Interpolate field of smaller gear on reference segment.

        Args:
            p_deg (int, optional): Polynomial degree. Defaults to 2.
            interpolate (str, optional): Number of interpolation steps. Defaults to "twice".
            use_Vm (bool, optional): Whether to use magnetic potential Vm. Defaults to True.
        """
        assert hasattr(self, "segment_mesh")

        use_ref_field = (interpolate == "twice")

        # interpolate fields of other gear on segment
        if use_Vm:
            self.Vm_segment = self.interpolate_field_gear(self.sg, self.segment_mesh, "Vm", "CG", p_deg=p_deg,
                                                          use_ref_field=use_ref_field)
        else:
            self.B_segment = self.interpolate_field_gear(self.sg, self.segment_mesh, "B", "CG", p_deg=p_deg,
                                                         use_ref_field=use_ref_field)

    def interpolate_field_gear(self, gear: MagneticGear, mesh, field_name, cell_type, p_deg, use_ref_field=True):
        """Interpolate a field of a gear on a given mesh.

        Args:
            gear (MagneticGear): Magnetic gear.
            mesh (dlf.Mesh): The finite element mesh.
            field_name (str): Name of the field, e.g. "B".
            cell_type (str): Finite element cell type.
            p_deg (int): Polynomial degree of finite element.
            use_ref_field (bool, optional): Whether or not to use a reference field or to evaluate
                                  the field of each magnet directly. Defaults to True.

        Returns:
            dlf.Function: The interpolated field.
        """
        assert field_name in ("B", "Vm")
        assert hasattr(self, "_domain_size")

        if use_ref_field:
            # make sure reference field has been loaded
            assert hasattr(gear, f"_{field_name}_ref"), "Reference field has not been set."

            # create reference field handler
            ref_field = gear.reference_field(field_name)
            ref_mesh = gear.reference_mesh(field_name)

        # create function space on target mesh
        if field_name == "B":
            V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)
        elif field_name == "Vm":
            V = dlf.FunctionSpace(mesh, cell_type, p_deg)
        else:
            raise RuntimeError()

        assert hasattr(self, "_interpolate_field_magnet")

        # initialize the sum over all fields
        field_sum = 0.

        logging.info("Interpolating field of gear... ")
        # interpolate field for every magnet and add it to the sum
        for mag in gear.magnets:
            logging.info("Interpolating field of magnet...")
            if use_ref_field:
                scale = gear.scale_parameter
                if not isinstance(mag, PermanentAxialMagnet):
                    scale *= mag.mag_sign * gear.reference_magnet().mag_sign

                interpol_field = self._interpolate_field_magnet(mag, ref_field, field_name, ref_mesh,
                                                                mesh, cell_type, p_deg, scale=scale)
                assert interpol_field.function_space().mesh() is mesh
            else:
                if field_name == "B":
                    interpol_field = interpolate_field(mag.B, mesh, cell_type, p_deg)
                elif field_name == "Vm":
                    interpol_field = interpolate_field(mag.Vm, mesh, cell_type, p_deg)
                else:
                    raise NotImplementedError()

            field_sum += interpol_field.vector().copy()
            logging.info("Done.")

        logging.info("Done.")
        return dlf.Function(V, field_sum)

    def _interpolate_field_magnet(self, magnet, ref_field: dlf.Function, field_name, reference_mesh, mesh, cell_type, p_deg, scale=1.0):
        """Interpolate a magnet's magnetic field on a mesh. 

        Args:
            magnet (PermanentMagnet): The Magnet.
            ref_field (dlf.Function): The reference field that contains the
                                field for a reference magnet.
            field_name (str): Name of the field.
            reference_mesh (dlf.mesh): The reference mesh.
            mesh (dlf.Mesh): The target mesh.
            cell_type (str): Finite element type.
            p_deg (int): Polynomial degree.
            scale (float): Scaling factor.

        Returns:
            dlf.Function: The field interpolated on the mesh.
        """
        # A copy of the reference mesh needs to be set as a class
        # variable because fenics does not delete the mesh when
        # the function terminates. Therefore, overwrite the mesh
        # each time the function is called. This way, only a single
        # copy is created.
        if field_name == "B":
            self._B_reference_mesh_copy = dlf.Mesh(reference_mesh)
            ref_mesh_copy = self._B_reference_mesh_copy
            assert ref_mesh_copy is self._B_reference_mesh_copy  # same object
            V_ref = dlf.VectorFunctionSpace(self._B_reference_mesh_copy, cell_type, p_deg)
            ref_field_copy = dlf.Function(V_ref, ref_field.vector().copy())
            V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)
            # scale field sign
            if not np.isclose(scale, 1.0):
                ref_field_copy.vector()[:] *= np.sign(scale)
            # rotate reference field
            rotate_vector_field(ref_field_copy, magnet.Q)
        elif field_name == "Vm":
            self._Vm_reference_mesh_copy = dlf.Mesh(reference_mesh)
            ref_mesh_copy = self._Vm_reference_mesh_copy
            assert ref_mesh_copy is self._Vm_reference_mesh_copy  # same object
            V_ref = dlf.FunctionSpace(self._Vm_reference_mesh_copy, cell_type, p_deg)
            # scale magnetic potential
            ref_field_copy = dlf.Function(V_ref, ref_field.vector().copy())
            if not np.isclose(scale, 1.0):
                ref_field_copy.vector()[:] *= scale
            V = dlf.FunctionSpace(mesh, cell_type, p_deg)
        else:
            raise RuntimeError()

        # scale, rotate and shift reference mesh according to magnet placement
        ref_mesh_copy.scale(abs(scale))
        # rotate first, then shift!
        coords = np.array(ref_mesh_copy.coordinates())
        ref_mesh_copy.coordinates()[:] = magnet.Q.dot(coords.T).T
        ref_mesh_copy.translate(dlf.Point(*magnet.x_M))

        # interpolate field to new function space and add the result
        interpol_field = dlf.Function(V)
        dlf.LagrangeInterpolator.interpolate(interpol_field, ref_field_copy)

        return interpol_field

    def compute_force_on_gear(self, gear: MagneticGear, B, p_deg=2):
        """Compute the force on a gear caused by a magnetic field B.

        Only computes the y- and z-components of the force as only
        those are necessary to compute the torque around the x-axis.

        Args:
            gear (MagneticGear): The magnetic gear.
            B (dlf.Function, dlf.ComponentTensor): The magnetic field.
            p_deg (int, optional): Polynomial degree. Defaults to 2.

        Returns:
            np.ndarray: The force.
        """
        assert hasattr(gear, "magnets")
        logging.info("Computing force on gear... ")

        # only compute y and z component!
        F = np.zeros(2)

        for mag, tag in zip(gear.magnets, gear._magnet_boundary_subdomain_tags):
            assert isinstance(mag, PermanentMagnet)
            if isinstance(mag, PermanentAxialMagnet):
                M_jump = dlf.as_vector(- mag.M)  # jump of magnetization
            else:
                M_jump = - mag.M_inner_as_expression(degree=p_deg)
            t = dlf.cross(dlf.cross(gear.normal_vector, M_jump), B)  # traction vector
            # select y and z components
            for i, c in enumerate((t[1], t[2])):
                f_expr = c * gear.dA(tag)
                F[i] += dlf.assemble(f_expr)  # add to force

        logging.info("Done.")
        return F

    def compute_torque_on_gear(self, gear: MagneticGear, B, p_deg=2):
        """Compute the torque on a gear caused by a magnetic field B.

        Args:
            gear (MagneticGear): The magnetic gear.
            B (dlf.Function, dlf.ComponentTensor): The magnetic field.
            p_deg (int, optional): Polynomial degree. Defaults to 2.

        Returns:
            float: The torque.
        """
        assert hasattr(gear, "magnets")
        logging.info("Computing torque on gear... ")
        # initialize torque, position vectors
        tau = 0.
        x = dlf.SpatialCoordinate(gear.mesh)
        x_M = dlf.as_vector(gear.x_M)

        for mag, tag in zip(gear.magnets, gear._magnet_boundary_subdomain_tags):
            if isinstance(mag, PermanentAxialMagnet):
                M_jump = dlf.as_vector(- mag.M)  # jump of magnetization
            else:
                M_jump = - mag.M_inner_as_expression(degree=p_deg)
            t = dlf.cross(dlf.cross(gear.normal_vector, M_jump), B)  # traction vector
            m = dlf.cross(x - x_M, t)  # torque density
            tau_mag = dlf.assemble(m[0] * gear.dA(tag))
            tau += tau_mag  # add to torque

        logging.info("Done.")
        return tau

    def compute_force_torque(self, p_deg=2, use_Vm=True):
        """
        Compute force and torque on larger gear.

        Args:
            p_deg (int, optional): Polynomial degree. Defaults to 2.
            use_Vm (bool, optional): Whether to use the magnetic potential.
                                     Defaults to True.

        Returns:
            tuple: Force and torque on smaller gear.
        """
        assert hasattr(self, "segment_mesh")
        # Copy everything
        # This is necessary because fenics returns zeros if the mesh
        # is rotated and used for interpolation multiple times. To avoid
        # this copy the entire reference function for interpolation
        seg_mesh = dlf.Mesh(self.segment_mesh)
        if use_Vm:
            V_seg = dlf.FunctionSpace(seg_mesh, "CG", p_deg)
            Vm_seg = dlf.Function(V_seg, self.Vm_segment.vector().copy())
        else:
            V_seg = dlf.VectorFunctionSpace(seg_mesh, "CG", p_deg)
            B_seg = dlf.Function(V_seg, self.B_segment.vector().copy())

        # if interpolation is used, use the field on the segment
        if use_Vm:
            # create function on smaller gear
            V = dlf.FunctionSpace(self.lg.mesh, "CG", p_deg)
            Vm_sg = dlf.Function(V)

            dlf.LagrangeInterpolator.interpolate(Vm_sg, Vm_seg)
            B_sg = compute_current_potential(Vm_sg, project=False)
        else:
            # create function on smaller gear
            V = dlf.VectorFunctionSpace(self.lg.mesh, "CG", p_deg)
            B_sg = dlf.Function(V)

            dlf.LagrangeInterpolator.interpolate(B_sg, B_seg)

        # compute force
        F = self.compute_force_on_gear(self.lg, B_sg, p_deg=p_deg)
        F_lg = np.array([0., F[0], F[1]])  # pad force (insert x-component)

        # compute torque
        tau_lg = self.compute_torque_on_gear(self.lg, B_sg, p_deg=p_deg)

        return F_lg, tau_lg
 
    def create_gear_mesh(self, gear: MagneticGear, **kwargs):
        """Mesh a gear.

        Args:
            gear (MagneticGear): Magnetic gear. 
            kwargs (any): Input to gear's mesh function.
        """
        # create directory if it does not exist
        gear_dir = os.path.join(self._main_dir, "data", "gears")
        if not os.path.exists(gear_dir):
            os.makedirs(gear_dir)
        if gear is self.gear_1:
            dir_name = "gear_1"
        elif gear is self.gear_2:
            dir_name = "gear_2"
        else:
            raise RuntimeError()

        target_dir = os.path.join(self._main_dir, "data", "gears", dir_name)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        fname = kwargs["fname"]
        kwargs.update({"fname": os.path.join(target_dir, fname)})

        mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags  \
            = gear.mesh_gear(gear, **kwargs)

        gear.set_mesh_markers_and_tags(mesh, cell_marker, facet_marker, magnet_subdomain_tags,
                                       magnet_boundary_subdomain_tags)

    def remove_magnets(self, gear: MagneticGear, D_ref):
        """
        Remove magnets if they are outside the relevant domain.

        Args:
            gear (MagneticGear): Magnetic gear.
            D_ref (float): Reference distance.
        """
        if gear is self.gear_1:
            other_gear = self.gear_2
        elif gear is self.gear_2:
            other_gear = self.gear_1
        else:
            raise RuntimeError()
 
        rot = get_rot(np.pi / gear.p)
        magnets = list(gear.magnets)  # copy magnet list: list is different, magnets are the same
        for magnet in magnets:
            x_M_max = rot.dot(magnet.x_M - gear.x_M) + gear.x_M
            x_M_min = rot.T.dot(magnet.x_M - gear.x_M) + gear.x_M
            # remove magnet if too far away
            if (np.linalg.norm(x_M_max - other_gear.x_M) > D_ref) and (np.linalg.norm(x_M_min - other_gear.x_M) > D_ref):
                gear.magnets.remove(magnet)
        # set new domain size
        self.set_domain_size()

    def set_domain_size(self):
        """Set domain size of the problem."""
        # get number of removed magnets
        n_rem_1 = self.gear_1.p - len(self.gear_1.magnets)
        n_rem_2 = self.gear_2.p - len(self.gear_2.magnets)
        if n_rem_1 > 0:
            alpha_r_1 = np.pi / self.gear_1.p * (self.gear_1.p - len(self.gear_1.magnets) - 1)  # allow rotation by +- pi / n
        else:
            alpha_r_1 = 0.
        if n_rem_2 > 0:
            alpha_r_2 = np.pi / self.gear_2.p * (self.gear_2.p - len(self.gear_2.magnets) - 1)  # allow rotation by +- pi / n
        else:
            alpha_r_2 = 0.
        # domain_size
        # compute position of the two points with the greatest distance
        xP1 = self.gear_1.x_M + self.gear_1.outer_radius * np.array([0., - np.cos(alpha_r_1), np.sin(alpha_r_1)])
        xP2 = self.gear_2.x_M + self.gear_2.outer_radius * np.array([0., np.cos(alpha_r_2), - np.sin(alpha_r_2)])
        self._domain_size = np.linalg.norm(xP1 - xP2)
