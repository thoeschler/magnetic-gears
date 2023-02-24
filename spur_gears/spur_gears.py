import dolfin as dlf
from dolfin import LagrangeInterpolator
import numpy as np
import os
import json
from source.magnetic_gear_classes import MagneticBarGear, MagneticGear
from source.tools.tools import create_reference_mesh, interpolate_field, write_hdf5_file, read_hd5f_file
from source.tools.mesh_tools import read_mesh


class CoaxialGearsProblem:
    def __init__(self, first_gear: MagneticGear, second_gear: MagneticGear, D, main_dir=None):
        """Constructor method.

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

        # set the domain size
        self.set_domain_size()

        # overwrite in subclass
        self.gear_2.x_M = self.gear_1.x_M + np.array([0., D, 0.])
        self._magnet_type = None

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
        return par

    def align_gears(self):
        """Rotate both gears such that magnets align."""
        assert hasattr(self.gear_1, "_magnets")
        assert hasattr(self.gear_1, "_magnets")
        self.align_gear(self.gear_1, self.gear_2)
        self.align_gear(self.gear_2, self.gear_1)

    def align_gear(self, gear, other_gear):
        """Align gear with another gear.

        The gear's angle is reset to zero in the aligned position.

        Args:
            gear (MagneticGear): Gear that should be moved such that it is aligned.
            other_gear (MagneticGear): Gear that the other should be aligned with.
        """
        assert hasattr(gear, "_magnets")
        vec = (other_gear.x_M - gear.x_M)
        vec /= np.linalg.norm(vec)
        x_M_magnet = gear.x_M + gear.R * vec  # goal position for magnet
        # rotate gear such that first magnet is aligned
        angle = np.abs(np.arccos(np.dot(gear.magnets[0].x_M - gear.x_M, x_M_magnet - gear.x_M) / gear.R ** 2))
        sign = np.sign(np.cross(gear.magnets[0].x_M - gear.x_M, x_M_magnet - gear.x_M).dot(np.array([1., 0., 0.])))
        if np.isclose(angle, np.pi):  # angle = pi leads to sign = 0, so set it manually
            sign = 1
        gear.update_parameters(sign * angle)
        assert np.allclose(x_M_magnet, gear.magnets[0].x_M)
        gear.reset_angle(0.)

    def _find_reference_files(self, gear, field_name, mesh_size_min, mesh_size_max, rtol=1.0):
        """Find reference reference mesh and reference field files.

        Args:
            gear (MagneticGear): Magnetic gear.
            field_name (str): Name of the reference field.
            mesh_size_min (float): Minimum mesh size in reference field.
            mesh_size_max (float): Maximum mesh size in reference field.
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
        domain_radius_file = (1 + rtol + 1e-2) * self._domain_size / gear.scale_parameter

        # search existing gear mesh directories for matching parameter file
        subdirs = [r[0] for r in os.walk(self._main_dir + "/data/reference/")]
        for subdir in subdirs:
            if "par.json" in os.listdir(subdir):
                with open(f"{subdir}/par.json", "r") as f:
                    # read paramters
                    par = json.loads(f.read())
                    # check if all paramters match
                    if par["domain_radius"] < domain_radius_file and \
                        self._match_reference_parameters(par, gear, field_name, mesh_size_min, mesh_size_max):
                        found_dir = True
                        domain_radius_file = par["domain_radius"]
                        dir_name = subdir
        return dir_name, found_dir

    def _match_reference_parameters(self, par_file, gear, field_name, mesh_size_min, mesh_size_max):
        """Check whether reference files are appropriate for the problem.

        Args:
            par_file (dict): A dictionary containing the reference file info.
            gear (MagneticGear): The magnetic gear the reference file should
                                 be used for.
            field_name (str): Name of the reference field.
            mesh_size_min (float): Minimum mesh size.
            mesh_size_max (float): Maximum mesh size.

        Returns:
            bool: True if parameters match (reference file can be used).
        """
        # check if magnet type is the same
        if not par_file["magnet_type"] == gear.magnet_type:
            return False

        # check if field name is the same
        if not par_file["name"] == field_name:
            return False

        # check if mesh size is correct
        if not np.isclose(par_file["mesh_size_min"], mesh_size_min):
            return False
        if not np.isclose(par_file["mesh_size_max"], mesh_size_max):
            return False

        # check if domain size and geometry matches
        if not par_file["domain_radius"] * gear.scale_parameter >= self._domain_size:
            return False
      
        if isinstance(gear, MagneticBarGear):
            if not np.allclose((gear.w, gear.d), \
                               (par_file["w"] * gear.scale_parameter, \
                                par_file["d"] * gear.scale_parameter)):
                return False

        return True

    def _load_reference_field(self, gear, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, domain_size):
        """Load reference field from hdf5 file. If no appropriate file is
           found the file will be written first.

        Args:
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
        assert isinstance(gear, MagneticGear)
        assert field_name in ("Vm", "B"), "What field is this?"
        vector_valued = (field_name == "B")  # check if field is vector valued

        # find directory with matching files
        ref_dir, found_dir = self._find_reference_files(gear, field_name, mesh_size_min, mesh_size_max, rtol=1.0)

        # check if both mesh file and hdf5 file exist; if not, create both
        if not found_dir:
            # create directory
            subdirs = [r[0] for r in os.walk(self._main_dir + "data/reference")]
            ref_dir = f"{self._main_dir}/data/reference/{gear.magnet_type}_{field_name}_{int(domain_size)}_{np.random.randint(10_000, 50_000)}"
            while ref_dir in subdirs:
                ref_dir = f"{self._main_dir}/data/reference/{gear.magnet_type}_{field_name}_{int(domain_size)}_{np.random.randint(10_000, 50_000)}"
            os.makedirs(ref_dir)

            # create reference mesh
            ref_mag = gear.reference_magnet()
            create_reference_mesh(ref_mag, domain_size / gear.scale_parameter, mesh_size_min, mesh_size_max, fname=f"{ref_dir}/reference_mesh")

            # read reference mesh
            reference_mesh = read_mesh(f"{ref_dir}/reference_mesh.xdmf")

            # create reference magnet and check if the field is implemented
            assert hasattr(ref_mag, field_name), f"{field_name} is not implemented for this magnet class"

            # interpolate reference field
            if field_name == "B":
                field_interpol = interpolate_field(ref_mag.B, reference_mesh, cell_type, p_deg, f"{ref_dir}/{field_name}", write_pvd=True)
            elif field_name == "Vm":
                field_interpol = interpolate_field(ref_mag.Vm, reference_mesh, cell_type, p_deg, f"{ref_dir}/{field_name}", write_pvd=True)
            else:
                raise RuntimeError()

            # write the field to hdf5 file
            write_hdf5_file(field_interpol, reference_mesh, fname=f"{ref_dir}/{field_name}.h5", field_name=field_name)

            # write parameter file
            with open(f"{ref_dir}/par.json", "w") as f:
                ref_par = self._reference_parameters_dict(gear, domain_size)
                ref_par.update({
                    "name": field_name,
                    "mesh_size_min": mesh_size_min, 
                    "mesh_size_max": mesh_size_max
                    })
                f.write(json.dumps(ref_par))
        else:
            # read the reference mesh
            reference_mesh = read_mesh(f"{ref_dir}/reference_mesh.xdmf")

        # read reference field from hd5f file
        reference_field = read_hd5f_file(f"{ref_dir}/{field_name}.h5", field_name, reference_mesh, cell_type, p_deg, vector_valued=vector_valued)

        # set reference field and mesh for gear
        gear.scale_mesh(reference_mesh)
        gear.set_reference_mesh(reference_mesh, field_name)
        gear.set_reference_field(reference_field, field_name)

    def interpolate_field_gear(self, gear, mesh, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max):
        """Interpolate a field of a gear on a given mesh.

        Args:
            gear (MagneticGear): The magnetic gear that owns the field.
            mesh (dlf.Mesh): The finite element mesh.
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

        Returns:
            dlf.Function: The interpolated field.
        """
        assert field_name in ("B", "Vm")
        assert hasattr(self, "_domain_size")

        # load reference field if not done yet
        if not hasattr(gear, f"_{field_name}_ref"):
            self._load_reference_field(gear, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, self._domain_size)

        # create reference field handler
        ref_field = gear.reference_field(field_name)
        ref_mesh = gear.reference_mesh(field_name)
        if field_name == "B":
            # new function space
            V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)
        elif field_name == "Vm":
            # new function space
            V = dlf.FunctionSpace(mesh, cell_type, p_deg)
        else:
            raise RuntimeError()

        assert hasattr(self, "_interpolate_field_magnet")

        # initialize the sum over all fields
        field_sum = 0.

        print(f"Interpolating magnetic field... ", end="")
        # interpolate field for every magnet and add it to the sum
        for mag in gear.magnets:
            interpol_field = self._interpolate_field_magnet(mag, ref_field, field_name,ref_mesh, \
                mesh, cell_type, p_deg)
            field_sum += interpol_field._cpp_object.vector()

        print("Done.")
        return dlf.Function(V, field_sum)

    def _interpolate_field_magnet(self, magnet, ref_field, field_name, reference_mesh, mesh, cell_type, p_deg):
        """Interpolate a magnet's magnetic field on a mesh. 

        Args:
            magnet (PermanentAxialMagnet): The Magnet.
            ref_field (dlf.Function): The reference field that contains the
                                magnetic field for a reference magnet.
            field_name (str): Name of the field.
            reference_mesh (dlf.mesh): The reference mesh.
            mesh (dlf.Mesh): The target mesh.
            cell_type (str): Finite element type.
            p_deg (int): Polynomial degree.

        Returns:
            dlf.Function: The magnetic field interpolated on the mesh.
        """
        # copy everything
        reference_mesh_copy = dlf.Mesh(reference_mesh)

        if field_name == "B":
            V_ref = dlf.VectorFunctionSpace(reference_mesh_copy, cell_type, p_deg)
            reference_field_copy = dlf.Function(V_ref, ref_field._cpp_object.vector())
            V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)
        elif field_name == "Vm":
            V_ref = dlf.FunctionSpace(reference_mesh_copy, cell_type, p_deg)
            reference_field_copy = dlf.Function(V_ref, ref_field._cpp_object.vector())
            V = dlf.FunctionSpace(mesh, cell_type, p_deg)
        else:
            raise RuntimeError()

        # scale, rotate and shift reference mesh according to magnet placement
        # rotate first, then shift!
        reference_mesh_copy.coordinates()[:] = magnet.Q.dot(reference_mesh_copy.coordinates().T).T
        reference_mesh_copy.translate(dlf.Point(*magnet.x_M))

        # interpolate field to new function space and add the result
        interpol_field = dlf.Function(V)
        LagrangeInterpolator.interpolate(interpol_field, reference_field_copy)

        return interpol_field

    def compute_force_on_gear(self, gear: MagneticGear, B):
        """Compute the force on a gear caused by a magnetic field B.

        Only computes the y- and z-components of the force as only
        those are necessary to compute the torque around the x-axis.

        Args:
            gear (MagneticGear): The magnetic gear.
            B (dlf.Function): The magnetic field.

        Returns:
            np.ndarray: The force.
        """
        assert hasattr(gear, "magnets")
        print("Computing force on gear... ", end="")

        # only compute y and z component!
        F = np.zeros(2)

        for mag, tag in zip(gear.magnets, gear._magnet_boundary_subdomain_tags):
            M_jump = dlf.as_vector(- mag.M)  # jump of magnetization
            t = dlf.cross(dlf.cross(gear.normal_vector, M_jump), B)  # traction vector
            # select y and z components
            for i, c in enumerate((t[1], t[2])):
                f_expr = c * gear.dA(tag)
                F[i] += dlf.assemble(f_expr)  # add to force

        print("Done.")
        return F

    def compute_torque_on_gear(self, gear: MagneticGear, B):
        """Compute the torque on a gear caused by a magnetic field B.

        Args:
            gear (MagneticGear): The magnetic gear.
            B (dlf.Function): The magnetic field.

        Returns:
            float: The torque.
        """
        assert hasattr(gear, "magnets")
        print("Computing torque on gear... ", end="")
        # initialize torque, position vectors
        tau = 0.
        x = dlf.Expression(("x[0]", "x[1]", "x[2]"), degree=1)
        x_M = dlf.as_vector(gear.x_M)

        for mag, tag in zip(gear.magnets, gear._magnet_boundary_subdomain_tags):
            M_jump = dlf.as_vector(- mag.M)  # jump of magnetization
            t = dlf.cross(dlf.cross(gear.normal_vector, M_jump), B)  # traction vector
            m = dlf.cross(x - x_M, t)  # torque density
            tau_mag = dlf.assemble(m[0] * gear.dA(tag))
            print("TORQUE_NUM", tau_mag)
            tau += tau_mag  # add to torque

        print("Done.")
        return tau
 
    def create_gear_mesh(self, gear: MagneticGear, **kwargs):
        """Mesh a gear.

        Args:
            gear (MagneticGear): The gear. 
            kwargs (any): Input to gear's mesh function.
        """
        assert hasattr(self, "_gear_1")
        assert hasattr(self, "_gear_2")

        # create directory if it does not exist
        if not os.path.exists(self._main_dir + "/data/gears/"):
            os.makedirs(self._main_dir + "/data/gears/")
        if gear is self.gear_1:
            dir_name = "gear_1"
        elif gear is self.gear_2:
            dir_name = "gear_2"
        else:
            raise RuntimeError()

        target_dir = f"{self._main_dir}/data/gears/{dir_name}"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        fname = kwargs["fname"] 
        kwargs.update({"fname": f"{target_dir}/{fname}"})

        mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags \
            = gear.mesh_gear(gear, **kwargs)

        gear.set_mesh_markers_and_tags(mesh, cell_marker, facet_marker, magnet_subdomain_tags, \
            magnet_boundary_subdomain_tags)


    def set_domain_size(self):
        """Set domain size of the problem."""
        self._domain_size = self.D + self.gear_1.outer_radius + self.gear_2.outer_radius
