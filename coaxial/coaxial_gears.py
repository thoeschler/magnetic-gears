import dolfin as dlf
from dolfin import LagrangeInterpolator
import numpy as np
import os
import json
from source.magnetic_gear_classes import MagneticBarGear, MagneticGear
from source.tools import create_reference_mesh, interpolate_field, write_hdf5_file, read_hd5f_file
from source.mesh_tools import read_mesh


class CoaxialGearsProblem:
    def __init__(self, first_gear: MagneticGear, second_gear: MagneticGear, D, main_dir=None):
        """Base class for Coaxial Gears problem.

        Args:
            first_gear (MagneticGear): First magnetic gear.
            second_gear (MagneticGear): Second magnetic gear.
            D (float): Distance between gears (from circumference).
            main_dir (str, optional): Main directory. Defaults to None. Current working
                                      directory is used in that case.
        """
        # make sure gears are coaxial
        assert np.isclose(np.dot(first_gear.axis, second_gear.axis), \
            np.linalg.norm(first_gear.axis) * np.linalg.norm(second_gear.axis))
        # set write and read directory
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert os.path.exists(main_dir)
            self._main_dir = main_dir
        assert D > first_gear.outer_radius + second_gear.outer_radius
        self._D = D
        self._gear_1 = first_gear
        self._gear_2 = second_gear
        # overwrite in subclass
        assert hasattr(self._gear_1, "outer_radius")
        assert hasattr(self._gear_2, "outer_radius")
        vec = self.gear_2.x_M - self.gear_1.x_M
        assert np.isclose(np.dot(first_gear.axis, vec), 0.), "Gears are seperated in direction of rotation axis!"
        assert not np.allclose(vec, 0.)
        vec /= np.linalg.norm(vec)
        self.gear_2.x_M = self.gear_1.x_M + D * vec
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
        par = {
            "magnet_type": gear.magnet_type,
            "domain_radius": domain_radius / gear.scale_parameter,
        }
        if isinstance(gear, MagneticBarGear):
            par.update({
                "w": gear.w / gear.h,
                "d": gear.d / gear.h
                })
        return par

    def align_gears(self):
        """Rotate both gears such that magnets align."""
        self.align_gear(self.gear_1, self.gear_2)
        self.align_gear(self.gear_2, self.gear_1)

    def align_gear(self, gear, other_gear):
        """Align gear with another gear.

        Args:
            gear (MagneticGear): Gear that should be aligned.
            other_gear (MagneticGear): Gear that the other should be aligned with.
        """
        assert hasattr(gear, "_magnets")
        vec = (other_gear.x_M - gear.x_M)
        vec /= np.linalg.norm(vec)
        x_M_magnet = gear.x_M + gear.R * vec  # goal position for magnet
        angle = np.arccos(np.dot(gear.magnets[0].x_M - gear.x_M, x_M_magnet - gear.x_M) / gear.R ** 2)
        sign = np.sign(np.cross(gear.magnets[0].x_M - gear.x_M, x_M_magnet - gear.x_M).dot(gear.axis))
        gear.update_parameters(sign * angle)
        assert np.allclose(x_M_magnet, gear.magnets[0].x_M)
        gear.reset_angle(0.)

    def _find_reference_files(self, gear, mesh_size_min, mesh_size_max, rtol=1.0):
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
                        self._match_reference_parameters(par, gear, mesh_size_min, mesh_size_max):
                        found_dir = True
                        domain_radius_file = par["domain_radius"]
                        dir_name = subdir
        return dir_name, found_dir

    def _match_reference_parameters(self, par_file, gear, mesh_size_min, mesh_size_max):
        # check if magnet type agrees
        if not par_file["magnet_type"] == gear.magnet_type:
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
            if not np.allclose((gear.w / gear.scale_parameter, gear.d / gear.scale_parameter), (par_file["w"], par_file["d"])):
                return False

        return True

    def _load_reference_field(self, gear, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, domain_size):
        """Load reference field from hdf5 file. If no appropriate file is
           found the file we be written first.

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
        assert field_name in ("Vm", "B"), "I do not know this field."
        vector_valued = (field_name == "B")  # check if field is vector valued

        # find directory with matching files
        ref_dir, found_dir = self._find_reference_files(gear, mesh_size_min, mesh_size_max)

        # check if both mesh file and hdf5 file exist; if not, create both
        if not found_dir:
            # create directory
            subdirs = [r[0] for r in os.walk(self._main_dir + "data/reference")]
            ref_dir = f"{self._main_dir}/data/reference/{gear.magnet_type}_{int(domain_size)}_{np.random.randint(10_000, 50_000)}"
            while ref_dir in subdirs:
                ref_dir = f"{self._main_dir}/data/reference/{gear.magnet_type}_R_{int(domain_size)}_{np.random.randint(10_000, 50_000)}"
            os.makedirs(ref_dir)

            # create reference mesh
            ref_mag = gear.reference_magnet()
            create_reference_mesh(ref_mag, domain_size / gear.scale_parameter, mesh_size_min, mesh_size_max, fname=f"{ref_dir}/reference_mesh")

            # read the reference mesh
            reference_mesh = read_mesh(f"{ref_dir}/reference_mesh.xdmf")

            # create reference magnet and check if the field is implemented
            assert hasattr(ref_mag, field_name), f"{field_name} is not implemented for this magnet class"

            # interpolate reference field
            if field_name == "B":
                field_interpol = interpolate_field(ref_mag.B, reference_mesh, cell_type, p_deg, f"{ref_dir}/{field_name}", write_pvd=True)
            elif field_name == "Vm":
                field_interpol = interpolate_field(ref_mag.Vm, f"{ref_dir}/{field_name}", write_pvd=True)
            else:
                raise RuntimeError()

            # write the field to hdf5 file
            write_hdf5_file(field_interpol, reference_mesh, fname=f"{ref_dir}/{field_name}.h5", field_name=field_name)

            # write parameter file
            with open(f"{ref_dir}/par.json", "w") as f:
                ref_par = self._reference_parameters_dict(gear, domain_size)
                ref_par.update({
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

    def interpolate_field_gear(self, field_gear, mesh_gear, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max):
        """Interpolate a field of the magnets of a gear on a given mesh of another gear.

        Args:
            field_gear (Magnetic gear): The magnetic gear that owns the field.
            mesh_gear (Magnetic gear): The magnetic gear which owns the mesh.
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
        if not hasattr(field_gear, f"_{field_name}_ref"):
            self._load_reference_field(field_gear, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, self._domain_size)

        # create reference field handler
        if field_name == "B":
            ref_field = field_gear._B_ref
            # new function space
            V = dlf.VectorFunctionSpace(mesh_gear.mesh, cell_type, p_deg)
        elif field_name == "Vm":
            ref_field = field_gear._Vm_ref
            # new function space
            V = dlf.FunctionSpace(mesh_gear.mesh, cell_type, p_deg)
        else:
            raise RuntimeError()

        assert hasattr(self, "_interpolate_field_magnet")

        # initialize the sum over all fields
        field_sum = 0.

        print(f"Interpolating magnetic field... ", end="")
        # interpolate field for every magnet and add it to the sum
        for mag in field_gear.magnets:
            if np.linalg.norm(mag.x_M - mesh_gear.x_M) <= self.D:
                interpol_field = self._interpolate_field_magnet(mag, ref_field, field_gear._B_reference_mesh, \
                    mesh_gear.mesh, cell_type, p_deg)
                field_sum += interpol_field._cpp_object.vector()

        print("Done.")
        return dlf.Function(V, field_sum)

    def _interpolate_field_magnet(self, magnet, ref_field, reference_mesh, mesh, cell_type, p_deg):
        # copy everything
        reference_mesh_copy = dlf.Mesh(reference_mesh)
        V_ref = dlf.VectorFunctionSpace(reference_mesh_copy, cell_type, p_deg)
        reference_field_copy = dlf.Function(V_ref, ref_field._cpp_object.vector())
        V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)

        # scale, rotate and shift reference mesh according to magnet placement
        # rotate first, then shift!
        reference_mesh_copy.coordinates()[:] = magnet.Q.dot(reference_mesh_copy.coordinates().T).T
        reference_mesh_copy.translate(dlf.Point(*magnet.x_M))

        # interpolate field to new function space and add the result
        interpol_field = dlf.Function(V)
        LagrangeInterpolator.interpolate(interpol_field, reference_field_copy)

        return interpol_field

    def compute_torque_on_gear(self, gear, B):
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
        if gear is self.gear_1:
            x_M_ref = self.gear_2.x_M
        elif gear is self.gear_2:
            x_M_ref = self.gear_1.x_M
        else:
            raise RuntimeError()
        D = np.linalg.norm(gear.x_M - x_M_ref)
        for mag, tag in zip(gear.magnets, gear._magnet_boundary_subdomain_tags):
            if np.linalg.norm(mag.x_M - x_M_ref) <= D:
                M = dlf.as_vector(mag.M)  # magnetization
                t = dlf.cross(dlf.cross(gear.normal_vector('+'), M), B)  # traction vector
                m = dlf.cross(x - x_M, t)  # torque density
                tau_expr = m[0] * gear.dA(tag)
                tau += dlf.assemble(tau_expr)  # add to torque

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

        mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags, \
            box_subdomain_tag = gear.mesh_gear(gear, **kwargs)

        gear.set_mesh_markers_and_tags(mesh, cell_marker, facet_marker, magnet_subdomain_tags, \
            magnet_boundary_subdomain_tags, box_subdomain_tag, padding=kwargs["padding"])

        if hasattr(self.gear_1, "_domain_radius") and hasattr(self.gear_2, "_domain_radius"):
            assert hasattr(self, "_set_domain_size")
            self._set_domain_size()

    def _set_domain_size(self):
        # set domain size (the maximum distance between two points on either of the two meshes)
        assert hasattr(self.gear_1, "domain_radius")
        assert hasattr(self.gear_2, "domain_radius")
        self._domain_size = self.D + self.gear_1.domain_radius + self.gear_2.domain_radius

    def update_angles(self, d_angle_1, d_angle_2):
        """Update the paramaters of the problem.

        Args:
            d_angle_1 (float): Angle increment for first gear.
            d_angle_2 (float): Angle increment for second gear.
        """
        self.gear_1.update_parameters(d_angle_1)
        self.gear_2.update_parameters(d_angle_2)
