import dolfin as dlf
from dolfin import LagrangeInterpolator
import numpy as np
import os
import json
from source.magnetic_gear_classes import MagneticGearWithBallMagnets, MagneticGearWithBarMagnets, MagneticGear
from source.field_interpolator import FieldInterpolator


class CoaxialGearsBase:
    def __init__(self, n1, n2, R1, R2, D, x_M_1, magnetization_strength_1,
                 magnetization_strength_2, magnet_type, init_angle_1=0.0, init_angle_2=0.0,
                 main_dir=None):
        """Base class for Coaxial Gears problem.

        Args:
            n1 (int): Number of magnets in first gear.
            n2 (int): Number of magnets in second gear.
            R1 (float): Radius of first gear.
            R2 (float_): Radius of second gear.
            D (float): Distance between gears (from circumference).
            x_M_1 (list): Midpoint of first gear. The midpoint of the second is computed
                          based on the geometry.
            magnetization_strength_1 (float): Magnetization strength of first gear.
            magnetization_strength_2 (float): Magnetization strength of second gear.
            init_angle_1 (float, optional): Initial angle of first gear. Defaults to 0.0.
            init_angle_2 (float, optional): Initial angle of second gear. Defaults to 0.0.
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

        self._n1 = n1
        self._n2 = n2
        self._R1 = R1
        self._R2 = R2
        self._D = D
        self._x_M_1 = x_M_1
        # compute midpoint of second gear based on geometry
        # overwrite in subclass if needed
        self._x_M_2 = x_M_1 + np.array([0., R1 + D + R2, 0.])
        self._M_0_1 = magnetization_strength_1
        self._M_0_2 = magnetization_strength_2
        self._magnet_type = magnet_type
        self._angle_1 = init_angle_1
        self._angle_2 = init_angle_2 

    @property
    def magnet_type(self):
        return self._magnet_type

    @property
    def n1(self):
        return self._n1

    @property
    def n2(self):
        return self._n2
    
    @property
    def R1(self):
        return self._R1
        
    @property
    def R2(self):
        return self._R2

    @property
    def D(self):
        return self._D

    @property
    def x_M_1(self):
        return self._x_M_1

    @property
    def x_M_2(self):
        return self._x_M_2

    @property
    def M_0_1(self):
        return self._M_0_1
    
    @property
    def M_0_2(self):
        return self._M_0_2

    @property
    def angle_1(self):
        return self._angle_1

    @property
    def angle_2(self):
        return self._angle_2

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

    def _reference_parameters_dict(self, gear, domain_radius):
        par = {
            "magnet_type": gear.magnet_type,
            "domain_radius": domain_radius,
        }
        return par

    def _create_gears(self):
        "Purely virtual method."
        pass

    def _find_reference_files(self, gear, rtol=1.0):
        assert hasattr(self, "_domain_size")
        # some parameters
        found_dir = False
        dir_name = None
        # start with some domain radius that is too large
        domain_radius_file = (1 + rtol + 1e-2) * self._domain_size

        # search existing gear mesh directories for matching parameter file
        subdirs = [r[0] for r in os.walk(self._main_dir + "/data/reference/")]
        for subdir in subdirs:
            if "par.json" in os.listdir(subdir):
                with open(f"{subdir}/par.json", "r") as f:
                    # read paramters
                    par = json.loads(f.read())
                    # check if all paramters match
                    if par["domain_radius"] < domain_radius_file and self._match_reference_parameters(gear, par):
                        found_dir = True
                        domain_radius_file = par["domain_radius"]
                        dir_name = subdir
        return dir_name, found_dir

    def _find_gear_mesh_file(self, gear):
        # some parameters
        found_file = False
        par_ref = gear.parameters

        # search existing gear mesh directories for matching parameter file
        subdirs = [r[0] for r in os.walk(self._main_dir + "/data/gears/")]
        for subdir in subdirs:
            if "par.json" in os.listdir(subdir):
                with open(f"{subdir}/par.json", "r") as f:
                    # read paramters
                    par = json.loads(f.read())
                    if par["magnet_type"] != gear.magnet_type:
                        continue
                    # get paramters that need to match (all except x_M and angle)
                    match_par = [p for p in par.keys() if p not in ("x_M", "angle")]
                    # if all paramters match, return the subdir
                    if all([par[p] == par_ref[p] for p in match_par]):
                        found_file = True
                        return subdir, found_file
        return None, found_file

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
        vector_valued = field_name == "B"  # check if field is vector valued
        domain_size = int(domain_size)  # cast domain size to int

        # create field interpolator
        fi = FieldInterpolator(domain_size, cell_type, p_deg, mesh_size_min, mesh_size_max)

        # find directory with matching files
        ref_dir, found_dir = self._find_reference_files(gear)

        # check if both mesh file and hdf5 file exist; if not, create both
        if not found_dir:
            # create directory
            subdirs = [r[0] for r in os.walk(self._main_dir + "data/reference")]
            ref_dir = f"{self._main_dir}/data/reference/{self.magnet_type}_{domain_size}_{np.random.randint(10_000, 50_000)}"
            while ref_dir in subdirs:
                ref_dir = f"{self._main_dir}/data/reference/{self.magnet_type}_R_{domain_size}_{np.random.randint(10_000, 50_000)}"
            os.makedirs(ref_dir)

            # create reference mesh
            fi.create_reference_mesh(reference_magnet=gear.reference_magnet(), fname=f"{ref_dir}/reference_mesh")

            # read the reference mesh
            fi.read_reference_mesh(f"{ref_dir}/reference_mesh.xdmf")
            gear.set_reference_mesh(dlf.Mesh(fi.mesh), field_name)

            # create reference magnet and check if the field is implemented
            ref_mag = gear.reference_magnet()
            assert hasattr(ref_mag, field_name), f"{field_name} is not implemented for this magnet class"

            # interpolate reference field
            if field_name == "B":
                field_interpol = fi.interpolate_reference_field(ref_mag.B, f"{ref_dir}/{field_name}", write_pvd=True)
            elif field_name == "Vm":
                field_interpol = fi.interpolate_reference_field(ref_mag.Vm, f"{ref_dir}/{field_name}", write_pvd=True)
            else:
                raise RuntimeError()

            # write the field to hdf5 file
            fi.write_hdf5_file(field_interpol, fname=f"{ref_dir}/{field_name}.h5", field_name=field_name)

            # write parameter file
            with open(f"{ref_dir}/par.json", "w") as f:
                f.write(json.dumps(self._reference_parameters_dict(gear, domain_size)))
        else:
            # read the reference mesh
            fi.read_reference_mesh(f"{ref_dir}/reference_mesh.xdmf")
            gear.set_reference_mesh(dlf.Mesh(fi.mesh), field_name)

        # read reference field from hd5f file
        reference_field = fi.read_hd5f_file(f"{ref_dir}/{field_name}.h5", field_name, vector_valued=vector_valued)

        # set reference field and mesh for gear
        gear.set_reference_field(reference_field, field_name)
        gear.set_reference_mesh(dlf.Mesh(fi.mesh), field_name)

    def interpolate_field_gear(self, gear, mesh, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max):
        """Interpolate a field of all magnets of a gear on a given mesh.

        Args:
            gear (Magnetic gear): The magnetic gear.
            mesh (dlf.Mesh): The mesh on which to interpolate the field.
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
        if field_name == "B":
            ref_field = gear._B_ref
            # new function space
            V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)
        elif field_name == "Vm":
            ref_field = gear._Vm_ref
            # new function space
            V = dlf.FunctionSpace(mesh, cell_type, p_deg)
        else:
            raise RuntimeError()

        assert hasattr(self, "_interpolate_field_magnet")

        # initialize the sum over all fields
        field_sum = 0.

        # get midpoint of the other magnetic gear
        assert gear.index in (1, 2)
        x_M_ref = self.gear_1.x_M if gear.index == 2 else self.gear_2.x_M
        midpoint_diff = np.linalg.norm(self.gear_1.x_M - self.gear_2.x_M)

        print(f"Interpolating magnetic field... ", end="")
        # interpolate field for every magnet and add it to the sum
        for mag in gear.magnets:
            if np.linalg.norm(mag.x_M - x_M_ref) < midpoint_diff:
                interpol_field = self._interpolate_field_magnet(mag, ref_field, gear._B_reference_mesh, mesh, cell_type, p_deg)
                field_sum += interpol_field._cpp_object.vector()
        print("Done.")
        return dlf.Function(V, field_sum)

    def _interpolate_field_magnet(self, magnet, ref_field, reference_mesh, mesh, cell_type, p_deg):
        # copy everything
        cell_type_copy = str(cell_type)
        p_deg_copy = int(p_deg)
        mesh_copy = dlf.Mesh(mesh)
        reference_mesh_copy = dlf.Mesh(reference_mesh)
        V_ref = dlf.VectorFunctionSpace(reference_mesh_copy, cell_type_copy, p_deg_copy)
        reference_field_copy = dlf.Function(V_ref, ref_field._cpp_object.vector())
        mesh_copy = dlf.Mesh(mesh)
        V_copy = dlf.VectorFunctionSpace(mesh_copy, cell_type_copy, p_deg_copy)

        # scale, rotate and shift reference mesh according to magnet placement
        # rotate first, then shift!
        self._scale_mesh(reference_mesh_copy, magnet)
        reference_mesh_copy.coordinates()[:] = magnet.Q.dot(reference_mesh_copy.coordinates().T).T
        reference_mesh_copy.translate(dlf.Point(*magnet.x_M))

        # interpolate field to new function space and add the result
        # interpol_field = dlf.interpolate(reference_field_copy, V_copy)
        interpol_field = dlf.Function(V_copy)
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

        for mag, tag in zip(gear.magnets, gear._magnet_boundary_subdomain_tags):
            M = dlf.as_vector(mag.M)  # magnetization
            t = dlf.cross(dlf.cross(gear.normal_vector('+'), M), B)  # traction vector
            m = dlf.cross(x - x_M, t)  # torque density
            tau_expr = m[0] * gear.dA(tag)
            tau += dlf.assemble(tau_expr)  # add to torque

        print("Done.")
        return tau
 
    def set_gear_meshes(self, mesh_size_space, mesh_size_magnets, write_to_pvd=True, verbose=False):
        """Mesh both gears.

        Args:
            mesh_size_space (str): Mesh size for the surrounding space.
            mesh_size_magnets (str): Mesh size for the magnets.
            write_to_pvd (bool, optional): If true write meshes to paraview-files.
                                           Defaults to True.
            verbose (bool, optional): If true the gmsh meshing information will be
                                      displayed. Defaults to False.
        """
        assert hasattr(self, "_gear_1")
        assert hasattr(self, "_gear_2")

        # create directory if it does not exist
        if not os.path.exists(self._main_dir + "/data/gears/"):
            os.makedirs(self._main_dir + "/data/gears/")

        for gear in self.gears:
            dir_name, found_file = self._find_gear_mesh_file(gear)
            # read markers and mesh from file
            # if no file was found, generate both 
            if found_file:
                print(f"Reading gear mesh... ", end="")
                gear.set_mesh_and_markers_from_file(f"{dir_name}/gear")
                print("Done.")
            else:
                # generate dir_name
                subdirs = [r[0] for r in os.walk(self._main_dir + "/data/gears/")]
                # generate some random integer value
                dir_name = f"{gear.magnet_type}_n_{gear.n}_{np.random.randint(10_000, 50_000)}"
                while dir_name in subdirs:
                    dir_name = f"{gear.magnet_type}_n_{gear.n}_{np.random.randint(10_000, 50_000)}"

                os.mkdir(f"{self._main_dir}/data/gears/{dir_name}")
                gear.generate_mesh_and_markers(mesh_size_space, mesh_size_magnets, fname=f"{dir_name}/gear", \
                    write_to_pvd=write_to_pvd, verbose=verbose)
                # write parameter file
                with open(f"{self._main_dir}/data/gears/{dir_name}/par.json", "w") as f:
                    f.write(json.dumps(gear.parameters))

        self._set_domain_size()

    def update_parameters(self, d_angle_1, d_angle_2):
        """Update the paramaters of the problem.

        Args:
            d_angle_1 (float): Angle increment for first gear.
            d_angle_2 (float): Angle increment for second gear.
        """
        self._angle_1 += d_angle_1
        self._angle_2 += d_angle_2 
        self._gear_1.update_parameters(d_angle_1)
        self._gear_2.update_parameters(d_angle_2)


class CoaxialGearsWithBallMagnets(CoaxialGearsBase):
    def __init__(self, n1, n2, r1, r2, R1, R2, D, x_M_1, magnetization_strength_1,
                 magnetization_strength_2, magnet_type="Ball", init_angle_1=0.,
                 init_angle_2=0., main_dir=None):
        """Class for Coaxial Gears problem with ball magnets.

        Args:
            r1 (float): Magnet radius in first gear.
            r2 (float): Magnet radius in second gear.
        """
        super().__init__(n1, n2, R1, R2, D, x_M_1, magnetization_strength_1,
                         magnetization_strength_2, magnet_type, init_angle_1,
                         init_angle_2, main_dir)
        self._r1 = r1
        self._r2 = r2
        # compute midpoint of second gear from geometry
        self._x_M_2 = x_M_1 + np.array([0., R1 + r1 + D + r2 + R2, 0.])
        self._create_gears()

    @property
    def r1(self):
        return self._r1

    @property
    def r2(self):
        return self._r2

    def _create_gears(self):
        print("Creating gears... ")
        self._gear_1 = MagneticGearWithBallMagnets(self.n1, self.r1, self.R1, self.x_M_1,
                                                   self.M_0_1, self.angle_1, index=1,
                                                   main_dir=self._main_dir)
        self._gear_2 = MagneticGearWithBallMagnets(self.n2, self.r2, self.R2, self.x_M_2,
                                                   self.M_0_2, self.angle_2, index=2.,
                                                   main_dir=self._main_dir)
        print("Done.")

    def _match_reference_parameters(self, gear, par_file):
        # check if magnet type agrees
        if not par_file["magnet_type"] == self.magnet_type:
            return False

        # check if domain size is large enough
        if not par_file["domain_radius"] * gear.R > self._domain_size:
            return False
        
        return True

    def _scale_mesh(self, mesh, magnet):
        mesh.scale(magnet.R)

    def _set_domain_size(self):
        # set domain size (the maximum distance between two points on either of the two meshes)
        assert hasattr(self._gear_1, "domain_radius")
        assert hasattr(self._gear_2, "domain_radius")
        self._domain_size = self.D + 2 * (self._gear_1.domain_radius + self._gear_2.domain_radius)


class CoaxialGearsWithBarMagnets(CoaxialGearsBase):
    def __init__(self, n1, n2, h1, h2, w1, w2, d1, d2, R1, R2, D, x_M_1,
                 magnetization_strength_1, magnetization_strength_2,
                 magnet_type="Bar", init_angle_1=0., init_angle_2=0.,
                 main_dir=None):
        super().__init__(n1, n2, R1, R2, D, x_M_1, magnetization_strength_1,
                         magnetization_strength_2, magnet_type, init_angle_1,
                         init_angle_2, main_dir)
        self._h1 = h1
        self._h2 = h2
        self._w1 = w1
        self._w2 = w2
        self._d1 = d1
        self._d2 = d2
        # compute midpoint of second gear from geometry
        self._x_M_2 = self._x_M_1 + np.array([0., self.R1 + self.w1 + self.D + self.w2 + self.R2, 0.])
        self._create_gears()

    @property
    def h1(self):
        return self._h1

    @property
    def h2(self):
        return self._h2

    @property
    def w1(self):
        return self._w1

    @property
    def w2(self):
        return self._w2

    @property
    def d1(self):
        return self._d1

    @property
    def d2(self):
        return self._d2

    def _create_gears(self):
        print("Creating gears... ")
        self._gear_1 = MagneticGearWithBarMagnets(self.n1, self.h1, self.w1, self.d1, self.R1, self.x_M_1,
                                                  self.M_0_1, self._angle_1, index=1, main_dir=self._main_dir)
        self._gear_2 = MagneticGearWithBarMagnets(self.n2, self.h2, self.w2, self.d2, self.R2, self.x_M_2,
                                                  self.M_0_2, self._angle_2, index=2, main_dir=self._main_dir)
        print("Done.")
    
    def _match_reference_parameters(self, gear, par_file):
        # check if magnet type agrees
        if not par_file["magnet_type"] == self.magnet_type:
            return False

        # check if domain size is large enough
        if not par_file["domain_radius"] * gear.h > self._domain_size:
            return False

        # check if geometry matches
        if not np.allclose((gear.w / gear.h, gear.d / gear.h), (par_file["w"], par_file["d"])):
            return False
        
        return True

    def _reference_parameters_dict(self, gear, domain_radius):
        par = super()._reference_parameters_dict(gear, domain_radius)
        par.update({
            "w": gear.w / gear.h,
            "d": gear.d / gear.h
        })
        return par

    def _scale_mesh(self, mesh, magnet):
        mesh.scale(magnet.h)

    def _set_domain_size(self):
        # set domain size (the maximum distance between two points on either of the two meshes)
        assert hasattr(self._gear_1, "domain_radius")
        assert hasattr(self._gear_2, "domain_radius")
        self._domain_size = self.D + 2 * (self._gear_1.domain_radius + self._gear_2.domain_radius)
