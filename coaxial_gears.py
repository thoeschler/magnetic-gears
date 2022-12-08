import dolfin as dlf
import numpy as np
import os
from os import path
from magnet_classes import BallMagnet, BarMagnet
from mesh_generator import read_markers_from_file
from magnetic_gear_classes import MagneticGearWithBallMagnets, MagneticGearWithBarMagnets
from field_interpolator import FieldInterpolator
from dolfin import LagrangeInterpolator
import time


class CoaxialGearsBase:
    def __init__(self, n1, n2, R1, R2, D, x_M_1, magnetization_strength_1,
                 magnetization_strength_2, init_angle_1=0., init_angle_2=0.,
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
            magnetization_strength_1 (float): _description_
            magnetization_strength_2 (float): _description_
            init_angle_1 (float, optional): Initial angle of first gear. Defaults to 0..
            init_angle_2 (float, optional): Initial angle of second gear. Defaults to 0..
            main_dir (str, optional): Main directory. Defaults to None. Current working
                                      directory is used in that case.
        """
        # set write and read directory
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert path.exists(main_dir)
            self._main_dir = main_dir

        self._n1 = n1
        self._n2 = n2
        self._R1 = R1
        self._R2 = R2
        self._D = D
        self._x_M_1 = x_M_1
        self._x_M_2 = x_M_1 + np.array([0., R1 + D + R2, 0.])  # overwrite if needed
        self._M_0_1 = magnetization_strength_1
        self._M_0_2 = magnetization_strength_2
        self._angle_1 = init_angle_1
        self._angle_2 = init_angle_2 

    @property
    def magnet_type(self):
        assert hasattr(self, "_magnet_type")
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

    def B(self, x_0):
        return self.gear_1.B(x_0) + self.gear_2.B(x_0)

    def _create_gears(self):
        "Purely virtual method"
        pass

    def _reference_magnet(self):
        "Purely virtual method."
        pass

    def _reference_field_file_name(self, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, domain_size):
        """Get file name of the reference field that is used for interpolation.

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


        Returns:
            str: The file name of the reference field (hdf5).
        """
        assert hasattr(self, "magnet_type")
        assert isinstance(field_name, str)
        assert isinstance(p_deg, int)
        assert isinstance(cell_type, str)
        assert isinstance(mesh_size_min, float)
        assert isinstance(mesh_size_min, float)
        assert isinstance(domain_size, int)
        assert field_name in ("V_m", "B")

        return f"{field_name}_{cell_type}_{p_deg}_{str(mesh_size_min).replace('.', 'p')}" + \
            f"_{str(mesh_size_max).replace('.', 'p')}_{self.magnet_type}_{domain_size}.h5"
    
    def _reference_mesh_file_name(self, mesh_size_min, mesh_size_max, domain_size):
        """Get file name of the reference mesh that is used for interpolation.

        Args:
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
            str: The name of the reference mesh file (xdmf).
        """
        assert hasattr(self, "magnet_type")
        assert isinstance(mesh_size_min, float)
        assert isinstance(mesh_size_max, float)
        return f"reference_mesh_{str(mesh_size_min).replace('.', 'p')}" + \
            f"_{str(mesh_size_max).replace('.', 'p')}_{self.magnet_type}_{int(domain_size)}.xdmf"

    def _gear_mesh_file_name(self, prefix, gear, mesh_size_space, mesh_size_magnets):
        """Get file name of a gear's mesh.

        Args:
            prefix (str): Some prefix (e.g. "gear_1").
            gear (MagneticGear): The magnetic gear.
            mesh_size_space (float): The mesh size for the surrounding space.
            mesh_size_magnets (float): The mesh size for the magnets.

        Returns:
            str: The name of the mesh file (xdmf).
        """
        assert "_" not in prefix  # this would break the logic
        return f"{prefix}_{self.magnet_type.lower()}_{self.n1}_{str(self.R1).replace('.', 'p')}_{str(self.r1).replace('.', 'p')}" \
                + f"_{str(mesh_size_space).replace('.', 'p')}_{str(mesh_size_magnets).replace('.', 'p')}" \
                    + f"_{str(gear.x_M[0]).replace('.', 'p')}_{str(gear.x_M[1]).replace('.', 'p')}" \
                        + f"_{str(gear.x_M[2]).replace('.', 'p')}"

    def _find_reference_field_file(self, reference_name, tol=1.0):
        """Finds an appropriate hdf5 file that contains the reference field if
           one exists.

        Args:
            reference_name (str): The reference file name to look for. The domain
                                  size can differ (according to tol).
            tol (float, optional): Relative tolerance between the problem's domain
                                   size and the file's domain size. If the diffe-
                                   rence is within tolerance, the file will be used.
                                   Defaults to 1.0.

        Returns:
            str: The name of the reference field file (hdf5).
        """
        # some parameters
        file_found = False
        chosen_file_name = None

        # Quotient between the file's and the problem's domain size.
        # Start with a value that is too high (according to tol). 
        domain_size_quotient = 1 + tol + 1e-2
        par_ref = reference_name.rstrip(".h5").split("_")  # reference parameters
        domain_size_ref = float(par_ref[-1])

        # get file names that end in .h5 
        fnames = list(filter(lambda f: f.endswith(".h5"), os.listdir(self._main_dir + "/data/")))
        for fname in fnames:
            par_file = fname.rstrip(".h5").split("_")
            domain_size_file = float(par_file[-1])
            # check if all argument up to the last one (domain size) agree
            # check if file is better than any other so far
            if par_ref[:-1] == par_file[:-1] and domain_size_file >= domain_size_ref \
                and domain_size_file / domain_size_ref < min(1 + tol, domain_size_quotient):
                file_found = True
                domain_size_quotient = domain_size_file / domain_size_ref
                chosen_file_name = fname
        if file_found:
            # if a file was found return its name 
            return chosen_file_name, file_found
        else:
            # if no file was found return the reference name
            return reference_name, file_found

    def _find_reference_mesh_file(self, reference_name, tol=1.0):
        """Finds an appropriate xdmf file that contains the reference mesh if
           one exists.

        Args:
            reference_name (str): The reference file name to look for. The domain
                                  size can differ (according to tol).
            tol (float): The tolerance for the relative difference between a
                         file's domain size and the problem's domain size.
                         Defaults to 1.0.

        Returns:
            str: The name of the reference mesh file (xdmf).
        """
        # some parameters
        file_found = False
        chosen_file_name = None

        # extract parameters from reference file name
        domain_size_quotient = 1 + tol + 1e-2
        par_ref = reference_name.rstrip(".xdmf").split("_")
        domain_size_ref = float(par_ref[-1])

        # get file names that end in .xdmf
        fnames = list(filter(lambda f: f.endswith(".xdmf"), os.listdir(self._main_dir + "/meshes/reference/")))
        for fname in fnames:
            par_file = fname.rstrip(".xdmf").split("_")
            domain_size_file = float(par_file[-1])
            # check if all parameters up to the last one (domain size) agree
            # check if file is better than any other so far
            if par_ref[:-1] == par_file[:-1] and domain_size_file >= domain_size_ref \
                and domain_size_file / domain_size_ref < min(1 + tol, domain_size_quotient):
                file_found = True
                domain_size_quotient = domain_size_file / domain_size_ref
                chosen_file_name = fname
        if file_found:
            # if a file was found return its name 
            return chosen_file_name, file_found
        else:
            # if no file was found return the reference name
            return reference_name, file_found

    def _find_gear_mesh_file(self, reference_name):
        """"""
        """Finds an appropriate xdmf file that contains the gear's mesh if
           one exists.

        Args:
            reference_name (str): The reference file name to look for. The domain
                                  size can differ (according to tol).
        Returns:
            str: The name of the gear mesh file (xdmf).
        """
        # some parameters
        file_found = False

        # extract parameters from reference file name
        par_ref = reference_name.rstrip(".xdmf").split("_")

        # get file names that end in .xdmf
        fnames = list(filter(lambda f: f.endswith(".xdmf"), os.listdir(self._main_dir + "/meshes/gears/")))
        for fname in fnames:
            par_file = fname.rstrip(".xdmf").split("_")
            # check if all parameters up to the last three (gear's mid point coordinates) agree
            if par_ref[:-3] == par_file[:-3]:
                file_found = True
                return fname, file_found
        
        # if no file was found return the reference name
        return reference_name, file_found

    def _load_reference_field(self, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, domain_size):
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
        assert field_name in ("V_m", "B"), "I do not know this field."
        vector_valued = field_name == "B"  # check if field is vector valued
        domain_size = int(domain_size)  # cast domain size to int

        # create field interpolator
        fi = FieldInterpolator(domain_size, cell_type, p_deg, mesh_size_min, mesh_size_max)

        # find hdf5 file if there exists one for the given domain size
        fname_reference_field = self._reference_field_file_name(field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, domain_size)
        # compare exisiting file names with this reference name
        fname_reference_field, file_found = self._find_reference_field_file(fname_reference_field)

        # get domain size of hdf5 file
        if file_found:  # if a file was found use the domain size of that file
            domain_size_file = int(fname_reference_field.rstrip(".h5").split("_")[-1])
        else:  # if no file was found write hdf5 file with the problem's domain size
            domain_size_file = domain_size
            
        # get file name for reference mesh
        fname_reference_mesh = self._reference_mesh_file_name(mesh_size_min, mesh_size_max, domain_size_file)

        # check if both mesh file and hdf5 file exist; if not, create both
        if not file_found or not os.path.exists(self._main_dir + "/meshes/reference/" + fname_reference_mesh):
            # create reference mesh
            fi.create_reference_mesh(fname=fname_reference_mesh.rstrip(".xdmf"), type_classifier=self.magnet_type)

            # read the reference mesh
            fi.read_reference_mesh(fname_reference_mesh)
            self._reference_mesh = dlf.Mesh(fi.mesh)

            # create reference magnet and check if the field is implemented
            assert hasattr(self, "_reference_magnet")
            ref_mag = self._reference_magnet()
            assert hasattr(ref_mag, field_name), f"{field_name} is not implemented for this magnet class"

            # interpolate reference field
            if field_name == "B":
                field_interpol = fi.interpolate_reference_field(ref_mag.B, field_name, write_pvd=True)
            elif field_name == "V_m":
                field_interpol = fi.interpolate_reference_field(ref_mag.V_m, field_name, write_pvd=True)
            else:
                raise RuntimeError()
            
            # write the field to hdf5 file
            fi.write_hdf5_file(field_interpol, fname_reference_field, field_name)
        else:
            # read the reference mesh
            fi.read_reference_mesh(fname_reference_mesh)
            self._reference_mesh = dlf.Mesh(fi.mesh)

        # read reference field from hd5f file
        if field_name == "B":
            self._B_ref = fi.read_hd5f_file(fname_reference_field, field_name, vector_valued=vector_valued)
        elif field_name == "V_m":
            self._V_m_ref = fi.read_hd5f_file(fname_reference_field, field_name, vector_valued=vector_valued)
        else:
            raise RuntimeError()

    def interpolate_field_gear(self, gear, mesh, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, parallel=False):
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
        assert field_name in ("B", "V_m")
        assert hasattr(self, "_domain_size")
  
        # load reference field if not done yet
        if not hasattr(self, f"_{field_name}_ref"):
            self._load_reference_field(field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, self._domain_size)

        # create reference field handler
        if field_name == "B":
            ref_field = self._B_ref
            # new function space
            V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)
        elif field_name == "V_m":
            ref_field = self._V_m_ref
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

        print(f"Interpolating magnetic field ... ", end="")
        # interpolate field for every magnet and add it to the sum
        for mag in gear.magnets:
            if np.linalg.norm(mag.x_M - x_M_ref) < midpoint_diff:
                interpol_field = self._interpolate_field_magnet(mag, ref_field, self._reference_mesh, mesh, cell_type, p_deg, parallel=False)
                field_sum += interpol_field._cpp_object.vector()
        print("Done.")

        return dlf.Function(V, field_sum)

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

        for mag, tag in zip(gear.magnets, gear.boundary_subdomain_tags):
            M = dlf.as_vector(mag.M)  # magnetization
            t = dlf.cross(dlf.cross(gear.normal_vector('+'), M), B)  # traction vector
            m = dlf.cross(x - x_M, t)  # torque density
            tau_expr = m[0] * gear.dA(tag)
            tau += dlf.assemble(tau_expr)  # add to torque

        print("Done.")
        return tau
 
    def create_gear_meshes(self, mesh_size_space, mesh_size_magnets, write_to_file=True, verbose=False):
        """Mesh both gears.

        Args:
            mesh_size_space (str): Mesh size for the surrounding space.
            mesh_size_magnets (str): Mesh size for the magnets.
            write_to_file (bool, optional): If true write meshes to paraview-files.
                                            Defaults to True.
            verbose (bool, optional): If true the gmsh meshing information will be
                                      displayed. Defaults to False.
        """
        assert hasattr(self, "gear_1")
        assert hasattr(self, "gear_2")

        # file names
        fname_1 = self._gear_mesh_file_name("gear1", self.gear_1, mesh_size_space, mesh_size_magnets)
        fname_2 = self._gear_mesh_file_name("gear2", self.gear_2, mesh_size_space, mesh_size_magnets)

        # create directory if it does not exist
        if not path.exists(self._main_dir + "/meshes/gears/"):
            os.mkdir(self._main_dir + "/meshes/gears/")

        for gear, fname in zip((self.gear_1, self.gear_2), (fname_1, fname_2)):
            fname, found_file = self._find_gear_mesh_file(fname)
            # read markers from file
            # if no file was found, mesh the gear first
            if found_file:
                print(f"Reading {fname.split('_')[0]} mesh... ", end="")
                gear._mesh, gear._cell_marker, gear._facet_marker = read_markers_from_file(self._main_dir + "/meshes/gears/" + fname.rstrip(".xdmf"))
                gear._domain_radius = gear.R + gear.r + gear._pad
                gear.subdomain_tags = np.unique(gear._cell_marker.array())[1:]  # exclude first index (the box)
                gear.boundary_subdomain_tags = np.unique(gear._facet_marker.array())[:-1]  # exclude last index (the box mesh)
                gear._set_differential_measures()
                print("Done.")
            else:
                gear.create_mesh(mesh_size_space, mesh_size_magnets, fname=fname, write_to_file=write_to_file, verbose=verbose)

        # set domain size (the maximum distance between two points on either of the two meshes)
        assert hasattr(self.gear_1, "domain_radius")
        assert hasattr(self.gear_2, "domain_radius")
        self._domain_size = self.D + 2 * (self.gear_1.domain_radius + self.gear_2.domain_radius)

    def update_parameters(self, d_angle_1, d_angle_2):
        """Update the paramaters of the problem.

        Args:
            d_angle_1 (float): Angle increment for first gear.
            d_angle_2 (float): Angle increment for second gear.
        """
        self._angle_1 += d_angle_1
        self._angle_2 += d_angle_2 
        self.gear_1.update_parameters(d_angle_1)
        self.gear_2.update_parameters(d_angle_2)


class CoaxialGearsWithBallMagnets(CoaxialGearsBase):
    def __init__(self, n1, n2, r1, r2, R1, R2, D, x_M_1, magnetization_strength_1,
                 magnetization_strength_2, init_angle_1=0., init_angle_2=0.,
                 main_dir=None):
        """Class for Coaxial Gears problem with ball magnets.

        Args:
            r1 (float): Magnet radius in first gear.
            r2 (float): Magnet radius in second gear.
        """
        super().__init__(n1, n2, R1, R2, D, x_M_1, magnetization_strength_1,
                         magnetization_strength_2, init_angle_1=init_angle_1,
                         init_angle_2=init_angle_2, main_dir=main_dir)
        self._magnet_type = "Ball"
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
        self.gear_1 = MagneticGearWithBallMagnets(self.n1, self.r1, self.R1, self.x_M_1,
                                                  self.M_0_1, self.angle_1, index=1)
        self.gear_2 = MagneticGearWithBallMagnets(self.n2, self.r2, self.R2, self.x_M_2,
                                                  self.M_0_2, self.angle_2, index=2)
        print("Done.")
    
    def _reference_magnet(self):
        return BallMagnet(radius=1.0, magnetization_strength=1.0, position_vector=np.zeros(3),
                          rotation_matrix=np.eye(3))

    def _interpolate_field_magnet(self, magnet, ref_field, reference_mesh, mesh, cell_type, p_deg, parallel=True):
        # copy everything
        cell_type_copy = str(cell_type)
        p_deg_copy = int(p_deg)
        parallel_copy = bool(parallel)
        mesh_copy = dlf.Mesh(mesh)
        reference_mesh_copy = dlf.Mesh(reference_mesh)
        V_ref = dlf.VectorFunctionSpace(reference_mesh_copy, cell_type_copy, p_deg_copy)
        reference_field_copy = dlf.Function(V_ref, ref_field._cpp_object.vector())
        mesh_copy = dlf.Mesh(mesh)
        V_copy = dlf.VectorFunctionSpace(mesh_copy, cell_type_copy, p_deg_copy)

        # scale, rotate and shift reference mesh according to magnet placement
        # rotate first, then shift!
        reference_mesh_copy.scale(magnet.R)  # scale reference field by magnet radius
        reference_mesh_copy.coordinates()[:] = magnet.Q.dot(reference_mesh_copy.coordinates().T).T
        reference_mesh_copy.translate(dlf.Point(*magnet.x_M))

        # interpolate field to new function space and add the result
        # interpol_field = dlf.interpolate(reference_field_copy, V_copy)
        interpol_field = dlf.Function(V_copy)
        LagrangeInterpolator.interpolate(interpol_field, reference_field_copy)

        if parallel_copy:
            self._interpol_fields.append(interpol_field)
        else:
            return interpol_field


class CoaxialGearsWithBarMagnets(CoaxialGearsBase):
    def __init__(self, n1, n2, h1, h2, w1, w2, d1, d2, R1, R2, D, x_M_1,
                 magnetization_strength_1, magnetization_strength_2, 
                 initial_angle_1=0., initial_angle_2=0., main_dir=None):
        super().__init__(n1, n2, R1, R2, D, x_M_1, magnetization_strength_1,
                         magnetization_strength_2, initial_angle_1=initial_angle_1,
                         initial_angle_2=initial_angle_2, main_dir=main_dir)
        self._magnet_type = "Bar"
        self._h1 = h1
        self._h2 = h2
        self._w1 = w1
        self._w2 = w2
        self._d1 = d1
        self._d2 = d2
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
        print("Creating gears ... ")
        self.gear_1 = MagneticGearWithBarMagnets(self.n1, self.h1, self.w1, self.d1, self.R1, self.x_M_1,
                                                self.M_0_1, self._alpha_1)
        self.gear_2 = MagneticGearWithBarMagnets(self.n2, self.h2, self.w2, self.d2, self.R2, self.x_M_2,
                                                self.M_0_2, self._alpha_2)
        print("Done.")


if __name__ == '__main__':

    par_ball = {"n1": 12,
        "n2": 16,
        "r1": 1.,
        "r2": 1.5,
        "R1": 8.0,
        "R2": 12.0,
        "D": 1.0,
        "x_M_1": np.array([0., 0., 0.]),
        "magnetization_strength_1": 1.,
        "magnetization_strength_2": 1.,
        "init_angle_1": 0.,
        "init_angle_2": 0.
        }

    CoaxialGears = CoaxialGearsWithBallMagnets(**par_ball)
    CoaxialGears.create_gear_meshes(mesh_size_space=1.0, mesh_size_magnets=0.2, write_to_file=True, verbose=False)

    n_iterations = 19
    d_alpha = 2. * np.pi / par_ball["n1"] / n_iterations
 
    for _ in range(n_iterations):

        start = time.perf_counter()
        B2 = CoaxialGears.interpolate_field_gear(CoaxialGears.gear_2, CoaxialGears.gear_1.mesh, "B", "CG", 1, 0.3, 4.0)
        B1 = CoaxialGears.interpolate_field_gear(CoaxialGears.gear_1, CoaxialGears.gear_2.mesh, "B", "CG", 1, 0.3, 4.0)
        end = time.perf_counter()
        print(end - start)

        tau1 = CoaxialGears.compute_torque_on_gear(CoaxialGears.gear_1, B2)
        tau2 = CoaxialGears.compute_torque_on_gear(CoaxialGears.gear_2, B1)
        print(tau1, tau2)
        with open("file.csv", "a+") as f:
            f.write(f"{CoaxialGears.angle_1} {tau1} {tau2}\n")
        CoaxialGears.update_parameters(d_alpha, 0)