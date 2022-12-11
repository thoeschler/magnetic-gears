import dolfin as dlf
import numpy as np
import os
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet, BarMagnet
from source.gear_mesh_generator import GearWithBallMagnetsMeshGenerator, GearWithBarMagnetsMeshGenerator
from source.mesh_tools import read_markers_from_file


class MagneticGear:
    def __init__(self, n, R, x_M, magnetization_strength, initial_angle, index, magnet_type, main_dir=None):
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert os.path.exists(main_dir)
            self._main_dir = main_dir
        self._n = n  # the number of magnets
        self._R = R  # the radius of the gear
        self._x_M = x_M  # the mid point
        self._M_0 = magnetization_strength  # the magnetization strength
        self._angle = initial_angle  # angle in direction of spin
        self._magnet_type = magnet_type
        self._index = index
        self._mesh_generator = None

    @property
    def n(self):
        return self._n

    @property
    def R(self):
        return self._R

    @property
    def x_M(self):
        return self._x_M

    @property
    def M_0(self):
        return self._M_0

    @property
    def angle(self):
        return self._angle
    
    @property
    def index(self):
        return self._index
    
    @property
    def magnets(self):
        assert hasattr(self, "_magnets")
        return self._magnets

    @property
    def mesh(self):
        assert hasattr(self, "_mesh")
        return self._mesh
    
    @property
    def domain_radius(self):
        assert hasattr(self, "_domain_radius")
        return self._domain_radius

    @property
    def magnet_type(self):
        assert hasattr(self, "_magnet_type")
        return self._magnet_type

    @property
    def normal_vector(self):
        assert hasattr(self, "_normal_vector")
        return self._normal_vector
    
    @property
    def dV(self):
        assert hasattr(self, "_dV")
        return self._dV
    
    @property
    def dA(self):
        assert hasattr(self, "_dA")
        return self._dA

    @property
    def reference_field(self):
        assert hasattr(self, "_reference_field")
        return self._reference_field

    def _create_magnets(self):
        "Purely virtual method."
        pass

    @property
    def parameters(self):
        par = {
            "n": self.n,
            "R": self.R,
            "x_M": tuple(self.x_M),
            "angle": self.angle,
            "magnet_type": self.magnet_type,
        }
        return par

    def generate_mesh_and_markers(self, mesh_size_space, mesh_size_magnets, fname, write_to_pvd=True, verbose=False):
        assert hasattr(self, "_mesh_generator")

        # set mesh, markers and subdomain tags
        mesh_and_marker, tags = self._mesh_generator.generate_mesh(
            mesh_size_space, mesh_size_magnets, fname, write_to_pvd=write_to_pvd, verbose=verbose
            )
        
        # unpack mesh, markers and tags
        self._mesh = mesh_and_marker.mesh
        self._cell_marker, self._facet_marker = mesh_and_marker.cell_marker, mesh_and_marker.facet_marker
        self._magnet_subdomain_tags, self._magnet_boundary_subdomain_tags = tags.magnet_subdomain, tags.magnet_boundary_subdomain
        self._box_subdomain_tag = tags.box_subdomain

        # set differential measures
        self._normal_vector, self._dV, self._dA = self._mesh_generator.get_differential_measures(self._mesh, self._cell_marker, self._facet_marker)

        # set differential measures and domain radius
        self._domain_radius = self._mesh_generator.get_padded_radius()

    def set_mesh_and_markers_from_file(self, file_name):
        """Set mesh and markers from xdmf file.

        Args:
            file_name (str): Absolute path to file.
        """
        self._mesh, self._cell_marker, self._facet_marker = read_markers_from_file(file_name)

        # set subdomain tags
        subdomain_tags = np.sort(np.unique(self._cell_marker.array()))
        self._box_subdomain_tag = subdomain_tags[0]
        self._magnet_subdomain_tags = subdomain_tags[1:]  # exclude first index (the box)
        self._magnet_boundary_subdomain_tags = np.sort(np.unique(self._facet_marker.array()))[:-1]  # exclude last index (the box)

        # set differential measures and domain radius
        self._normal_vector, self._dV, self._dA = self._mesh_generator.get_differential_measures(self._mesh, self._cell_marker, self._facet_marker)
        self._domain_radius = self._mesh_generator.get_padded_radius()

    def set_reference_field(self, reference_field, field_name):
        """Set reference field for magnetic gear.

        Args:
            reference_field (dlf.Function): The reference field (field on a suitable
                                            reference mesh).
        """
        assert isinstance(reference_field, dlf.Function)
        assert isinstance(field_name, str)
        
        if field_name == "B":
            self._B_ref = reference_field
        elif field_name == "Vm":
            self._Vm_ref = reference_field
        else:
            raise RuntimeError()

    def update_parameters(self, d_angle):
        # update the angle
        self._angle += d_angle
        # then update the rest
        self.update_magnets()
        self.update_mesh(d_angle)

    def update_magnets(self):
        assert hasattr(self, "magnets")
        for k, mag in enumerate(self._magnets):
            mag._Q = Rotation.from_rotvec((2 * np.pi / self.n * k + self._angle) * \
                np.array([1., 0., 0.])).as_matrix()
            mag._xM = self.x_M + np.array([0., self.R * np.cos(2 * np.pi / self.n * k + self._angle),
                                           self.R * np.sin(2 * np.pi / self.n * k + self._angle)])
            mag._M = mag._Q.dot(np.array([0., 0., 1.]))

    def update_mesh(self, d_angle):
        """Update mesh coordinates.

        Args:
            d_angle (float): Angle increment in rad.
        """
        # rotate mesh around axis 0 (x-axis) through gear midpoint by angle d_angle
        self.mesh.rotate(d_angle * 180 / np.pi, 0, dlf.Point(*self.x_M))


class MagneticGearWithBallMagnets(MagneticGear):
    def __init__(self, n, r, R, x_M, magnetization_strength, init_angle, index, magnet_type="Ball", main_dir=None):
        super().__init__(n, R, x_M, magnetization_strength, init_angle, index, magnet_type, main_dir)
        self._r = r  # the magnet radius
        self._create_magnets()
        self._mesh_generator = GearWithBallMagnetsMeshGenerator(self, main_dir)

    @property
    def parameters(self):
        par = super().parameters
        par.update({
            "r": self.r
        })
        return par

    @property
    def r(self):
        return self._r

    def reference_magnet(self):
        """Return reference magnet instance.

        Returns:
            BallMagnet: The reference magnet.
        """
        return BallMagnet(radius=1.0, magnetization_strength=1.0, position_vector=np.zeros(3),
                          rotation_matrix=np.eye(3))

    def _create_magnets(self):
        print("Creating magnets... ", end="")

        self._magnets = []
        for k in range(self.n):
            # compute position and rotation matrix
            x_M = self.x_M + np.array([0.,
                                       self.R * np.cos(2 * np.pi / self.n * k + self.angle),
                                       self.R * np.sin(2 * np.pi / self.n * k + self.angle)])
            Q = Rotation.from_rotvec((2 * np.pi / self.n * k + self.angle) * \
                    np.array([1., 0., 0.])).as_matrix()
            self._magnets.append(BallMagnet(self.r, self.M_0, x_M, Q))
        
        print("Done.")

    def get_reference_field_file_name(self, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, domain_size):
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
        assert isinstance(mesh_size_max, float)
        assert isinstance(domain_size, int)
        assert field_name in ("Vm", "B")

        return f"{field_name}_{cell_type}_{p_deg}_{str(mesh_size_min).replace('.', 'p')}" + \
            f"_{str(mesh_size_max).replace('.', 'p')}_{self.magnet_type}_{domain_size}.h5"

    def get_reference_mesh_file_name(self, mesh_size_min, mesh_size_max, domain_size):
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
        assert isinstance(mesh_size_min, float)
        assert isinstance(mesh_size_max, float)
        return f"reference_mesh_{str(mesh_size_min).replace('.', 'p')}" + \
            f"_{str(mesh_size_max).replace('.', 'p')}_{self.magnet_type}_{int(domain_size)}.xdmf"

    def get_padded_radius(self):
        return self.R + self.r + self._mesh_generator.pad


class MagneticGearWithBarMagnets(MagneticGear):
    def __init__(self, n, h, w, d, R, x_M, magnetization_strength, init_angle, index, magnet_type="Bar", main_dir=None):
        super().__init__(n, R, x_M, magnetization_strength, init_angle, index, magnet_type, main_dir)
        self._h = h  # the magnet height
        self._w = w  # the magnet width
        self._d = d  # the magnet depth
        self._create_magnets()
        self._mesh_generator = GearWithBarMagnetsMeshGenerator(self, main_dir)

    @property
    def h(self):
        return self._h

    @property
    def w(self):
        return self._w

    @property
    def d(self):
        return self._d

    @property
    def parameters(self):
        par = super().parameters
        par.update({
            "h": self.h,
            "w": self.w,
            "d": self.d
        })
        return par
    
    def reference_magnet(self):
        """Return reference magnet instance.

        The reference magnet has a fixed height of 1.0. Width and depth are
        scaled accordingly.

        Returns:
            BarMagnet: The reference magnet.
        """
        return BarMagnet(height=1.0, width=self.w / self.h, depth=self.d / self.h, magnetization_strength=1.0, \
            position_vector=np.zeros(3), rotation_matrix=np.eye(3))

    def _create_magnets(self):
        print("Creating magnets... ")

        self._magnets = []
        for k in range(self.n):
            # compute position and rotation matrix
            pos = self.x_M + np.array([0.,
                                       self.R * np.cos(2 * np.pi / self.n * k + self.angle),
                                       self.R * np.sin(2 * np.pi / self.n * k + self.angle)])
            rot = Rotation.from_rotvec((2 * np.pi / self.n * k + self.angle) * \
                np.array([1., 0., 0.])).as_matrix()
            self._magnets.append(BarMagnet(self.h, self.w, self.d, self.M_0, pos, rot))
        
        print("Done.")

    def get_reference_field_file_name(self, field_name, cell_type, p_deg, mesh_size_min, mesh_size_max, domain_size):
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
        assert isinstance(mesh_size_max, float)
        assert isinstance(domain_size, int)
        assert field_name in ("Vm", "B")
        
        # reference width and depth
        w_ref = self.w / self.h
        d_ref = self.d / self.h

        return f"{field_name}_{cell_type}_{p_deg}_{str(mesh_size_min).replace('.', 'p')}" + \
            f"_{str(mesh_size_max).replace('.', 'p')}_{self.magnet_type}_{str(w_ref).replace('.', 'p'):.4s}" + \
                f"_{str(d_ref).replace('.', 'p'):.4s}_{domain_size}.h5"

    def get_reference_mesh_file_name(self, mesh_size_min, mesh_size_max, domain_size):
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
        assert isinstance(mesh_size_min, float)
        assert isinstance(mesh_size_max, float)

        # reference width and depth
        w_ref = self.w / self.h
        d_ref = self.d / self.h

        return f"reference_mesh_{str(mesh_size_min).replace('.', 'p')}" + \
            f"_{str(mesh_size_max).replace('.', 'p')}_{self.magnet_type}_{str(w_ref).replace('.', 'p'):.4s}" + \
                f"_{str(d_ref).replace('.', 'p'):.4s}_{int(domain_size)}.xdmf"

    def get_padded_radius(self):
        return self.R + self.w + self._mesh_generator.pad


"""    def save_gear_par(self):
        par = {
            "n": self.n,

        }

    def load_gear_par(self):

"""