import dolfin as dlf
import numpy as np
import os
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet, BarMagnet
from source.grid_generator import ball_gear_mesh, bar_gear_mesh
from source.mesh_tools import read_mesh_and_markers


class MagneticGear:
    def __init__(self, n, R, x_M, magnetization_strength, initial_angle, magnet_type):
        self._n = n  # the number of magnets
        self._R = R  # the radius of the gear
        self._x_M = x_M  # the mid point
        self._M_0 = magnetization_strength  # the magnet's magnetization strength
        self._angle = initial_angle  # angle in direction of spin
        self._magnet_type = magnet_type
        self._scale_parameter = None

    @property
    def n(self):
        return self._n

    @property
    def R(self):
        return self._R

    @property
    def x_M(self):
        return self._x_M

    @x_M.setter
    def x_M(self, x_M):
        assert len(x_M) == 3
        # update mesh coordinates
        if hasattr(self, "_mesh"):
            self.translate_mesh(x_M - self.x_M)
        self._x_M = x_M
        # update magnet coordinates
        if hasattr(self, "_magnets"):
            self.update_magnets()

    @property
    def M_0(self):
        return self._M_0

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        assert isinstance(angle, float)
        # update mesh coordinates
        if hasattr(self, "_mesh"):
            self.rotate_mesh(angle - self.angle)
        self._angle = angle
        if hasattr(self, "_magnets"):
            self.update_magnets()

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
    def scale_parameter(self):
        return self._scale_parameter

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
        if hasattr(self, "_domain_radius"):
            par.update({"domain_radius": self._domain_radius})
        return par

    def create_mesh(self):
        "Purely virtual method."
        pass

    def set_mesh_and_markers_from_file(self, file_name, domain_radius):
        """Set mesh and markers from xdmf file.

        Args:
            file_name (str): Absolute path to file.
        """
        self._mesh, self._cell_marker, self._facet_marker = read_mesh_and_markers(file_name)

        # set subdomain tags
        subdomain_tags = np.sort(np.unique(self._cell_marker.array()))
        self._box_subdomain_tag = subdomain_tags[0]
        self._magnet_subdomain_tags = subdomain_tags[1:]  # exclude first index (the box)
        self._magnet_boundary_subdomain_tags = np.sort(np.unique(self._facet_marker.array()))[:-1]  # exclude last index (the box)

        # set differential measures and domain radius
        self.set_differential_measures()
        self._set_padded_radius(domain_radius)

    def _set_padded_radius(self):
        "Purely virtual method."
        pass

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

    def set_reference_mesh(self, reference_mesh, field_name):
        assert isinstance(reference_mesh, dlf.Mesh)
        
        if field_name == "B":
            self._B_reference_mesh = reference_mesh
        elif field_name == "Vm":
            self._Vm_reference_mesh = reference_mesh
        else:
            raise RuntimeError()

    def update_parameters(self, d_angle):
        # update the angle
        self._angle += d_angle
        # then update the rest
        self.update_magnets()
        self.rotate_mesh(d_angle, axis=0)

    def update_magnets(self):
        assert hasattr(self, "magnets")
        for k, mag in enumerate(self._magnets):
            x_M = self.x_M + np.array([0., self.R * np.cos(2 * np.pi / self.n * k + self.angle),
                                    self.R * np.sin(2 * np.pi / self.n * k + self.angle)])
            Q = Rotation.from_rotvec((2 * np.pi / self.n * k + self.angle) * \
                np.array([1., 0., 0.])).as_matrix()
            mag.update_parameters(x_M, Q)

    def set_differential_measures(self):
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_facet_marker")
        assert hasattr(self, "_cell_marker")
        # set differential measures
        self._normal_vector = dlf.FacetNormal(self._mesh)
        self._dV = dlf.Measure('dx', domain=self._mesh, subdomain_data=self._cell_marker)
        self._dA = dlf.Measure('dS', domain=self._mesh, subdomain_data=self._facet_marker)

    def rotate_mesh(self, d_angle, axis=0):
        """Rotate mesh by angle.

        Args:
            d_angle (float): Angle increment in rad.
        """
        # rotate mesh around axis 0 (x-axis) through gear midpoint by angle d_angle
        self.mesh.rotate(d_angle * 180 / np.pi, axis, dlf.Point(*self.x_M))

    def translate_mesh(self, vec):
        """Translate mesh.

        Args:
            vec (list): Translation vector.
        """
        # rotate mesh around axis 0 (x-axis) through gear midpoint by angle d_angle
        self.mesh.translate(dlf.Point(vec))


class MagneticGearWithBallMagnets(MagneticGear):
    def __init__(self, n, r, R, x_M, magnetization_strength, init_angle, magnet_type="Ball"):
        super().__init__(n, R, x_M, magnetization_strength, init_angle, magnet_type)
        self._r = r  # the magnet radius
        self._create_magnets()
        self._scale_parameter = r

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

    def create_mesh(self, mesh_size_space, mesh_size_magnets, fname, padding=None, write_to_pvd=False, verbose=False):
        """Create ball gear mesh. Set mesh, markers and subdomain tags.

        Args:
            mesh_size_space (float): Global mesh size.
            mesh_size_magnets (float): Magnet mesh size.
            fname (str): File name.
            padding (float or None): Padding added to the mesh exterior. Defaults to None. 
            verbose (bool, optional): If true display gmsh info text. Defaults to False.
        """
        if padding is None:
            padding = self.R / 10
        self._padding = padding
        self._set_padded_radius()
        self._mesh, self._cell_marker, self._facet_marker, self._magnet_subdomain_tags, \
            self._magnet_boundary_subdomain_tags, self._box_subdomain_tag = ball_gear_mesh(
                self, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd, verbose=verbose
                )
        self.set_differential_measures()

    def _set_padded_radius(self, val=None):
        if val is not None:
            self._domain_radius = val
            self._padding = val - self.R - self.r
        else:
            assert hasattr(self, "_padding")
            self._domain_radius = self.R + self.r + self._padding


class MagneticGearWithBarMagnets(MagneticGear):
    def __init__(self, n, h, w, d, R, x_M, magnetization_strength, init_angle, magnet_type="Bar"):
        super().__init__(n, R, x_M, magnetization_strength, init_angle, magnet_type)
        self._h = h  # the magnet height
        self._w = w  # the magnet width
        self._d = d  # the magnet depth
        self._create_magnets()
        self._scale_parameter = h

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

    def create_mesh(self, mesh_size_space, mesh_size_magnets, fname, padding=None, write_to_pvd=False, verbose=False):
        """Create bar gear mesh. Set mesh, markers and subdomain tags.

        Args:
            mesh_size_space (float): Global mesh size.
            mesh_size_magnets (float): Magnet mesh size.
            fname (str): File name.
            padding (float or None): Padding added to the mesh exterior. Defaults to None. 
            verbose (bool, optional): If true display gmsh info text. Defaults to False.
        """
        if padding is None:
            padding = self.R / 10
        self._padding = padding
        self._set_padded_radius()
        self._mesh, self._cell_marker, self._facet_marker, self._magnet_subdomain_tags, \
            self._magnet_boundary_subdomain_tags, self._box_subdomain_tag = bar_gear_mesh(
                self, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd, verbose=verbose
                )
        self.set_differential_measures()

    def _set_padded_radius(self, val=None):
        if val is not None:
                self._domain_radius = val
                self._padding = val - self.R - self.w
        else:
            assert hasattr(self, "_padding")
            self._domain_radius = self.R + self.w + self._padding