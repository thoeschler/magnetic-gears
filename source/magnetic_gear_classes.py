import dolfin as dlf
import numpy as np
import os
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet, BarMagnet
from source.gear_mesh_generator import GearWithBallMagnetsMeshGenerator, GearWithBarMagnetsMeshGenerator


class MagneticGear:
    def __init__(self, n, R, x_M, magnetization_strength, initial_angle, index, main_dir=None):
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
    def alpha(self):
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

    def _create_magnets(self):
        "Purely virtual method."
        pass

    def generate_mesh(self, mesh_size_space, mesh_size_magnets, fname, write_to_pvd=True, verbose=False):
        assert hasattr(self, "_mesh_generator")

        # generate mesh, unpack namedtuple
        self._mesh, self._cell_marker, self._facet_marker = self._mesh_generator.generate_mesh(mesh_size_space, mesh_size_magnets, fname, \
            write_to_pvd=write_to_pvd, verbose=verbose)

        # set differential measures, unpack namedtuple
        self._normal_vector, self._dV, self._dA = self._mesh_generator.get_differential_measures(
            self._mesh, self._cell_marker, self._facet_marker
        )

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
        self._mesh.rotate(d_angle * 180 / np.pi, 0, dlf.Point(*self.x_M))

    def set_subdomain_tags(self, box_subdomain_tag, magnet_subdomain_tags, magnet_boundary_subdomain_tags):
        self._magnet_subdomain_tags = magnet_subdomain_tags
        self._magnet_boundary_subdomain_tags = magnet_boundary_subdomain_tags
        self._box_subdomain_tag = box_subdomain_tag

class MagneticGearWithBallMagnets(MagneticGear):
    def __init__(self, n, r, R, x_M, magnetization_strength, init_angle, index, main_dir=None):
        super().__init__(n, R, x_M, magnetization_strength, init_angle, index, main_dir)
        self._r = r  # the magnet radius
        self._create_magnets()
        self._mesh_generator = GearWithBallMagnetsMeshGenerator(self, main_dir)

    @property
    def r(self):
        return self._r

    def _reference_magnet(self):
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
                                       self.R * np.cos(2 * np.pi / self.n * k + self.alpha),
                                       self.R * np.sin(2 * np.pi / self.n * k + self.alpha)])
            Q = Rotation.from_rotvec((2 * np.pi / self.n * k + self.alpha) * \
                    np.array([1., 0., 0.])).as_matrix()
            self._magnets.append(BallMagnet(self.r, self.M_0, x_M, Q))
        
        print("Done.")

    def get_padded_radius(self):
        return self.R + self.r + self._mesh_generator._pad


class MagneticGearWithBarMagnets(MagneticGear):
    def __init__(self, n, h, w, d, R, x_M, magnetization_strength, init_angle, index, main_dir=None):
        super().__init__(n, R, x_M, magnetization_strength, init_angle, index, main_dir)
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
    
    def _reference_magnet(self):
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
                                       self.R * np.cos(2 * np.pi / self.n * k + self.alpha),
                                       self.R * np.sin(2 * np.pi / self.n * k + self.alpha)])
            rot = Rotation.from_rotvec((2 * np.pi / self.n * k + self.alpha) * \
                np.array([1., 0., 0.])).as_matrix()
            self._magnets.append(BarMagnet(self.h, self.w, self.d, self.M_0, pos, rot))
        
        print("Done.")

    def get_padded_radius(self):
        return self.R + self.w + self._mesh_generator._pad
