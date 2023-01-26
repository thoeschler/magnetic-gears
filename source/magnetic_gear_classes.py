import dolfin as dlf
import numpy as np
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet, BarMagnet
from source.mesh_tools import read_mesh_and_markers
from source.math_tools import get_perpendicular


class MagneticGear:
    def __init__(self, n, R, x_M, axis):
        self._n = n  # the number of magnets
        self._R = R  # the radius of the gear
        assert len(x_M) == 3
        self._x_M = x_M  # the mid point
        self._angle = 0.  # angle
        self._axis = axis / np.linalg.norm(axis)  # rotation axis
        self._scale_parameter = None
        self._magnet_type = None

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
        d_x_M = x_M - self.x_M
        # update mesh coordinates
        if hasattr(self, "_mesh"):
            self.translate_mesh(x_M - self.x_M)
        self._x_M = x_M
        # update magnet coordinates
        if hasattr(self, "_magnets"):
            self.update_magnets(d_angle=0, d_x_M=d_x_M)

    @property
    def angle(self):
        return self._angle

    def reset_angle(self, angle):
        self._angle = angle

    @angle.setter
    def angle(self, angle):
        d_angle = angle - self.angle
        assert isinstance(angle, float)
        # update mesh coordinates
        if hasattr(self, "_mesh"):
            self.rotate_mesh(angle - self.angle)
        self._angle = angle
        if hasattr(self, "_magnets"):
            self.update_magnets(d_angle, np.zeros(3))

    @property
    def axis(self):
        return self._axis

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
    def magnet_type(self):
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
    def parameters(self):
        par = {
            "n": self.n,
            "R": self.R,
            "x_M": tuple(self.x_M),
            "magnet_type": self.magnet_type,
        }
        if hasattr(self, "_domain_radius"):
            par.update({"domain_radius": self._domain_radius})
        return par

    def create_magnets(self):
        """Purely virtual method"""
        pass

    def create_mesh(self):
        "Purely virtual method."
        pass

    def set_mesh_markers_and_tags(self, mesh, cell_marker, facet_marker, magnet_subdomain_tags, \
        magnet_boundary_subdomain_tags, box_subdomain_tag, padding=None):
        """Set mesh, cell- and facet markers as well as the respective tags. 

        Args:
            mesh (dlf.Mesh): Finite element mesh.
            cell_marker (dlf.cpp.mesh.MeshFunctionSizet): Cell marker.
            facet_marker (dlf.cpp.mesh.MeshFunctionSizet): Facet marker.
            magnet_subdomain_tags (list): Magnet volume tags.
            magnet_boundary_subdomain_tags (list): Magnet boundary tags.
            box_subdomain_tag (int): Box tag.
            padding (float, optional): Padding value. Defaults to None.
        """
        self._mesh = mesh
        self._cell_marker = cell_marker
        self._facet_marker = facet_marker
        self._magnet_subdomain_tags = magnet_subdomain_tags
        self._magnet_boundary_subdomain_tags = magnet_boundary_subdomain_tags
        self._box_subdomain_tag = box_subdomain_tag
        if padding is not None:
            self._padding = padding
            self._set_padded_radius()
        self.set_differential_measures()

    def set_mesh_and_markers_from_file(self, file_name, domain_radius):
        """Set mesh and markers from xdmf file.

        Args:
            file_name (str): Absolute path to file.
        """
        self._mesh, self._cell_marker, self._facet_marker = read_mesh_and_markers(file_name)

        # set subdomain tags
        subdomain_tags = np.sort(np.unique(self._cell_marker.array()))
        self._box_subdomain_tag = subdomain_tags[0]  # make sure box is labeled first!
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
        if hasattr(self, "_magnets"):
            self.update_magnets(d_angle, np.zeros(3))
        if hasattr(self, "_mesh"):
            self.rotate_mesh(d_angle)

    def update_magnets(self, d_angle, d_x_M):
        """Update magnets according to increment of angle and gear's center of mass.

        Make sure that the center of mass has been updated first

        Args:
            d_angle (float): Angle incerement.
            d_x_M (np.ndarray): Center of mass increment.
        """
        assert hasattr(self, "magnets")
        rot = Rotation.from_rotvec(d_angle * self.axis).as_matrix()
        for mag in self._magnets:
            x_M = mag.x_M + d_x_M
            x_M = self._x_M + rot.dot(x_M - self._x_M)
            Q = rot.dot(mag.Q)
            mag.update_parameters(x_M, Q)

    def set_differential_measures(self):
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_facet_marker")
        assert hasattr(self, "_cell_marker")
        # set differential measures
        self._normal_vector = dlf.FacetNormal(self._mesh)
        self._dV = dlf.Measure('dx', domain=self._mesh, subdomain_data=self._cell_marker)
        self._dA = dlf.Measure('dS', domain=self._mesh, subdomain_data=self._facet_marker)

    def set_mesh_function(self, mesh_function):
        """Set function used to mesh the gear (may differ depending on the problem).

        Args:
            mesh_function (callable): Function that is used to mesh the gear.
        """
        assert callable(mesh_function)
        self.mesh_gear = mesh_function

    def rotate_mesh(self, d_angle):
        """Rotate mesh by angle.

        Args:
            d_angle (float): Angle increment in rad.
        """
        # rotate mesh around axis 0 (x-axis) through gear midpoint by angle d_angle
        # rotation only possible around x, y, z axes
        Q = Rotation.from_rotvec(d_angle * self.axis).as_matrix()
        self.mesh.coordinates()[:] = Q.dot((self.mesh.coordinates()[:]).T - self.x_M[:, None]).T + self.x_M[None]

    def scale_mesh(self, mesh):
        """Scale mesh by gear's scaling parameter.

        Args:
            mesh (dlf.Mesh): Finite element mesh.
        """
        mesh.scale(self.scale_parameter)

    def translate_mesh(self, vec):
        """Translate mesh.

        Args:
            vec (list): Translation vector.
        """
        # rotate mesh around axis 0 (x-axis) through gear midpoint by angle d_angle
        self.mesh.translate(dlf.Point(vec))


class MagneticBallGear(MagneticGear):
    def __init__(self, n, R, r, x_M, axis):
        super().__init__(n, R, x_M, axis)
        self._r = r  # the magnet radius
        self._scale_parameter = r
        self._magnet_type = "Ball"

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

    @property
    def outer_radius(self):
        return self.R + self.r

    def reference_magnet(self):
        """Return reference magnet instance.

        Returns:
            BallMagnet: The reference magnet.
        """
        return BallMagnet(radius=1.0, magnetization_strength=1.0, position_vector=np.zeros(3),
                          rotation_matrix=np.eye(3))

    def create_magnets(self, magnetization_strength):
        """Create ball magnets, given some magnetization strength.

        Args:
            magnetization_strength (float): Magnetization strength.
        """
        print("Creating magnets... ", end="")
        self._magnets = []
        # create unit vector perpendicular to axis
        unit_vec = get_perpendicular(self.axis, normalized=True)
        x_M_base = self.R * unit_vec
        for k in range(self.n):
            rot = Rotation.from_rotvec(2 * np.pi / self.n * k * self.axis).as_matrix()
            # compute position and rotation matrix
            x_M_rel = rot.dot(x_M_base)
            x_M = self.x_M + x_M_rel
            # rotation:
            # 1. turn magnet until magnetization vector aligns with axis
            d1 = np.cross(np.array([0., 0., 1.]), self.axis)
            d1 /= np.linalg.norm(d1)
            Q1 = Rotation.from_rotvec(
                np.arccos(np.dot(self.axis, np.array([0., 0., 1.]))) * d1
                ).as_matrix()
            # 2. turn magnet to align with relative position vector
            d2 = np.cross(self.axis, x_M_rel)
            d2 /= np.linalg.norm(d2)
            Q2 = Rotation.from_rotvec(
                np.arccos(np.dot(self.axis, x_M_rel / self.R)) * d2
                ).as_matrix()
            # 3. turn by pi/2 around axis
            Q3 = Rotation.from_rotvec(np.pi / 2 * self.axis).as_matrix()
            Q = Q3.dot(Q2).dot(Q1)
            assert np.isclose(x_M_rel.dot(Q.dot(np.array([0., 0., 1.]))), 0.)  # magnetization is tangential
            assert np.isclose(self.axis.dot(Q.dot(np.array([0., 0., 1.]))), 0.)
            self._magnets.append(BallMagnet(self.r, magnetization_strength, x_M, Q))
        print("Done.")

    def _set_padded_radius(self, val=None):
        if val is not None:
            self._domain_radius = val
            self._padding = val - self.R - self.r
        else:
            assert hasattr(self, "_padding")
            self._domain_radius = self.R + self.r + self._padding


class MagneticBarGear(MagneticGear):
    def __init__(self, n, R, h, w, d, x_M, axis):
        super().__init__(n, R, x_M, axis)
        self._h = h  # the magnet height
        self._w = w  # the magnet width
        self._d = d  # the magnet depth
        self._scale_parameter = h
        self._magnet_type = "Bar"

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
    def outer_radius(self):
        return self.R + self.w

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

    def create_magnets(self, magnetization_strength):
        """Create ball magnets, given some magnetization strength.

        Args:
            magnetization_strength (float): Magnetization strength.
        """
        print("Creating magnets... ", end="")
        self._magnets = []
        # create random unit vector different from axis
        unit_vec = get_perpendicular(self.axis, normalized=True)
        x_M_base = self.R * unit_vec
        for k in range(self.n):
            rot = Rotation.from_rotvec(2 * np.pi / self.n * k * self.axis).as_matrix()
            x_M_rel = rot.dot(x_M_base)
            x_M = self.x_M + x_M_rel
            # rotation:
            # 1. turn magnet until magnetization vector aligns with axis
            d1 = np.cross(np.array([0., 0., 1.]), self.axis)
            d1 /= np.linalg.norm(d1)
            Q1 = Rotation.from_rotvec(
                np.arccos(np.dot(self.axis, np.array([0., 0., 1.]))) * d1
                ).as_matrix()
            # 2. turn magnet to align with relative position vector
            d2 = np.cross(self.axis, x_M_rel)
            d2 /= np.linalg.norm(d2)
            Q2 = Rotation.from_rotvec(
                np.arccos(np.dot(self.axis, x_M_rel / self.R)) * d2
                ).as_matrix()
            # 3. turn by pi/2 around axis
            Q3 = Rotation.from_rotvec(np.pi / 2 * self.axis).as_matrix()
            # 4. turn around magnetization direction until x direction is aligned with axis
            x = Q3.dot(Q2).dot(Q1).dot(np.array([1., 0., 0.]))
            d4 = np.cross(x, self.axis)
            d4 /= np.linalg.norm(d4)
            Q4 = Rotation.from_rotvec(np.arccos(np.dot(x, self.axis)) * d4).as_matrix()
            # compute final rotation
            Q = Q4.dot(Q3).dot(Q2).dot(Q1)

            assert np.isclose(x_M_rel.dot(Q.dot(np.array([0., 0., 1.]))), 0.)  # magnetization is tangential
            assert np.isclose(self.axis.dot(Q.dot(np.array([0., 0., 1.]))), 0.)
            self._magnets.append(BarMagnet(self.h, self.w, self.d, magnetization_strength, x_M, Q))
        print("Done.")

    def _set_padded_radius(self, val=None):
        if val is not None:
            self._domain_radius = val
            self._padding = val - self.R - self.w
        else:
            assert hasattr(self, "_padding")
            self._domain_radius = self.R + self.w + self._padding