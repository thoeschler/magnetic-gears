import dolfin as dlf
import numpy as np
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet, BarMagnet, MagnetSegment
from source.tools.mesh_tools import read_mesh_and_markers


class MagneticGear:
    def __init__(self, n, R, x_M):
        self._n = n  # the number of magnets
        self._R = R  # the radius of the gear
        assert len(x_M) == 3
        self._x_M = x_M  # the mid point
        self._angle = 0.  # angle
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
        rot = Rotation.from_rotvec(d_angle * np.array([1., 0., 0.])).as_matrix()
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
        self.mesh.rotate(d_angle * 180 / np.pi, 0, dlf.Point(*self.x_M))

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
    def __init__(self, n, R, r, x_M):
        super().__init__(n, R, x_M)
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

        for k in range(self.n):
            Q = Rotation.from_rotvec(2 * np.pi / self.n * k * np.array([1., 0., 0.])).as_matrix()
            x_M = self.x_M + Q.dot(self.R * np.array([0., 1., 0.]))
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
    def __init__(self, n, R, d, w, h, x_M):
        super().__init__(n, R, x_M)
        self._d = d  # the magnet depth (here: x-direction!)
        self._w = w  # the magnet width (here: y-direction!)
        self._h = h  # the magnet height (here: z-direction!)
        self._scale_parameter = h
        self._magnet_type = "Bar"

    @property
    def d(self):
        return self._d

    @property
    def w(self):
        return self._w

    @property
    def h(self):
        return self._h

    @property
    def outer_radius(self):
        return self.R + self.w

    @property
    def parameters(self):
        par = super().parameters
        par.update({
            "d": self.d,
            "w": self.w,
            "h": self.h
        })
        return par
    
    def reference_magnet(self):
        """Return reference magnet instance.

        The reference magnet has a fixed height of 1.0. Width and depth are
        scaled accordingly.

        Returns:
            BarMagnet: The reference magnet.
        """
        return BarMagnet(width=self.w / self.scale_parameter, depth=self.d / self.scale_parameter, \
                         height=self.h / self.scale_parameter, magnetization_strength=1.0, \
                            position_vector=np.zeros(3), rotation_matrix=np.eye(3))

    def create_magnets(self, magnetization_strength):
        """Create ball magnets, given some magnetization strength.

        Args:
            magnetization_strength (float): Magnetization strength.
        """
        print("Creating magnets... ", end="")
        self._magnets = []

        for k in range(self.n):
            Q = Rotation.from_rotvec(2 * np.pi / self.n * k * np.array([1., 0., 0.])).as_matrix()
            x_M = self.x_M + Q.dot(np.array([0., self.R, 0.]))
            self._magnets.append(BarMagnet(width=self.w, depth=self.d, height=self.h, \
                                           magnetization_strength=magnetization_strength, \
                                           position_vector=x_M, rotation_matrix=Q
                                           ))
        print("Done.")

    def _set_padded_radius(self, val=None):
        if val is not None:
            self._domain_radius = val
            self._padding = val - self.R - self.w
        else:
            assert hasattr(self, "_padding")
            self._domain_radius = self.R + self.w + self._padding


class SegmentGear(MagneticGear):
    def __init__(self, n, R, d, w, x_M):
        super().__init__(n, R, x_M)
        self._d = d  # the magnet depth (here: x-direction!)
        self._w = w  # the magnet width (here: y-direction!)
        self._alpha = np.pi / n
        self._scale_parameter = self.w
        self._magnet_type = "SegmentMagnet"

    @property
    def w(self):
        return self._w

    @property
    def d(self):
        return self._d

    @property
    def alpha(self):
        return self._alpha

    @property
    def outer_radius(self):
        return self.R + self.w

    @property
    def parameters(self):
        par = super().parameters
        par.update({
            "alpha": self.alpha,
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
        return MagnetSegment(radius=self.R / self.scale_parameter, width=self.w / self.scale_parameter, \
                             depth=self.d / self.scale_parameter, magnetization_strength=1.0, \
                                position_vector=np.zeros(3), rotation_matrix=np.eye(3))

    def create_magnets(self, magnetization_strength):
        """Create ball magnets, given some magnetization strength.

        Args:
            magnetization_strength (float): Magnetization strength.
        """
        print("Creating magnets... ", end="")
        self._magnets = []

        for k in range(self.n):
            Q = Rotation.from_rotvec(2 * np.pi / self.n * k * np.array([1., 0., 0.])).as_matrix()
            x_M = self.x_M + Q.dot(np.array([0., self.R, 0.]))
            self._magnets.append(MagnetSegment(self.R, self.w, self.d, self.alpha, magnetization_strength, x_M, Q))
        print("Done.")

    def _set_padded_radius(self, val=None):
        if val is not None:
            self._domain_radius = val
            self._padding = val - self.R - self.w
        else:
            assert hasattr(self, "_padding")
            self._domain_radius = self.R + self.w + self._padding
