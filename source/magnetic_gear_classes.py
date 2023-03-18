import dolfin as dlf
import numpy as np
from source.magnet_classes import BallMagnet, BarMagnet, CylinderSegment
from source.tools.math_tools import get_rot


class MagneticGear:
    def __init__(self, n, R, x_M):
        """Magnetic Gear.

        Args:
            n (int): Even number of magnets.
            R (float): The gear's radius.
            x_M (np.ndarray): Position vector.
        """
        assert n % 2 == 0  # even number of magnets
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
            self.translate_mesh(d_x_M)
        self._x_M = x_M
        # update magnet coordinates
        if hasattr(self, "_magnets"):
            self.update_magnets(d_angle=0., d_x_M=d_x_M)

    @property
    def angle(self):
        return self._angle

    @property
    def width(self):
        """The gear's width."""

    def reset_angle(self, angle):
        self._angle = angle

    @angle.setter
    def angle(self, angle):
        assert isinstance(angle, (float, int))
        d_angle = angle - self.angle
        # update mesh coordinates
        if hasattr(self, "_mesh"):
            self.rotate_mesh(d_angle)
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

    def reference_field(self, name):
        """Get reference field by name.

        Args:
            name (str): Name of the field. Can be "B" or "Vm".

        Returns:
            dlf.Function: The reference field.
        """
        if name == "B":
            assert hasattr(self, "_B_ref")
            return self._B_ref
        elif name == "Vm":
            assert hasattr(self, "_Vm_ref")
            return self._Vm_ref
        else:
            raise RuntimeError()

    def reference_mesh(self, name):
        """Get reference mesh by name.

        Args:
            name (str): Name of the field. Can be "B" or "Vm".

        Returns:
            dlf.Mesh: The reference mesh.
        """
        if name == "B":
            assert hasattr(self, "_B_ref")
            return self._B_reference_mesh
        elif name == "Vm":
            assert hasattr(self, "_Vm_ref")
            return self._Vm_reference_mesh
        else:
            raise RuntimeError()

    def create_magnets(self):
        """Purely virtual method"""
        pass

    def create_mesh(self):
        "Purely virtual method."
        pass

    def set_mesh_markers_and_tags(self, mesh, cell_marker, facet_marker, magnet_subdomain_tags, \
        magnet_boundary_subdomain_tags):
        """Set mesh, cell- and facet markers as well as the respective tags. 

        Args:
            mesh (dlf.Mesh): Finite element mesh.
            cell_marker (dlf.cpp.mesh.MeshFunctionSizet): Cell marker.
            facet_marker (dlf.cpp.mesh.MeshFunctionSizet): Facet marker.
            magnet_subdomain_tags (list): Magnet volume tags.
            magnet_boundary_subdomain_tags (list): Magnet boundary tags.
        """
        self._mesh = mesh
        self._cell_marker = cell_marker
        self._facet_marker = facet_marker
        self._magnet_subdomain_tags = magnet_subdomain_tags
        self._magnet_boundary_subdomain_tags = magnet_boundary_subdomain_tags
        self.set_differential_measures()

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
        """Update parameters given an angle increment.

        Updates mesh and magnets automatically.

        Args:
            d_angle (float): Angle increment.
        """
        # update the angle
        self._angle += d_angle
        # then update the rest
        if hasattr(self, "_magnets"):
            self.update_magnets(d_angle, np.zeros(3))
        if hasattr(self, "_mesh"):
            self.rotate_mesh(d_angle)

    def update_magnets(self, d_angle, d_x_M):
        """Update magnets according to increment of angle and gear's center of mass.

        Make sure that the center of mass has been updated first.

        Args:
            d_angle (float): Angle incerement.
            d_x_M (np.ndarray): Center of mass increment.
        """
        assert hasattr(self, "magnets")
        rot = get_rot(d_angle)
        for mag in self.magnets:
            # new position vector
            x_M = mag.x_M + d_x_M  # shift by d_x_M
            x_M_rot = self._x_M + rot.dot(x_M - self._x_M)  # rotate around new center
            # new rotation matrix
            Q = rot.dot(mag.Q)
            # update both
            mag.update_parameters(x_M_rot, Q)

    def set_differential_measures(self):
        """Set differential measures. 

        This includes volume and surface measure as well as
        the normal vector.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_facet_marker")
        assert hasattr(self, "_cell_marker")
        # set differential measures
        self._dV = dlf.Measure('dx', domain=self._mesh, subdomain_data=self._cell_marker)
        self._dA = dlf.Measure('ds', domain=self._mesh, subdomain_data=self._facet_marker)
        self._normal_vector = dlf.FacetNormal(self._mesh)

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
        assert hasattr(self, "_mesh")
        # convert rad to deg
        self._mesh.rotate(d_angle * 180 / np.pi, 0, dlf.Point(*self.x_M))

    def scale_mesh(self, mesh):
        """Scale mesh by gear's scaling parameter.

        Args:
            mesh (dlf.Mesh): Finite element mesh.
        """
        mesh.scale(self.scale_parameter)

    def translate_mesh(self, vec):
        """Translate gear's mesh by vec.

        Args:
            vec (list): Translation vector.
        """
        assert hasattr(self, "_mesh")
        self._mesh.translate(dlf.Point(*vec))


class MagneticBallGear(MagneticGear):
    def __init__(self, n, R, r, x_M):
        """Magnetic gear with ball magnets.

        Args:
            n (int): Even number of magnets.
            R (float): Gear's radius.
            r (float): Magnet's radius.
            x_M (np.ndarray): Position vector.
        """
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
    def width(self):
        return 2 * self.r

    @property
    def outer_radius(self):
        return self.R + self.r

    def reference_magnet(self):
        """Get reference ball magnet.

        Returns:
            BallMagnet: The reference magnet.
        """
        return BallMagnet(radius=self.r / self.scale_parameter, magnetization_strength=1.0, \
                          position_vector=np.zeros(3), rotation_matrix=np.eye(3))

    def create_magnets(self, magnetization_strength):
        """Create ball magnets, given some magnetization strength.

        Args:
            magnetization_strength (float): Magnetization strength.
        """
        print("Creating magnets... ", end="")
        self._magnets = []

        # create magnets, align them in counter-clockwise order
        for k in range(self.n):
            if k % 2 == 0:
                # magnetization pointing outward
                Q = get_rot(- np.pi / 2 + 2 * np.pi / self.n * k)
            else:
                # magnetization pointing inward
                Q = get_rot(np.pi / 2 + 2 * np.pi / self.n * k)
            x_M = self.x_M + get_rot(2 * np.pi / self.n * k).dot(np.array([0., self.R, 0.]))
            self._magnets.append(BallMagnet(self.r, magnetization_strength, x_M, Q))
        print("Done.")


class MagneticBarGear(MagneticGear):
    def __init__(self, n, R, w, t, d, x_M):
        """Magnetic gear with bar magnets.

        Args:
            n (int): Even number of magnets.
            R (float): Gear's radius.
            w (float): Gear width (x-direction), corresponds to magnet width w.
            t (float): Gear thickness (radial direction), corresponds to magnet
                       height h.
            d (float): Magnet depth (tangential direction), same in BarMagnet.
            x_M (np.ndarray): Position vector.
        """
        super().__init__(n, R, x_M)
        self._w = w
        self._t = t
        self._d = d
        self._scale_parameter = t  # use thickness for scaling
        self._magnet_type = "Bar"

    @property
    def w(self):
        return self._w

    @property
    def t(self):
        return self._t

    @property
    def d(self):
        return self._d

    @property
    def width(self):
        return self.w

    @property
    def outer_radius(self):
        return np.sqrt((self.R + self.t / 2) ** 2 + self.d ** 2 / 4)

    @property
    def parameters(self):
        par = super().parameters
        par.update({
            "w": self.w,
            "t": self.t,
            "d": self.d
        })
        return par

    def reference_magnet(self):
        """Get reference bar magnet.

        Returns:
            BarMagnet: The reference magnet.
        """
        return BarMagnet(width=self.w / self.scale_parameter, depth=self.d / self.scale_parameter, \
                         height=self.t / self.scale_parameter, magnetization_strength=1.0, \
                            position_vector=np.zeros(3), rotation_matrix=np.eye(3))

    def create_magnets(self, magnetization_strength):
        """Create bar magnets, given some magnetization strength.

        Args:
            magnetization_strength (float): Magnetization strength.
        """
        print("Creating magnets... ", end="")
        self._magnets = []

        # create magnets, align them in counter-clockwise order
        for k in range(self.n):
            if k % 2 == 0:
                # magnetization pointing outward
                Q = get_rot(- np.pi / 2 + 2 * np.pi / self.n * k)
            else:
                # magnetization pointing inward
                Q = get_rot(np.pi / 2 + 2 * np.pi / self.n * k)
            x_M = self.x_M + get_rot(2 * np.pi / self.n * k).dot(np.array([0., self.R, 0.]))
            self._magnets.append(BarMagnet(width=self.w, depth=self.d, height=self.t, \
                                           magnetization_strength=magnetization_strength, \
                                           position_vector=x_M, rotation_matrix=Q
                                           ))
        print("Done.")


class SegmentGear(MagneticGear):
    def __init__(self, n, R, w, t, x_M):
        """Magnetic gear with cylinder segment magnets.

        Args:
            n (int): Even number of magnets.
            R (float): Radius.
            w (float): The magnet width (thickness).
            t (float): Thickness (radial direction), corresponds to magnet
                       height h.
            x_M (np.ndarray): Position vector.
        """
        super().__init__(n, R, x_M)
        
        self._w = w
        self._t = t
        self._alpha = 2 * np.pi / n  # angle per magnet
        self._scale_parameter = self.t  # use thickness for scaling
        self._magnet_type = "CylinderSegment"

    @property
    def w(self):
        return self._w

    @property
    def t(self):
        return self._t

    @property
    def alpha(self):
        return self._alpha

    @property
    def width(self):
        return self._w

    @property
    def outer_radius(self):
        return self.R + self.t / 2

    @property
    def parameters(self):
        par = super().parameters
        par.update({
            "alpha": self.alpha,
            "w": self.w,
            "t": self.t
        })
        return par

    def reference_magnet(self):
        """Get reference magnet.

        Returns:
            CylinderSegment: The reference magnet.
        """
        return CylinderSegment(radius=self.R / self.scale_parameter, width=self.w / self.scale_parameter, \
                             thickness=self.t / self.scale_parameter, alpha=self.alpha, \
                                magnetization_strength=1.0, position_vector=np.zeros(3), \
                                    rotation_matrix=np.eye(3))

    def create_magnets(self, magnetization_strength):
        """Create magnets, given some magnetization strength.

        Args:
            magnetization_strength (float): Magnetization strength.
        """
        print("Creating magnets... ", end="")
        self._magnets = []

        # create magnets, align them in counter-clockwise order
        for k in range(self.n):
            if k % 2 == 0:
                # magnetization pointing outward
                magnetization_direction = "out"
            else:
                # magnetization pointing inward
                magnetization_direction = "in"
            Q = get_rot(2 * np.pi / self.n * k)
            x_M = self.x_M + Q.dot(np.array([0., self.R, 0.]))
            self._magnets.append(CylinderSegment(radius=self.R, width=self.w, thickness=self.t, \
                                                 alpha=self.alpha, magnetization_strength=magnetization_strength, \
                                                    position_vector=x_M, rotation_matrix=Q, \
                                                        magnetization_direction=magnetization_direction))
        print("Done.")
