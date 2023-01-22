import dolfin as dlf
import numpy as np
from source.magnetic_field import free_current_potential_bar_magnet, magnetic_potential_bar_magnet


class CustomVectorExpression(dlf.UserExpression):
    def __init__(self, f_callable, dim=3, **kwargs):
        self.f = f_callable
        self.dim = dim
        super().__init__(**kwargs)

    def eval(self, values, x):
        val = self.f(x)
        for ind, c_val in enumerate(val):
            values[ind] = c_val

    def value_shape(self):
        return (self.dim, )

class CustomScalarExpression(dlf.UserExpression):
    def __init__(self, f_callable, dim=1, **kwargs):
        self.f = f_callable
        self.dim = dim
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = self.f(x)

    def value_shape(self):
        return tuple()


class PermanentAxialMagnet():
    def __init__(self, type_classifier, magnetization_strength, position_vector, rotation_matrix):
        r"""Base constructor method.
        Args:
            type_classifier (str): the magnet geometry type (e.g. 'Ball' in subclass Ball_Magnet)
            magnetization_strength (float): the magnetization strength
            position_vector (np.ndarray): magnet's center of mass in laboratory cartesian coordinates
            rotation_matrix (np.ndarray): the rotation defines the magnet's orientation
                Mathematically, we have for the cartesian basis vectors in the laboratory system (termed (0))
                and in the magnets eigensystem (termed (1)):
                    Q^{(01})_{ij} = e_i^{(0)} \cdot e_j^{(1)}
                The transformation behavior of the cartesian basis vector is then:
                    e_i^{(1)} = Q^{(01})_{ji} e_j^{(0)}, e_i^{(0)} = Q^{(01})_{ij} e_j^{(1)}
                The transformation is equivalent for vector components.
        """
        self._type = type_classifier
        self._xM = position_vector
        self._Q = rotation_matrix
        self._M = self._Q.dot(np.array([0., 0., 1.]))
        self._M0 = magnetization_strength

    @property
    def type(self):
        return self._type

    @property
    def x_M(self):
        return self._xM

    @x_M.setter
    def x_M(self, xM):
        assert len(xM) == 3
        self._xM = xM

    @property
    def M(self):
        return self._M

    @property
    def M0(self):
        return self._M0
    
    @property
    def Q(self):
        return self._Q
    
    @Q.setter
    def Q(self, Q):
        self._Q = Q
        self._M = self._Q.dot(np.array([0., 0., 1.]))

    def B_eigen(self):
        """Purely virtual method."""

    def Vm_eigen(self):
        """Purely virtual method."""

    def B(self, x_0, dynamic=False):
        """The magnetic field at point x_0 in laboratory cartesian coordinate system and reference frame.

        Args:
            x_0 (np.ndarray): A point in space.
            dynamic (bool, optional): True if a dynamic problem is considered. Defaults to False.

        Raises:
            NotImplementedError: dynamic=True is not yet available

        Returns:
            np.ndarray: The magnetic field at point x_0 in laboratory cartesian coordinates.
        """

        if dynamic:
            raise NotImplementedError()
        else:
            assert hasattr(self, 'B_eigen')

            x_eigen = np.dot(self.Q.T, x_0 - self.x_M)
            B_eigen = self.B_eigen(x_eigen)
            return np.dot(self.Q, B_eigen)

    def Vm(self, x_0):
        """The magnetic potential at point x_0 in laboratory cartesian coordinate system and reference frame.

        Args:
            x_0 (np.ndarray): A point in space.

        Returns:
            np.ndarray: The magnetic field at point x_0 in laboratory cartesian coordinates.
        """
        assert hasattr(self, 'Vm_eigen')

        x_eigen = np.dot(self.Q.T, x_0 - self.x_M)
        return self.Vm_eigen(x_eigen)

    def B_as_expression(self):
        B = lambda x: self.Q.dot(self.B_eigen(self.Q.T.dot(x - self.x_M)))
        return CustomVectorExpression(B)

    def Vm_as_expression(self):
        Vm = lambda x: self.Vm_eigen(self.Q.T.dot(x - self.x_M))
        return CustomScalarExpression(Vm)

    def update_parameters(self, x_M, Q):
        self._Q = Q  # updates M automatically
        self._xM = x_M


class BallMagnet(PermanentAxialMagnet):
    def __init__(self, radius, magnetization_strength, position_vector, rotation_matrix):
        super().__init__(type_classifier='Ball',
                         position_vector=position_vector,
                         magnetization_strength=magnetization_strength,
                         rotation_matrix=rotation_matrix
                         )
        self._radius = radius
    
    @property
    def R(self):
        return self._radius

    def is_inside(self, x0):
        diff = self.x_M - x0
        return (np.dot(diff, diff) < self.R**2)
    
    def on_boundary(self, x0):
        diff = self.x_M - x0
        return np.isclose(np.dot(diff, diff), self.R ** 2)
    
    def Vm_eigen(self, x_eigen, limit_direction=-1):
        """Magnetic potential of spherical magnet.

        A factor of M0 is excluded.

        Args:
            x_eigen (_type_): Position vector in eigen cartesian coordinates.
            limit_direction (int, optional): Limit_direction in {-1, +1} where -1
                                             corresponds to limit from the inside
                                             and +1 limit from the outside. Defaults to -1.

        Returns:
            float: The magnetic potential at point x_eigen.
        """
        inside = x_eigen.dot(x_eigen) < self.R ** 2
        on_boundary = abs(x_eigen.dot(x_eigen) - self.R ** 2) < 1e-5

        r = np.linalg.norm(x_eigen)
        r_tilde = r / self.R

        if inside or (on_boundary and limit_direction == -1):
            return 1. / 3. * x_eigen[2] / self.R
        else:
            cos_theta = x_eigen[2] / r
            return 1. / 3. / r_tilde ** 2 * cos_theta

    def B_eigen_plus(self, x_eigen):
        """External magnetic field in eigen coordinates.
        
        A factor of mu0 * M0 is excluded.
        """
        r = np.linalg.norm(x_eigen) / self.R
        x, y, z = x_eigen
        return 2. / 3. / r ** 5 * np.array([3. / 2. * x * z,
                                            3. / 2. * y * z,
                                            - 1. / 2. * (x ** 2 + y ** 2) + z ** 2
                                            ])

    def B_eigen_minus(self, x_eigen):
        """Internal magnetic field in eigen coordinates.

        A factor of mu0 * M0 is excluded.
        """
        return np.array([0., 0., 2. / 3.])

    def B_eigen(self, x_eigen, limit_direction=-1):
        """Magnetic field in eigen coordinates.

        A factor of mu0 * M0 is excluded.
        """
        inside = (x_eigen.dot(x_eigen) < self.R ** 2)
        on_boundary = np.isclose(x_eigen.dot(x_eigen), self.R ** 2)
        if inside or (on_boundary and limit_direction == -1):
            return np.array([0., 0., 2. / 3.])
        else:
            r = np.linalg.norm(x_eigen) / self.R
            x, y, z = x_eigen
            return 2. / 3. / r ** 5 * np.array([3. / 2. * x * z,
                                                3. / 2. * y * z,
                                                - 1. / 2. * (x ** 2 + y ** 2) + z ** 2
                                                ])


class BarMagnet(PermanentAxialMagnet):
    def __init__(self, height, width, depth, magnetization_strength, position_vector, rotation_matrix):
        super().__init__(type_classifier='Bar',
                         position_vector=position_vector,
                         magnetization_strength=magnetization_strength,
                         rotation_matrix=rotation_matrix
                         )
        # note: input values for height, width and depth only half the actual values
        self._height = height  # z-direction
        self._width = width  # y-direction
        self._depth = depth  # x-direction
        self._set_Vm_eigen()
        self._set_H_eigen()

    @property
    def h(self):
        return self._height

    @property
    def w(self):
        return self._width

    @property
    def d(self):
        return self._depth

    def is_inside(self, x0):
        x_eigen = self.Q.T.dot(x0)
        return np.all(np.absolute(x_eigen) < np.array([self.d, self.w, self.h]))

    def on_boundary(self, x0):
        x_eigen = self.Q.T.dot(x0)
        return np.any(np.isclose(np.absolute(x_eigen), np.array([self.d, self.w, self.h])))

    def B_eigen_plus(self, x_eigen):
        """External magnetic field in eigen coordinates.

        A factor of mu0 * M0 is excluded.
        """
        return self.H_eigen(x_eigen)

    def B_eigen_minus(self, x_eigen):
        """Internal magnetic field in eigen coordinates.

        A factor of mu0 * M0 is excluded.
        """
        return self.H_eigen(x_eigen) + np.array([0., 0., 1.])

    def B_eigen(self, x_eigen, limit_direction=-1):
        """Magnetic field in eigen coordinates.
        
        A factor of mu0 * M0 is excluded.
        """
        inside = np.all(np.absolute(x_eigen) < np.array([self.d, self.w, self.h]))
        on_boundary = np.any(np.isclose(np.absolute(x_eigen), np.array([self.d, self.w, self.h])))

        if inside or (on_boundary and (limit_direction == -1)):
            return self.H_eigen(x_eigen) + np.array([0., 0., 1.])
        else:
            return self.H_eigen(x_eigen)

    def _set_H_eigen(self):
        """Free current potential in eigen coordinates.
        
        A factor of M0 is excluded.
        """
        # set dimensionless free current potential as a lambda
        # function of tuple of eigen cordinates (x, y, z), e.g.:
        #   H_test = self.H_eigen((1, 2, 3))
        self.H_eigen = free_current_potential_bar_magnet(
            self._height, self._width, self._depth, lambdify=True
            )

    def _set_Vm_eigen(self):
        """Magnetic potential in eigen coordinates.

        A factor of M0 is excluded.
        """
        # set dimensionless free current potential as a lambda
        # function of tuple of eigen cordinates (x, y, z), e.g.:
        #   H_test = self.H_eigenfield_dimless((1, 2, 3))
        self.Vm_eigen = magnetic_potential_bar_magnet(
            self._height, self._width, self._depth, lambdify=True
            )
