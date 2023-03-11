import numpy as np
from source.magnetic_field import free_current_potential_bar_magnet, magnetic_potential_bar_magnet
from source.tools.fenics_tools import CustomScalarExpression, CustomVectorExpression

class PermanentMagnet:
    def __init__(self, type_classifier, magnetization_strength, position_vector, rotation_matrix):
        """Base constructor method.
        Args:
            type_classifier (str): Magnet geometry type (e.g. "Ball", "Bar").
            magnetization_strength (float): Magnetization strength.
            position_vector (np.ndarray): Magnet's position vector (not necessarily the center
                                          of mass) in laboratory cartesian coordinates.
            rotation_matrix (np.ndarray): Rotation matrix defining the magnet's orientation.
                Mathematically, we have for the cartesian basis vectors in the laboratory system
                (termed (0)) and in the magnets eigensystem (termed (1)):
                    Q^{(01})_{ij} = e_i^{(0)} \cdot e_j^{(1)}
                The transformation behavior of the cartesian basis vector is then:
                    e_i^{(1)} = Q^{(01})_{ji} e_j^{(0)}, e_i^{(0)} = Q^{(01})_{ij} e_j^{(1)}
                The transformation is equivalent for vector components. A multiplication of vector
                components from the right side can be used for the transformation (1) -> (0).
        """
        self._type = type_classifier
        self._xM = position_vector
        assert np.allclose(rotation_matrix.dot(rotation_matrix.T), np.eye(3))
        self._Q = rotation_matrix
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
    def M0(self):
        return self._M0

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q

    def is_inside(self, x0):
        """
        Check if point x0 is inside the domain.

        Args:
            x0 (np.ndarray): A point in space.
        """
        pass
    
    def on_boundary(self, x0):
        """
        Check if point x0 is on the magnet's boundary.

        Args:
            x0 (np.ndarray): A point in space.
        """
        pass

    def B_eigen(self):
        """Magnetic field in eigen coordinates."""

    def M_eigen(self):
        """Magnetization in eigen coordinates."""

    def Vm_eigen(self):
        """Magnetic potential in eigen coordinates."""

    def B(self, x0):
        """The magnetic field at point x0 in laboratory cartesian coordinate system and reference frame.

        Args:
            x0 (np.ndarray): A point in space.
            dynamic (bool, optional): True if a dynamic problem is considered. Defaults to False.

        Returns:
            np.ndarray: The magnetic field at point x0 in laboratory cartesian coordinates.
        """

        assert hasattr(self, 'B_eigen')

        x_eigen = np.dot(self.Q.T, x0 - self.x_M)
        B_eigen = self.B_eigen(x_eigen)
        return np.dot(self.Q, B_eigen)

    def Vm(self, x0):
        """The magnetic potential at point x0 in laboratory cartesian coordinate system and reference frame.

        Args:
            x0 (np.ndarray): A point in space.

        Returns:
            np.ndarray: The magnetic field at point x0 in laboratory cartesian coordinates.
        """
        assert hasattr(self, 'Vm_eigen')

        x_eigen = np.dot(self.Q.T, x0 - self.x_M)
        return self.Vm_eigen(x_eigen)

    def B_as_expression(self, degree=1):
        """Magnetic field as dolfin expression.

        Args:
            degree (int, optional): Polynomial degree. Defaults to 1.

        Returns:
            CustomVectorExpression: Magnetic field as dolfin expression.
        """
        B = lambda x: self.B(x)
        return CustomVectorExpression(B, degree=degree)

    def Vm_as_expression(self, degree=1):
        """Magnetic potential as dolfin expression.

        Args:
            degree (int, optional): Polynomial degree. Defaults to 1.

        Returns:
            CustomScalarExpression: Magnetic field as dolfin expression.
        """
        Vm = lambda x: self.Vm_eigen(self.Q.T.dot(x - self.x_M))
        return CustomScalarExpression(Vm, degree=degree)

    def M_as_expression(self, degree=1):
        """Magnetic potential as dolfin expression.

        Args:
            degree (int, optional): Polynomial degree. Defaults to 1.

        Returns:
            CustomVectorExpression: Magnetization field.
        """
        M_lambda = lambda x: self.M_eigen(self.Q.T.dot(x - self.x_M))
        return CustomVectorExpression(M_lambda, degree=degree)

    def update_parameters(self, x_M, Q):
        """Update magnet parameters by specifying new x_M and Q.

        Args:
            x_M (np.ndarray): New position vector.
            Q (np.ndarray): New rotation matrix.
        """
        self._Q = Q
        # automatically update magnetization vector
        self._M = self._Q.dot(np.array([0., 0., 1.]))
        self._xM = x_M


class PermanentAxialMagnet(PermanentMagnet):
    def __init__(self, type_classifier, magnetization_strength, position_vector, rotation_matrix):
        super().__init__(type_classifier, magnetization_strength, position_vector, rotation_matrix)
        # for the axial magnet the magnetization is in z-direction
        self._M = self._Q.dot(np.array([0., 0., 1.]))

    @property
    def M(self):
        return self._M

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        self._Q = Q
        # automatically adjust magnetization vector
        self._M = self._Q.dot(np.array([0., 0., 1.]))

    def M_as_expression(self, degree=1):
        def M(x0):
            if self.is_inside(x0) or self.on_boundary(x0):
                return self.M
            else:
                return np.zeros(3)
        return CustomVectorExpression(M, degree=degree)


class BallMagnet(PermanentAxialMagnet):
    def __init__(self, radius, magnetization_strength, position_vector, rotation_matrix):
        super().__init__(type_classifier='Ball',
                         magnetization_strength=magnetization_strength,
                         position_vector=position_vector,
                         rotation_matrix=rotation_matrix
                         )
        self._radius = radius

    @property
    def R(self):
        return self._radius

    def is_inside(self, x0):
        """Check if point x0 is inside the magnet's domain.

        Args:
            x0 (np.ndarray): A point in space.

        Returns:
            bool: True if x0 is inside the domain.
        """
        diff = self.x_M - x0
        return (np.dot(diff, diff) < self.R ** 2)

    def on_boundary(self, x0):
        """Check if point x0 is on the magnet's boundary.

        Args:
            x0 (np.ndarray): A point in space.

        Returns:
            bool: True if x0 is on the boundary.
        """
        diff = self.x_M - x0
        return np.isclose(np.dot(diff, diff), self.R ** 2)

    def Vm_eigen(self, x_eigen, limit_direction=-1):
        """Magnetic potential of spherical magnet.

        A factor of M0 is excluded.

        Args:
            x_eigen (_type_): Position vector in eigen cartesian coordinates.
            limit_direction (int, optional): Limit_direction in {-1, +1} where -1
                                             corresponds to limit from the inside
                                             and +1 limit from the outside.
                                             Defaults to -1.

        Returns:
            float: The magnetic potential at point x_eigen.
        """
        inside = x_eigen.dot(x_eigen) < self.R ** 2
        on_boundary = abs(x_eigen.dot(x_eigen) - self.R ** 2) < 1e-5

        r = np.linalg.norm(x_eigen)
        r_tilde = r / self.R

        if inside or (on_boundary and limit_direction == -1):
            return self.R / 3. * x_eigen[2]
        else:
            cos_theta = x_eigen[2] / r
            return self.R / 3. / r_tilde ** 2 * cos_theta

    def B_eigen_plus(self, x_eigen):
        """External magnetic field in eigen coordinates.

        A factor of mu_0 * M0 is excluded.

        Args:
            x_eigen (np.ndarray): A point in eigen coordinates.

        Returns:
            np.ndarray: The magnetic field's value at x_eigen.
        """
        r = np.linalg.norm(x_eigen)
        x, y, z = x_eigen
        return 2. / 3. * self.R ** 3 / r ** 5 * np.array([3. / 2. * x * z,
                                                          3. / 2. * y * z,
                                                          - 1. / 2. * (x ** 2 + y ** 2) + z ** 2
                                                          ])

    def B_eigen_minus(self, x_eigen):
        """Internal magnetic field in eigen coordinates.

        A factor of mu_0 * M0 is excluded.

        Args:
            x_eigen (np.ndarray): A point in eigen coordinates.

        Returns:
            np.ndarray: The magnetic field's value at x_eigen.
        """
        return np.array([0., 0., 2. / 3.])

    def B_eigen(self, x_eigen, limit_direction=-1):
        """Magnetic field in eigen coordinates.

        A factor of mu_0 * M0 is excluded.

        Args:
            x_eigen (np.ndarray): A point in eigen coordinates.
            limit_direction (int, optional): Limit_direction in {-1, +1} where -1
                                             corresponds to limit from the inside
                                             and +1 limit from the outside.
                                             Defaults to -1.

        Returns:
            np.ndarray: The magnetic field's value at x_eigen.
        """
        inside = (x_eigen.dot(x_eigen) < self.R ** 2)
        on_boundary = np.isclose(x_eigen.dot(x_eigen), self.R ** 2)
        if inside or (on_boundary and limit_direction == -1):
            return np.array([0., 0., 2. / 3.])
        else:
            r = np.linalg.norm(x_eigen)
            x, y, z = x_eigen
            return 2. / 3. * self.R ** 3 / r ** 5 * np.array([3. / 2. * x * z,
                                                              3. / 2. * y * z,
                                                              - 1. / 2. * (x ** 2 + y ** 2) + z ** 2
                                                              ])


class BarMagnet(PermanentAxialMagnet):
    def __init__(self, width, depth, height, magnetization_strength, position_vector, rotation_matrix):
        super().__init__(type_classifier='Bar',
                         position_vector=position_vector,
                         magnetization_strength=magnetization_strength,
                         rotation_matrix=rotation_matrix
                         )
        # note: input values for height, width and depth only half the actual values
        self._width = width  # x-direction
        self._depth = depth  # y-direction
        self._height = height  # z-direction
        self._set_Vm_eigen()
        self._set_H_eigen()

    @property
    def w(self):
        return self._width

    @property
    def d(self):
        return self._depth

    @property
    def h(self):
        return self._height

    def is_inside(self, x0):
        """Check if point x0 is inside the magnet's domain.

        Args:
            x0 (np.ndarray): A point in space.

        Returns:
            bool: True if x0 is inside the domain.
        """
        x_eigen = self.Q.T.dot(x0 - self.x_M)
        return np.all(np.absolute(x_eigen) < np.array([self.w, self.d, self.h]))

    def on_boundary(self, x0):
        """Check if point x0 is on the magnet's boundary.

        Args:
            x0 (np.ndarray): A point in space.

        Returns:
            bool: True if x0 is on the boundary.
        """
        x_eigen = self.Q.T.dot(x0 - self.x_M)
        return np.any(np.isclose(np.absolute(x_eigen), np.array([self.w, self.d, self.h])))

    def B_eigen_plus(self, x_eigen):
        """External magnetic field in eigen coordinates.
        
        A factor of mu_0 * M0 is excluded.

        Args:
            x_eigen (np.ndarray): A point in eigen coordinates.

        Returns:
            np.ndarray: The magnetic field's value at x_eigen.
        """
        return self.H_eigen(x_eigen)

    def B_eigen_minus(self, x_eigen):
        """Internal magnetic field in eigen coordinates.

        A factor of mu_0 * M0 is excluded.

        Args:
            x_eigen (np.ndarray): A point in eigen coordinates.

        Returns:
            np.ndarray: The magnetic field's value at x_eigen.
        """
        return self.H_eigen(x_eigen) + np.array([0., 0., 1.])

    def B_eigen(self, x_eigen, limit_direction=-1):
        """Magnetic field in eigen coordinates.
        
        A factor of mu_0 * M0 is excluded.

        Args:
            x_eigen (np.ndarray): A point in eigen coordinates.
            limit_direction (int, optional): Limit_direction in {-1, +1} where -1
                                             corresponds to limit from the inside
                                             and +1 limit from the outside.
                                             Defaults to -1.

        Returns:
            np.ndarray: The magnetic field's value at x_eigen.
        """
        inside = np.all(np.absolute(x_eigen) < np.array([self.w, self.d, self.h]))
        on_boundary = np.any(np.isclose(np.absolute(x_eigen), np.array([self.w, self.d, self.h])))

        if inside or (on_boundary and (limit_direction == -1)):
            return self.H_eigen(x_eigen) + np.array([0., 0., 1.])
        else:
            return self.H_eigen(x_eigen)

    def H(self, x0):
        """Free current potential.

        A factor of M0 is excluded.

        Args:
            x0 (np.ndarray): A point in space in laboratory coordinates.

        Returns:
            np.ndarray: The free current potential at x0.
        """
        x_eigen = self.Q.T.dot(x0 - self.x_M)
        H_eigen = self.H_eigen(x_eigen)
        return self.Q.dot(H_eigen)

    def _set_H_eigen(self):
        """Set free current potential in eigen coordinates.

        A factor of M0 is excluded.
        """
        # set dimensionless free current potential as a lambda
        # function of tuple of eigen cordinates (x, y, z), e.g.:
        #   H_test = self.H_eigen((1, 2, 3))
        self.H_eigen = free_current_potential_bar_magnet(
            self._width, self._depth, self._height
            )

    def _set_Vm_eigen(self):
        """Set magnetic potential in eigen coordinates.

        A factor of M0 is excluded.
        """
        # set dimensionless magnetic potential as a lambda
        # function of tuple of eigen cordinates (x, y, z), e.g.:
        #   Vm_test = self.Vm_eigen((1, 2, 3))
        self.Vm_eigen = magnetic_potential_bar_magnet(
            self._width, self._depth, self._height, lambdify=True
            )


class CylinderSegment(PermanentMagnet):
    def __init__(self, radius, width, depth, alpha, magnetization_strength, position_vector, rotation_matrix):
        super().__init__(type_classifier='CylinderSegment',
                         position_vector=position_vector,
                         magnetization_strength=magnetization_strength,
                         rotation_matrix=rotation_matrix
                         )
        # For the cylinder segment the position vector is not set as
        # the gear's center of mass. Instead, it is the point at dis-
        # tance Rm with a zero angle (in eigen coordinates). This can
        # be considered the mid point.
        self._Rm = radius  # the mid radius
        self._width = width
        self._depth = depth
        # half the angle of the segment
        self._alpha = alpha

    @property
    def Rm(self):
        return self._Rm

    @property
    def w(self):
        return self._width

    @property
    def d(self):
        return self._depth

    @property
    def alpha(self):
        return self._alpha

    def is_inside(self, x0):
        """Check if point x0 is inside the magnet's domain.

        Args:
            x0 (np.ndarray): A point in space.

        Returns:
            bool: True if x0 is inside the domain.
        """
        x_eigen = self.Q.T.dot(x0 - self.x_M) + np.array([0., self.Rm, 0.])
        return self.is_inside_eigen(x_eigen)

    def is_inside_eigen(self, x_eigen):
        """Check if point x_eigen is inside the magnet's domain.

        Args:
            x_eigen (np.ndarray): A point in cartesian eigen coordinates.

        Returns:
            bool: True if x_eigen is inside the domain.
        """
        x, y, z = x_eigen
        rho = np.sqrt(y ** 2 + z ** 2)
        phi = np.arctan2(z, y)
        if np.abs(phi) >= self.alpha:
            return False
        if rho <= (self.Rm - self.d) or rho >= (self.Rm + self.d):
            return False
        if np.abs(x) >= self.w:
            return False
        return True

    def on_boundary(self, x0):
        """Check if point x0 is on the magnet's boundary.

        Args:
            x0 (np.ndarray): A point in space in laboratory coordinates.

        Returns:
            bool: True if x0 is on the boundary.
        """
        x_eigen = self.Q.T.dot(x0 - self.x_M) + np.array([0., self.Rm, 0.])
        return self.on_boundary_eigen(x_eigen)

    def on_boundary_eigen(self, x_eigen):
        x, y, z = x_eigen
        rho = np.sqrt(y ** 2 + z ** 2)
        phi = np.arctan2(z, y)

        # check if phi is too large
        if np.abs(phi) > self.alpha and not(np.isclose(np.abs(phi), self.alpha)):
            return False
        # check if rho is too small or too large
        if (rho < (self.Rm - self.d) or rho > (self.Rm + self.d)) and \
            not(np.isclose(rho, self.Rm - self.d) or np.isclose(rho, self.Rm + self.d)):
            return False
        # check if z is too small or too large
        if (np.abs(z) > self.w) and not(np.isclose(np.abs(z), self.w)):
            return False
        # now, if a single value corresponds to the boundary value, the point
        # is indeed on the boundary 
        if np.isclose(np.abs(phi), self.alpha) or np.isclose(rho, self.Rm - self.d) or \
            np.isclose(rho, self.Rm + self.d) or np.isclose(np.abs(z), self.w):
            return True
        return False

    def M_eigen(self, x_eigen):
        """Dimensionless magnetization in eigen coordinates. 

        Args:
            x_eigen (np.ndarray): A point in cartesian eigen coordinates.

        Returns:
            np.ndarray: Magnetization vector.
        """
        assert self.is_inside_eigen(x_eigen) or self.on_boundary_eigen(x_eigen)
        _, y, z = x_eigen
        rho = np.sqrt(y ** 2 + z ** 2)
        return np.array([y / rho, z / rho, 0.])

    def M(self, x0):
        """
        Dimensionless magnetization at point x0.

        Args:
            x0 (np.ndarray): Point in space.

        Returns:
            np.ndarray: Magnetization vector in laboratory coordinates.
        """
        if self.is_inside(x0):
            x_eigen = self.Q.T.dot(x0 - self.x_M) + np.array([0., self.Rm, 0.])
            M_eigen = self.M_eigen(x_eigen)

            return self.Q.dot(M_eigen)
        else:
            return np.zeros(3)
