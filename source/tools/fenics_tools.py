import dolfin as dlf
dlf.set_log_level(dlf.LogLevel.ERROR)

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

def compute_current_potential(Vm: dlf.Function, project=False, cell_type="CG"):
    """Compute free current potential from magnetic potential.

    If the magnetization is zero the free current potential
    is equal to the magnetic field (divided by permeability).

    Args:
        Vm (dlf.Function): Magnetic potential.
        project (bool, optional): Whether to project to a finite element space.
                                  Defaults to False.
        cell_type (str, optional): Cell type used if projection is True.
                                   Defaults to "CG".

    Returns:
        dlf.Function or dlf.ComponentTensor: Free current potential.
    """

    # compute gradient
    H = - dlf.grad(Vm)

    # return gradient if no projection is needed
    if not project:
        return H

    Vm_p_deg = Vm.ufl_element().degree()
    V = dlf.VectorFunctionSpace(Vm.function_space().mesh(), cell_type, Vm_p_deg)

    # compute magnetic field and project to function space
    # Use CG solver for projection. This is due to an UMFPACK error that
    # limits the memory usage to 4GB
    # https://fenicsproject.org/qa/4177/reason-petsc-error-code-is-76/
    H_func = dlf.project(H, V, solver_type="cg")

    return H_func

def rotate_vector_field(f: dlf.Function, Q):
    """Rotate components of vector field.

    Args:
        f (dlf.Function): Vector field.
        Q (np.ndarray): Rotation matrix.
    """
    ndim = f.geometric_dimension()
    # get values
    vals = f.vector().get_local().reshape(-1, ndim).T
    # rotate
    rotated_vals = Q.dot(vals).T
    assert rotated_vals.shape[1] == ndim
    # set rotated values
    f.vector().set_local(rotated_vals.flatten())
