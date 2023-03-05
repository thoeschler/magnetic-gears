import dolfin as dlf


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

def compute_magnetic_field(Vm: dlf.Function, p_deg=1):
    """Compute magnetic field from magnetic potential.

    Args:
        Vm (dlf.Function): Magnetic potential.
    """
    # create function space
    Vm_p_deg = Vm.ufl_element().degree()
    if Vm_p_deg > 0:
        p_deg = Vm_p_deg - 1 
    else:
        p_deg = 0
    V = dlf.VectorFunctionSpace(Vm.function_space().mesh(), "DG", p_deg)

    # compute magnetic field and project to function space
    # use "mumps"-direct solver. This is due to an UMFPACK error
    # that limits the memory usage to 4GB
    # https://fenicsproject.org/qa/4177/reason-petsc-error-code-is-76/
    H = dlf.project(- dlf.grad(Vm), V, solver_type="mumps")

    return H