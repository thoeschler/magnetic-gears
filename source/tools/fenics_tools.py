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

def compute_magnetic_field(Vm: dlf.Function):
    """Compute magnetic field from magnetic potential.

    Args:
        Vm (dlf.Function): Magnetic potential.
    """
    # create function space
    V = dlf.VectorFunctionSpace(Vm.function_space().mesh(), "CG", 1)
    
    # compute magnetic field and project to function space
    H = dlf.project(- dlf.grad(Vm), V)

    return H