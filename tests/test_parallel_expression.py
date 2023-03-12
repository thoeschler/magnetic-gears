import dolfin as dlf
from dolfin import LagrangeInterpolator

# Initialize mesh and function space
mesh = dlf.UnitSquareMesh(4,4)
V = dlf.FunctionSpace(mesh,'CG', 1)
# Initialize some functions in V
u = dlf.Expression('x[0]',degree=1)
f = dlf.Function(V)
LagrangeInterpolator.interpolate(f, u)

print(f)