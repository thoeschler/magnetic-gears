import dolfin as dlf
from dolfin import LagrangeInterpolator
import numpy as np
from source.magnet_classes import BallMagnet

# Initialize mesh and function space
n = 10
mesh = dlf.BoxMesh(dlf.Point(np.zeros(3)), dlf.Point(np.ones(3)), n, n, n)
p_deg = 2
V = dlf.FunctionSpace(mesh, 'CG', 1)

# Initialize some functions in V
f = dlf.Function(V)

# create difficult function to interpolate
ball_magnet = BallMagnet(1., 1., np.zeros(3), np.eye(3))
f.interpolate(ball_magnet.Vm_as_expression(degree=p_deg))
