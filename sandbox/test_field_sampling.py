import dolfin as dlf
import gmsh
import numpy as np
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet, CustomVectorExpression
from source.mesh_tools import generate_mesh_with_markers
import itertools as it
from time import time


r1 = 1.
r2 = 1.
xM1 = np.zeros(3)

mag_1 = BallMagnet(radius=r1, magnetization_strength=1., position_vector=xM1, rotation_matrix=np.eye(3))
mag_2 = BallMagnet(radius=r2, magnetization_strength=1., position_vector=np.zeros(3), rotation_matrix=np.eye(3))

# create mesh
gmsh.initialize()

model = gmsh.model()
sphere = model.occ.addSphere(*xM1, r1)
model.occ.synchronize()

sphere_bndry = model.getBoundary([(3, sphere)])
sphere_bndry_tag = sphere_bndry[0][1]
sphere_bndry_group = model.addPhysicalGroup(2, [sphere_bndry_tag])
sphere_group = model.addPhysicalGroup(3, [sphere])
model.mesh.generate(3)

fname = "mesh"
gmsh.write(fname + '.msh')

mesh, cell_marker, facet_marker = generate_mesh_with_markers(fname, delete_source_files=True)

Vh = dlf.VectorFunctionSpace(mesh, "DG", 1)

gmsh.finalize()

# magnetic field
B_eigen = mag_2.B_eigen_plus

dV = dlf.Measure('dx', domain=mesh, subdomain_data=cell_marker)
dA = dlf.Measure('dS', domain=mesh, subdomain_data=facet_marker)
n = dlf.FacetNormal(mesh)
M = dlf.as_vector(mag_1.M)
x = dlf.Expression(("x[0]", "x[1]", "x[2]"), degree=1)
x_M = dlf.as_vector(mag_1.x_M)

d_vals = np.linspace(1., 5., 1)
alpha_vals = np.linspace(0., 2 * np.pi, 10, endpoint=False)

F_vals = np.zeros((d_vals.size, alpha_vals.size, 3), dtype=float)
tau_vals = np.zeros((d_vals.size, alpha_vals.size), dtype=float)

# time
start = time()

for i, d in enumerate(d_vals):
    # set center of mass
    xM2 = mag_1.x_M + np.array([0., r1 + d + r2, 0.])
    for j, alpha_rel in enumerate(alpha_vals):
        # set rotation matrix
        Q = Rotation.from_rotvec(alpha_rel * np.array([1., 0., 0.])).as_matrix()

        # compute field
        B_func = lambda x_0: np.dot(Q, B_eigen(np.dot(Q.T, x_0 - xM2)))
        B = CustomVectorExpression(B_func)

        # compute traction
        t = dlf.cross(dlf.cross(n('-'), M), B)

        # compute torque density (x - component) w.r.t center of mass of magnet 1
        m = (x[1] - x_M[1]) * t[2] - (x[2] - x_M[2]) * t[1]

        # resulting force
        F = np.empty(3)
        for k, c in enumerate(t):
            F[k] = dlf.assemble(c * dA)

        # x-component of torque vector 
        tau = dlf.assemble(m * dA)

        print(F, tau)

        F_vals[i, j] = F
        tau_vals[i, j] = tau

stop = time()
print(stop - start)

with open("data.csv", "a+") as f:
    for i, d in enumerate(d_vals):
        for j, alpha in enumerate(alpha_vals):
            f.write(f"{d} {alpha} ")
            for c in F_vals[i, j]:
                f.write(f"{c} ")
            f.write(f"{tau_vals[i, j]}")
            # new line
            f.write("\n")
