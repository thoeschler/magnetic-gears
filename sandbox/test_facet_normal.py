import dolfin as dlf
from source.magnet_classes import BallMagnet


mesh = dlf.UnitSquareMesh(10,10)
bmesh = dlf.BoundaryMesh(mesh, "exterior")

f = dlf.Constant(('1','1'))
n = dlf.FacetNormal(mesh)

# construct normal vector by hand
left = dlf.CompiledSubDomain("near(x[0], 0) && on_boundary")
right = dlf.CompiledSubDomain("near(x[0], 1) && on_boundary")
bottom = dlf.CompiledSubDomain("near(x[1], 0) && on_boundary")
top = dlf.CompiledSubDomain("near(x[1], 1) && on_boundary")

sub_domains = dlf.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
right.mark(sub_domains, 1)
left.mark(sub_domains, 2)

dV = dlf.Measure("dx", mesh)
dA = dlf.Measure("dS", mesh)
ds = dlf.Measure("ds", mesh, subdomain_data=sub_domains)

dlf.File("mesh.pvd") << mesh

surf = dlf.assemble(n[0] * ds(1))
print(surf)
