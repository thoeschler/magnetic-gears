import dolfin as dlf
from source.field_interpolator import FieldInterpolator
from source.magnet_classes import BallMagnet, CustomVectorExpression
import numpy as np
import gmsh
from source.mesh_tools import generate_xdmf_mesh
from scipy.spatial.transform import Rotation


# create coaxial gears problem
"""par_ball = {"n1": 12,
        "n2": 16,
        "r1": 1.0,
        "r2": 1.0,
        "R1": 10.0,
        "R2": 14.0,
        "D": 5.0,
        "x_M_1": np.array([0., 0., 0.]),
        "magnetization_strength_1": 1e3,
        "magnetization_strength_2": 1e3,
        "initial_angle_1": np.pi/4.,
        "initial_angle_2": np.pi/4
        }
CoaxialGears = CoaxialGearsWithBallMagnets(**par_ball)
CoaxialGears.create_mesh(mesh_size_space=2.0, mesh_size_magnets=0.5)
"""
# create box mesh
gmsh.initialize()
model = gmsh.model()
A = - 3 * np.ones(3)
B = 3 * np.ones(3)
box = model.occ.addBox(*A, *(B - A))
model.occ.synchronize()

bndry_ids = np.array(model.getBoundary([(3, box)], oriented=False))[:, 1]

model.addPhysicalGroup(2, bndry_ids)
model.addPhysicalGroup(3, [box])

model.mesh.setSize(model.occ.getEntities(0), 0.3)
model.mesh.generate(3)

# write mesh to msh file
gmsh.write("meshes/mesh.msh")

# generate xdmf mesh
generate_xdmf_mesh("meshes/mesh.msh", delete_source_files=True)

mesh_file = dlf.XDMFFile("meshes/mesh.xdmf")
mesh = dlf.Mesh()
mesh_file.read(mesh)

gmsh.finalize()

mesh.translate(dlf.Point(0., 0., 0.1))

# create field interpolator
fs = FieldInterpolator(8, "CG", 1, 0.6, 4.0)

# create reference mesh to interpolate field on
fname_mesh = "reference_mesh"
fname_mesh_xdmf = fname_mesh + ".xdmf"

fs.create_reference_mesh(fname_mesh, verbose=True)
fs.read_reference_mesh(fname_mesh_xdmf)

# create reference magnet
ref_magnet = BallMagnet(1., 1., np.zeros(3), np.eye(3))

# interpolate reference field, read it from hd5f file
fname = "B"
field_file_name = "B.h5"
field_name = "B"

B_interpol = fs.interpolate_reference_field(ref_magnet.B, fname, write_pvd=True)
fs.write_hdf5_file(B_interpol, field_file_name, field_name)
B = fs.read_hd5f_file(field_file_name, field_name, vector_valued=True)

# copy mesh
reference_mesh_copy = dlf.Mesh(fs.mesh)
reference_mesh_copy_copy = dlf.Mesh(fs.mesh)
B_copy = dlf.Function(dlf.VectorFunctionSpace(reference_mesh_copy, "CG", 1), B._cpp_object.vector())

# new function space
V = dlf.VectorFunctionSpace(mesh, "CG", 1)

# interpolate field to new function space
B_1 = dlf.interpolate(B, V)

# second magnet may be translated by 0., 2., 0.
reference_mesh_copy.translate(dlf.Point(0., 2., 0.))
B_2 = dlf.interpolate(B_copy, V)

# sum up magnetic field
B_sum = dlf.Function(V, B_1._cpp_object.vector() + B_2._cpp_object.vector())


dlf.File("results/B_1.pvd") << B_1
dlf.File("results/B_2.pvd") << B_2
dlf.File("results/B_sum.pvd") << B_sum
