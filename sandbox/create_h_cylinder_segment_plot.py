import dolfin as dlf
import numpy as np
from source.magnet_classes import CylinderSegment
from source.tools.mesh_tools import read_mesh
from source.tools.tools import read_hd5f_file
from source.tools.fenics_tools import compute_current_potential
from spur_gears.grid_generator import cylinder_segment_mesh

# read mesh 
mesh = read_mesh("sample_cylinder_segment_gear/data/reference/CylinderSegment_Vm_34_24556/reference_mesh.xdmf")
Vm = read_hd5f_file("sample_cylinder_segment_gear/data/reference/CylinderSegment_Vm_34_24556/Vm.h5", \
                    "Vm", mesh, "CG", 2, vector_valued=False)

# create magnet mesh
t = 1.0
Ro = 10.
w = 3.0
alpha = 2 * np.pi / 6
cylinder_segment = CylinderSegment(Ro - t / 2, w, t, alpha, 1.0, \
                                   np.zeros(3), np.eye(3), 1)
cylinder_mesh, _, _ = cylinder_segment_mesh(Ro - t, Ro, w, - alpha / 2, alpha / 2, \
                                      np.array([0., - (Ro - t / 2), 0.]), x_axis=np.array([0., 1., 0.]), \
                                      fname="cylinder_segment_mesh", mesh_size=0.25, pad=False, \
                                        write_to_pvd=True)

# create function space on cylinder segment
V_cyl = dlf.FunctionSpace(cylinder_mesh, "CG", 2)
Vm_cyl = dlf.Function(V_cyl)

# interpolate Vm to cylinder segment
dlf.LagrangeInterpolator.interpolate(Vm_cyl, Vm)
H_cyl = compute_current_potential(Vm_cyl, project=True, cell_type="DG")

dlf.File("H_cyl.pvd") << H_cyl