import dolfin as dlf
import gmsh
import numpy as np
from source.tools.mesh_tools import generate_mesh_with_markers
from source.magnet_classes import BallMagnet, BarMagnet, CylinderSegment, PermanentAxialMagnet, PermanentMagnet
from source.grid_generator import add_ball_magnet, add_bar_magnet, add_magnet_segment
from spur_gears.grid_generator import segment_mesh

def magnet_mesh(magnet, R_m, R_box, mesh_size_magnet, mesh_size_mid_layer_min, mesh_size_mid_layer_max, \
                mesh_size_space, fname=None, write_to_pvd=False, verbose=False):
    if write_to_pvd:
        assert fname is not None
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    # create box
    box = model.occ.addSphere(*magnet.x_M, R_box)

    # create magnet
    if isinstance(magnet, BallMagnet):
        mag = add_ball_magnet(model, magnet)
    elif isinstance(magnet, BarMagnet):
        mag = add_bar_magnet(model, magnet)
    elif isinstance(magnet, CylinderSegment):
        mag = add_magnet_segment(model, magnet)
    else:
        raise RuntimeError()

    box = model.occ.cut([(3, box)], [(3, mag)], removeObject=True, removeTool=False)[0][0][1]

    # get boundary tags
    model.occ.synchronize()
    box_boundary = np.array(model.getBoundary([(3, box)], oriented=False))[:, 1]
    mag_boundary = np.array(model.getBoundary([(3, mag)], oriented=False))[:, 1]
    for b in box_boundary:
        if b not in mag_boundary:
            outer_boundary = b

    # add physical groups
    box_tag = model.addPhysicalGroup(3, [box])
    mag_tag = model.addPhysicalGroup(3, [mag])
    box_boundary_tag = model.addPhysicalGroup(2, [outer_boundary])
    mag_boundary_tag = model.addPhysicalGroup(2, mag_boundary)
    model.occ.synchronize()
    
    # mesh size fields
    model.mesh.setSize(model.occ.getEntities(0), mesh_size_space)

    # set magnet mesh size
    mag_field_tag = model.mesh.field.add("Constant")
    model.mesh.field.setNumber(mag_field_tag, "VIn", mesh_size_magnet)
    model.mesh.field.setNumbers(mag_field_tag, "VolumesList", [mag_tag])

    # set medium layer mesh size

    # add distance field
    mid_point = model.occ.addPoint(*magnet.x_M)
    model.occ.synchronize()
    distance_tag = model.mesh.field.add("Distance")
    model.mesh.field.setNumbers(distance_tag, "PointsList", [mid_point])

    # add MathEval field that depends on distance
    math_eval_tag = model.mesh.field.add("MathEval")
    model.mesh.field.setString(math_eval_tag, "F", f"F{distance_tag} / {R_m} * "\
        + f"{mesh_size_mid_layer_max - mesh_size_mid_layer_min} + {mesh_size_mid_layer_min}")

    # use the minimum of all the fields as the mesh size field
    min_tag = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(min_tag, "FieldsList", [mag_field_tag, math_eval_tag])
    model.mesh.field.setAsBackgroundMesh(min_tag)

    model.mesh.generate(dim=3)

    # write mesh to msh file
    gmsh.write(fname.rstrip("/") + '.msh')

    # create namedtuple
    mesh, cell_marker, facet_marker = generate_mesh_with_markers(fname, delete_source_files=False)
    if write_to_pvd:
        dlf.File(fname.rstrip("/") + "_mesh.pvd") << mesh
        dlf.File(fname.rstrip("/") + "_cell_marker.pvd") << cell_marker
        dlf.File(fname.rstrip("/") + "_facet_marker.pvd") << facet_marker

    gmsh.finalize()

    return mesh, cell_marker, facet_marker, mag_tag, mag_boundary_tag, box_tag, box_boundary_tag

def solve_poisson(magnet, R_m, R_box, mesh_size_magnet=0.2, mesh_size_mid_layer_min=2.0, mesh_size_mid_layer_max=2.0, mesh_size_space=3.0, p_deg=2, fname=None, write_to_pvd=False):
    if write_to_pvd:
        assert fname is not None
    mesh, cell_marker, facet_marker, mag_tag, mag_boundary_tag, box_tag, box_boundary_tag = magnet_mesh(
        magnet, R_m=R_m, R_box=R_box, fname="test", mesh_size_magnet=mesh_size_magnet, \
            mesh_size_mid_layer_min=mesh_size_mid_layer_min, mesh_size_mid_layer_max=mesh_size_mid_layer_max, \
                mesh_size_space=mesh_size_space, write_to_pvd=write_to_pvd)

    # magnetization
    if isinstance(magnet, PermanentAxialMagnet):
        V_vec = dlf.VectorFunctionSpace(mesh, "DG", p_deg)
        M = dlf.Function(V_vec)
        assert hasattr(magnet, "M_as_expression")
        M_expr = magnet.M_as_expression()
        dlf.LagrangeInterpolator.interpolate(M, M_expr)
        #M = dlf.as_vector(magnet.M)
    else:
        V_vec = dlf.VectorFunctionSpace(mesh, "DG", p_deg)
        M = dlf.Function(V_vec)
        assert hasattr(magnet, "M_as_expression")
        M_expr = magnet.M_as_expression()
        dlf.LagrangeInterpolator.interpolate(M, M_expr)

    dV = dlf.Measure("dx", domain=mesh, subdomain_data=cell_marker)
    dA = dlf.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # create function space
    V = dlf.FunctionSpace(mesh, "CG", p_deg)

    # boundary conditions
    bc_inf = dlf.DirichletBC(V, dlf.Constant(0.0), facet_marker, box_boundary_tag)

    # weak form
    u = dlf.TrialFunction(V)
    v = dlf.TestFunction(V)
    a = - dlf.inner(dlf.grad(u), dlf.grad(v)) * dV
    n = dlf.FacetNormal(mesh)

    L = dlf.div(M) * v * dV + dlf.dot(n, M) * v * dA #dlf.inner(dlf.Constant(1.0), v) * dA(mag_boundary_tag)

    # solve
    Vm = dlf.Function(V)

    dlf.set_log_level(False)
    problem = dlf.LinearVariationalProblem(a, L, Vm, bcs=bc_inf)
    solver = dlf.LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "cg"
    solver.parameters["preconditioner"] = "ilu"
    solver.solve()

    dlf.File("Vm.pvd") << Vm
    return Vm

def convergence_test(magnet: PermanentMagnet, R_m, box_sizes, mesh_sizes_magnet, mesh_sizes_mid_layer_min, \
                     mesh_sizes_mid_layer_max, mesh_size_space, p_deg):
    for ms_mag, ms_layer_min, ms_layer_max in zip(mesh_sizes_magnet[::-1], 
                                                  mesh_sizes_mid_layer_min[::-1], 
                                                  mesh_sizes_mid_layer_max[::-1]
                                                  ):
        for r in box_sizes:
            Vm_num = solve_poisson(magnet, R_m=R_m, R_box=r, mesh_size_magnet=ms_mag, \
                                   mesh_size_mid_layer_min=ms_layer_min, mesh_size_mid_layer_max=ms_layer_max, \
                                   mesh_size_space=mesh_size_space, p_deg=p_deg
                                   )
            Vm_expr = magnet.Vm_as_expression(degree=Vm_num.ufl_element().degree() + 3)
            print("Compute_error...")
            error = dlf.errornorm(Vm_expr, Vm_num, "L2", mesh=Vm_num.function_space().mesh())
            print(f"error: {error}, r: {r}")
            with open("error.csv", "a+") as f:
                f.write(f"{r} {error} \n")

if __name__ == "__main__":
    #cylinder_magnet = CylinderSegment(5, 1, 1, np.pi / 4, 1.0, np.zeros(3), np.eye(3))
    ball_magnet = BallMagnet(1, 1, np.zeros(3), np.eye(3))
    #bar_magnet = BarMagnet(1, 1, 3, 1.0, np.zeros(3), np.eye(3))

    mesh_sizes_magnet = [0.05]
    mesh_sizes_mid_layer_min = [0.5]
    mesh_sizes_mid_layer_max = [1.0]
    R_m = 5 * ball_magnet.R
    box_sizes = np.geomspace(R_m, 8e1, num=5)

    convergence_test(ball_magnet, R_m, box_sizes, mesh_sizes_magnet, mesh_sizes_mid_layer_min, \
                     mesh_sizes_mid_layer_max, mesh_size_space=8.0, p_deg=2)
