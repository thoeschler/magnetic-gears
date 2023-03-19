import dolfin as dlf
import gmsh
import numpy as np
from source.magnet_classes import BallMagnet, BarMagnet, CylinderSegment, PermanentAxialMagnet
from source.grid_generator import add_ball_magnet, add_bar_magnet, add_cylinder_segment_magnet
from source.tools.mesh_tools import generate_mesh_with_markers


def compute_magnetic_potential(magnet, R_domain, R_inf=None, mesh_size_magnet=0.2, mesh_size_mid_layer_min=0.2, \
                               mesh_size_mid_layer_max=0.5, p_deg=2, fname=None, \
                                write_to_pvd=False):
    """
    Compute magnetic potential of magnet.

    The computational mesh consist of a magnet surrounded by a mid layer and
    a very large surrounding domain that is used to approximate the Dirichlet
    boundary condition.

    Args:
        magnet (PermanentMagnet): Permanent magnet.
        R_domain (float): Radius of domain.
        mesh_size_magnet (float, optional): Mesh size for magnet. Defaults to 0.2.
        mesh_size_mid_layer_min (float, optional): Mesh size of of mid layer.
                                                   Defaults to 2.0.
        mesh_size_mid_layer_max (float, optional): Maximum mesh size of mid layer.
                                                   Defaults to 2.0.
        p_deg (int, optional): Polynomial degree. Defaults to 2.
        fname (str, optional): File name. Defaults to None.
        write_to_pvd (bool, optional): If True write field to pvd file. Defaults to False.
    """
    if write_to_pvd:
        assert fname is not None
    
    if R_domain <= magnet.size:
        R_domain = magnet.size + mesh_size_magnet
    if R_inf is None:
        R_inf = R_domain + 3 * mesh_size_magnet
    if R_inf < R_domain:
        R_inf = R_domain + 3 * mesh_size_magnet

    mesh, cell_marker, facet_marker, mag_tag, mag_boundary_tag, box_tag, box_boundary_tag = \
        magnet_mesh(magnet, R_domain=R_domain, R_box=R_inf, \
                    mesh_size_magnet=mesh_size_magnet, mesh_size_mid_layer_min=mesh_size_mid_layer_min, \
                        mesh_size_mid_layer_max=mesh_size_mid_layer_max, mesh_size_space=40.0, \
                            fname=fname, write_to_pvd=write_to_pvd)

    # magnetization
    if isinstance(magnet, PermanentAxialMagnet):
        M = dlf.as_vector(magnet.M)
    else:
        VM = dlf.VectorFunctionSpace(mesh, "CG", p_deg)
        M = dlf.Function(VM)
        assert hasattr(magnet, "M_as_expression")
        dlf.LagrangeInterpolator.interpolate(M, magnet.M_as_expression(degree=p_deg))
        dlf.File("M.pvd") << M

    # volume measure
    dV = dlf.Measure("dx", domain=mesh, subdomain_data=cell_marker)

    # boundary condition
    V = dlf.FunctionSpace(mesh, "CG", p_deg)
    bc_inf = dlf.DirichletBC(V, dlf.Constant(0.0), facet_marker, box_boundary_tag)

    # weak form
    u = dlf.TrialFunction(V)
    v = dlf.TestFunction(V)
    a = dlf.inner(dlf.grad(u), dlf.grad(v)) * dV
    L = dlf.inner(M, dlf.grad(v)) * dV(mag_tag)

    # solve problem
    Vm = dlf.Function(V)

    problem = dlf.LinearVariationalProblem(a, L, Vm, bcs=bc_inf)
    solver = dlf.LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "cg"
    solver.parameters["preconditioner"] = "ilu"
    solver.solve()

    if write_to_pvd:
        dlf.File(f"{fname}.pvd") << Vm

    return Vm

def magnet_mesh(magnet, R_domain, R_box, mesh_size_magnet, mesh_size_mid_layer_min, mesh_size_mid_layer_max, \
                mesh_size_space, fname=None, write_to_pvd=False, verbose=False):
    if write_to_pvd:
        assert fname is not None
    if fname is None:
        fname = "magnet_mesh"
    print("Creating magnet mesh ...", end="")
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
        mag = add_cylinder_segment_magnet(model, magnet)
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
    model.mesh.field.setString(math_eval_tag, "F", f"F{distance_tag} / {R_domain} * "\
        + f"{mesh_size_mid_layer_max - mesh_size_mid_layer_min} + {mesh_size_mid_layer_min}")

    # use the minimum of all the fields as the mesh size field
    min_tag = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(min_tag, "FieldsList", [mag_field_tag, math_eval_tag])
    model.mesh.field.setAsBackgroundMesh(min_tag)

    # use parallel "HXT" algorithm
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
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
    print("Done.")

    return mesh, cell_marker, facet_marker, mag_tag, mag_boundary_tag, box_tag, box_boundary_tag