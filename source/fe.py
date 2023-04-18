import dolfin as dlf
import gmsh
import numpy as np
from source.magnet_classes import BallMagnet, BarMagnet, CylinderSegment, PermanentAxialMagnet
from source.grid_generator import add_ball_magnet, add_bar_magnet, add_cylinder_segment_magnet
from source.tools.mesh_tools import generate_mesh_with_markers


def compute_magnetic_potential(magnet, R_domain, R_inf=None, mesh_size_magnet=0.2, mesh_size_domain_min=0.2,
                               mesh_size_domain_max=0.5, mesh_size_space=None, p_deg=2, cylinder_mesh_size_field=True,
                               mesh_size_field_thickness=None, autorefine=True, check_input=True,
                               fname=None, write_to_pvd=False):
    """
    Compute magnetic potential of magnet.

    The computational mesh consist of a magnet surrounded by a mid layer and
    a very large surrounding domain that is used to approximate the Dirichlet
    boundary condition.

    Args:
        magnet (PermanentMagnet): Permanent magnet.
        R_domain (float): Radius of domain.
        R_inf (float, optional): Radius of outer domain to impose condition at "infinity".
                                 Defaults to None. R_inf is then set automatically.
        mesh_size_magnet (float, optional): Mesh size for magnet. Defaults to 0.2.
        mesh_size_domain_min (float, optional): Minimum domain mesh size. Defaults to 0.2.
        mesh_size_domain_max (float, optional): Maximum domain mesh size. Defaults to 0.5.
        mesh_size_space (float, optional): Maximum mesh size outside magnet and domain.
                                           Defaults to None. Mesh size is then set automatically.
        p_deg (int, optional): Polynomial degree. Defaults to 2.
        cylinder_mesh_size_field (bool, optional): If True set a mesh size field in the
                                                   form of a cylinder (a magnetic gear).
                                                   Defaults to True.
        mesh_size_field_thickness (float, optional): Thickness of cylinder mesh size field.
                                                     Defaults to None. Thickness is then set
                                                     automatically.
        check_input (bool, optional): Adjust input automatically. Defaults to True.
        autorefine (bool, optional): Refine magnet automatically. Defaults to True.
        fname (str, optional): File name. Defaults to None. Specify only if write_to_pvd is True.
        write_to_pvd (bool, optional): If True write field to pvd file. Defaults to False.
    """
    if write_to_pvd:
        assert fname is not None

    mesh, cell_marker, facet_marker, mag_tag, mag_boundary_tag, box_tag, box_boundary_tag = \
        magnet_mesh(magnet, R_domain=R_domain, R_inf=R_inf,
                    mesh_size_magnet=mesh_size_magnet, mesh_size_domain_min=mesh_size_domain_min,
                    mesh_size_domain_max=mesh_size_domain_max, mesh_size_space=mesh_size_space,
                    cylinder_mesh_size_field=cylinder_mesh_size_field,
                    mesh_size_field_thickness=mesh_size_field_thickness, autorefine=autorefine,
                    check_input=check_input, fname=fname, write_to_pvd=write_to_pvd)

    # magnetization
    if isinstance(magnet, PermanentAxialMagnet):
        M = dlf.as_vector(magnet.M)
    else:
        VM = dlf.VectorFunctionSpace(mesh, "CG", p_deg)
        M = dlf.Function(VM)
        # Use inner magnetic field expression to make sure
        # magnetization is set correctly near the boundary.
        # The integration that follows is only over the magnet 
        # volume so this is a valid procedure
        assert hasattr(magnet, "M_inner_as_expression")
        dlf.LagrangeInterpolator.interpolate(M, magnet.M_inner_as_expression(degree=p_deg))

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

def magnet_mesh(magnet, R_domain, R_inf, mesh_size_magnet, mesh_size_domain_min, mesh_size_domain_max,
                mesh_size_space, cylinder_mesh_size_field, mesh_size_field_thickness=None, fname=None,
                    autorefine=True, check_input=True, write_to_pvd=False, verbose=False):
    """
    Create magnet mesh for finite element computation.

    Args:
        magnet (PermanentMagnet): Permanent magnet.
        R_domain (float): Radius of domain (mesh will be fine inside the domain).
        R_inf (float or None): Radius of surrounding domain.
        mesh_size_magnet (float): Mesh size of magnet.
        mesh_size_domain_min (float): Minimum domain mesh size.
        mesh_size_domain_max (float): Maximum domain mesh size.
        mesh_size_space (float or None): Mesh size of surrounding space.
        cylinder_mesh_size_field (bool): Whether to apply a mesh size field in the form
                                         of a cylinder.
        mesh_size_field_thickness (float, optional): Thickness of cylinder mesh size field.
                                                     Defaults to None.
        fname (str, optional): File name. Defaults to None.
        check_input (bool, optional): Adjust input automatically. Defaults to True.
        autorefine (bool, optional): Refine magnet automatically. Defaults to True.
        write_to_pvd (bool, optional): If True write field to pvd file. Defaults to False.
        verbose (bool, optional): If True output gmsh info. Defaults to False.

    Returns:
        tuple: Mesh, cell_marker, facet_marker, mag_tag, mag_boundary_tag, box_tag, box_boundary_tag.
    """
    # check input
    if check_input:
        if cylinder_mesh_size_field:
            if mesh_size_field_thickness is None:
                mesh_size_field_thickness = magnet.size + 1e-3

        if R_domain <= magnet.size:
            R_domain = magnet.size + mesh_size_magnet
        if R_inf is None:
            R_inf = 50 * magnet.size
        if R_inf < R_domain:
            if R_domain > 50 * magnet.size:
                R_inf = R_domain + mesh_size_space
            else:
                R_inf = 50 * magnet.size
        
        if 10 * mesh_size_domain_min > magnet.size and autorefine:
            mesh_size_magnet = magnet.size / 10

        if write_to_pvd:
            assert fname is not None
    # always adjust some input
    if mesh_size_space is None:
        mesh_size_space = 8.
    if R_inf < R_domain:
        R_inf = R_domain + 1e-2

    print("Creating mesh for fe computation ...", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    # create box
    box = model.occ.addSphere(*magnet.x_M, R_inf)

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

    # set cylinder mesh size field
    if cylinder_mesh_size_field:
        cylinder_ms_field = model.mesh.field.add("Cylinder")
        model.mesh.field.setNumber(cylinder_ms_field, "Radius", R_domain)
        model.mesh.field.setNumber(cylinder_ms_field, "VIn", mesh_size_domain_min)
        model.mesh.field.setNumber(cylinder_ms_field, "VOut", mesh_size_space)
        model.mesh.field.setNumber(cylinder_ms_field, "XAxis", mesh_size_field_thickness)
        model.mesh.field.setNumber(cylinder_ms_field, "YAxis", 0.)
        model.mesh.field.setNumber(cylinder_ms_field, "ZAxis", 0.)
        model.mesh.field.setNumber(cylinder_ms_field, "XCenter", magnet.x_M[0])
        model.mesh.field.setNumber(cylinder_ms_field, "YCenter", magnet.x_M[1])
        model.mesh.field.setNumber(cylinder_ms_field, "ZCenter", magnet.x_M[2])
 
        ms_fields = [mag_field_tag, cylinder_ms_field]
    else:
        # define threshold field for relevant domain
        ball_field = model.mesh.field.add("Ball")
        model.mesh.field.setNumber(ball_field, "XCenter", magnet.x_M[0])
        model.mesh.field.setNumber(ball_field, "YCenter", magnet.x_M[1])
        model.mesh.field.setNumber(ball_field, "ZCenter", magnet.x_M[2])
        model.mesh.field.setNumber(ball_field, "Radius", R_domain)
        model.mesh.field.setNumber(ball_field, "Thickness", 1 / 2)
        model.mesh.field.setNumber(ball_field, "VIn", mesh_size_domain_min)
        model.mesh.field.setNumber(ball_field, "VOut", mesh_size_space)

        # threshold field for surrounding space
        distance_space = model.mesh.field.add("Distance")
        mid_point = model.occ.addPoint(*magnet.x_M)
        model.mesh.field.setNumbers(distance_space, "PointsList", [mid_point])

        threshold_space = model.mesh.field.add("Threshold")
        model.mesh.field.setNumber(threshold_space, "InField", distance_space)
        model.mesh.field.setNumber(threshold_space, "SizeMin", 10 * mesh_size_domain_max)
        model.mesh.field.setNumber(threshold_space, "SizeMax", 20.)
        model.mesh.field.setNumber(threshold_space, "DistMin", R_domain)
        model.mesh.field.setNumber(threshold_space, "DistMax", R_inf)

        ms_fields = [mag_field_tag, ball_field, threshold_space]

    # use the minimum of all the fields as the mesh size field
    min_tag = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(min_tag, "FieldsList", ms_fields)
    model.mesh.field.setAsBackgroundMesh(min_tag)

    # use parallel "HXT" algorithm
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    model.mesh.generate(dim=3)

    # write mesh to msh file
    gmsh.write(fname.rstrip("/") + '.msh')

    # create namedtuple
    mesh, cell_marker, facet_marker = generate_mesh_with_markers(fname, delete_source_files=True)
    if write_to_pvd:
        dlf.File(fname.rstrip("/") + "_mesh.pvd") << mesh
        dlf.File(fname.rstrip("/") + "_cell_marker.pvd") << cell_marker
        dlf.File(fname.rstrip("/") + "_facet_marker.pvd") << facet_marker

    gmsh.finalize()
    print("Done.")

    return mesh, cell_marker, facet_marker, mag_tag, mag_boundary_tag, box_tag, box_boundary_tag
