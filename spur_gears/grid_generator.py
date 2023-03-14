import gmsh
import dolfin as dlf
import numpy as np
from source.tools.math_tools import get_rot
import source.magnetic_gear_classes as mgc
from source.grid_generator import add_ball_magnet, add_bar_magnet, add_cylinder_segment_magnet, \
    add_physical_groups, set_mesh_size_fields_gmsh
from source.tools.mesh_tools import generate_mesh_with_markers


def add_cylinder_segment(model, Ri, Ro, t, angle, x_M, x_axis):
    """
    Add cylinder segment to gmsh model.

    Args:
        model (gmsh.model): A gmsh model.
        Ri (float): Inner radius.
        Ro (float): Outer radius.
        t (float): Segment thickness.
        angle (float): Opening angle.
        x_M (np.ndarray): Reference/mid point. 
        x_axis (np.ndarray): Starting axis for angle.

    Returns:
        int: Segment tag.
    """
    if np.isclose(Ri, 0.):
        Ri = Ro / 30.  # cannot be zero, otherwise gmsh cannot create any elements
    # add first surface
    # add points
    p1 = model.occ.addPoint(*(x_M + Ro * x_axis + np.array([t / 2., 0., 0.])))
    p2 = model.occ.addPoint(*(x_M + Ri * x_axis + np.array([t / 2., 0., 0.])))
    p3 = model.occ.addPoint(*(x_M + Ri * x_axis + np.array([-t / 2., 0., 0.])))
    p4 = model.occ.addPoint(*(x_M + Ro * x_axis + np.array([-t / 2., 0., 0.])))

    # combine points with lines
    l1 = model.occ.addLine(p1, p2)
    l2 = model.occ.addLine(p2, p3)
    l3 = model.occ.addLine(p3, p4)
    l4 = model.occ.addLine(p4, p1)

    # add front surface
    loop = model.occ.addCurveLoop([l1, l2, l3, l4])
    surf = model.occ.addPlaneSurface([loop])

    # add wire
    r = (Ri + Ro) / 2
    curve = model.occ.addCircle(*x_M, r=r, angle1=-angle / 2, angle2=angle / 2, zAxis=np.array([1., 0., 0.]), xAxis=x_axis)
    wire = model.occ.addWire([curve])
    segment = model.occ.addPipe([(2, surf)], wire)[0][1]

    return segment

def gear_mesh(gear, x_M_ref, mesh_size, fname, write_to_pvd=False, verbose=False):
    """
    Create gear mesh.

    Args:
        gear (MagneticGear): Magnetic gear
        x_M_ref (np.ndarray): Reference point for magnet deletion.
        mesh_size (float): Mesh size.
        fname (str): File name.
        write_to_pvd (bool, optional): If True write mesh and markers to pvd file.
                                       Defaults to False.
        verbose (bool, optional): If True output gmsh info. Defaults to False.

    Returns:
        tuple: Mesh, cell marker and facet marker.
    """
    assert x_M_ref is not None
    assert len(x_M_ref) == 3
    print("Meshing gear... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    
    # add/remove magnets
    magnet_entities = []
    D = np.linalg.norm(x_M_ref - gear.x_M)
    # if magnet is not close enough to other gear for rotation by +- pi / n, remove it 
    rot =  get_rot(np.pi / gear.n)
    magnets = list(gear.magnets)  # copy magnet list: list is different, magnets are the same
    removed_magnets = list()
    for magnet in magnets:
        x_M_max = rot.dot(magnet.x_M - gear.x_M) + gear.x_M
        x_M_min = rot.T.dot(magnet.x_M - gear.x_M) + gear.x_M
        # remove magnet if too far away
        if (np.linalg.norm(x_M_max - x_M_ref) > D) and (np.linalg.norm(x_M_min - x_M_ref) > D):
            removed_magnets.append(magnet)
            gear.magnets.remove(magnet)
        else:
            if isinstance(gear, mgc.MagneticBallGear):
                magnet_tag = add_ball_magnet(model, magnet)
            elif isinstance(gear, mgc.MagneticBarGear):
                magnet_tag = add_bar_magnet(model, magnet)
            elif isinstance(gear, mgc.SegmentGear):
                magnet_tag = add_cylinder_segment_magnet(model, magnet)
            else:
                raise RuntimeError()
            magnet_entities.append(magnet_tag)

    model.occ.synchronize()

    # get boundary entities
    magnet_boundary_entities = [model.getBoundary([(3, magnet_tag)], oriented=False)[0][1] \
        for magnet_tag in magnet_entities]

    # add physical groups
    magnet_subdomain_tags, magnet_boundary_subdomain_tags = add_physical_groups(
        model, magnet_entities, magnet_boundary_entities
        )

    # set mesh size fields
    set_mesh_size_fields_gmsh(model, magnet_entities, mesh_size)

    # generate mesh
    model.mesh.generate(dim=3)

    # write mesh to msh file
    gmsh.write(fname.rstrip("/") + '.msh')

    # generate mesh and markers
    mesh, cell_marker, facet_marker = generate_mesh_with_markers(fname, delete_source_files=False)
    if write_to_pvd:
        dlf.File(fname.rstrip("/") + "_mesh.pvd") << mesh
        dlf.File(fname.rstrip("/") + "_cell_marker.pvd") << cell_marker
        dlf.File(fname.rstrip("/") + "_facet_marker.pvd") << facet_marker

    gmsh.finalize()
    print("Done.")
    return mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags

def cylinder_segment_mesh(Ri, Ro, t, angle, x_M_ref, x_axis, fname, mesh_size, pad=True, write_to_pvd=False, \
    verbose=False):
    """
    Create a mesh of a cylinder segment

    Args:
        Ri (float): Inner radius.
        Ro (float): Outer radius
        t (float): Thickness.
        angle (float): Opening angle.
        x_M_ref (np.ndarray): Reference/mid point.
        x_axis (np.ndarray): Starting axis for angle.
        fname (str): File name.
        mesh_size (float): Mesh size.
        pad (bool, optional): If True add some padding. Defaults to True.
        write_to_pvd (bool, optional): _description_. Defaults to False.

    Returns:
        tuple: Mesh, cell marker and facet marker.
    """
    if pad:
        padding = Ro / 40.
        Ri -= padding
        Ro += padding
        t += 2 * padding
        angle += padding
    
    # make sure angle is at most 2 pi
    if angle > 2 * np.pi:
        angle = 2 * np.pi

    assert len(x_M_ref) == 3
    # make sure x_axis is normalized
    x_axis /= np.linalg.norm(x_axis)
    print("Meshing segment... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    segment = add_cylinder_segment(model, Ri, Ro, t, angle=angle, x_M=x_M_ref, x_axis=x_axis)
    model.occ.synchronize()

    segment_boundary = model.getBoundary([(3, segment)], oriented=False)[0][1]

    # add physical groups
    model.addPhysicalGroup(2, np.atleast_1d(segment_boundary))
    model.addPhysicalGroup(3, [segment])

    # set mesh size
    model.mesh.setSize(model.occ.getEntities(0), mesh_size)

    # generate mesh
    model.mesh.generate(dim=3)

    # write mesh to msh file
    gmsh.write(fname.rstrip("/") + '.msh')

    # generate mesh and markers
    mesh, cell_marker, facet_marker = generate_mesh_with_markers(fname, delete_source_files=False)
    if write_to_pvd:
        dlf.File(fname.rstrip("/") + "_mesh.pvd") << mesh
        dlf.File(fname.rstrip("/") + "_cell_marker.pvd") << cell_marker
        dlf.File(fname.rstrip("/") + "_facet_marker.pvd") << facet_marker

    gmsh.finalize()
    print("Done.")

    return mesh, cell_marker, facet_marker