import gmsh
import dolfin as dlf
import numpy as np
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
        write_to_pvd (bool, optional): If True write pvd files. Defaults to False.

    Returns:
        tuple: Mesh, cell marker and facet marker.
    """
    if pad:
        padding = Ro / 60.
        Ri -= padding
        Ro += padding
        t += 2 * padding
        angle += angle / 60.

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
    # use parallel "HXT" algorithm
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    model.mesh.generate(dim=3)

    # write mesh to msh file
    gmsh.write(fname.rstrip("/") + '.msh')

    # generate mesh and markers
    mesh, cell_marker, facet_marker = generate_mesh_with_markers(fname, delete_source_files=True)
    if write_to_pvd:
        dlf.File(fname.rstrip("/") + "_mesh.pvd") << mesh
        dlf.File(fname.rstrip("/") + "_cell_marker.pvd") << cell_marker
        dlf.File(fname.rstrip("/") + "_facet_marker.pvd") << facet_marker

    gmsh.finalize()
    print("Done.")

    return mesh, cell_marker, facet_marker