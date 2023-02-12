import gmsh
import dolfin as dlf
import numpy as np
from scipy.spatial.transform import Rotation
import source.magnetic_gear_classes as mgc
from source.grid_generator import add_ball_magnet, add_bar_magnet, add_magnet_segment, add_box_ball_gear, \
    add_box_bar_gear, add_physical_groups, set_mesh_size_fields_gmsh
from source.tools.mesh_tools import generate_mesh_with_markers


def add_segment(model, Ri, Ro, t, angle, x_M, x_axis):
    t += 1e-3
    Ro += 1e-3
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

def gear_mesh(gear, x_M_ref, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd=False, \
    verbose=False):
    assert x_M_ref is not None
    assert len(x_M_ref) == 3
    print("Meshing gear... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    # add surrounding box
    if isinstance(gear, mgc.MagneticBallGear):
        box_entity = add_box_ball_gear(model, gear, padding)
    elif isinstance(gear, (mgc.MagneticBarGear, mgc.SegmentGear)):
        box_entity = add_box_bar_gear(model, gear, padding)
    else:
        raise RuntimeError()
    
    # add/remove magnets
    magnet_entities = []
    D = np.linalg.norm(x_M_ref - gear.x_M)
    rot =  Rotation.from_rotvec((2 * np.pi / gear.n) * np.array([1., 0., 0.])).as_matrix()
    magnets = list(gear.magnets)  # copy magnet list: list is different, magnets are the same
    removed_magnets = list()
    for magnet in magnets:
        x_M_max = rot.dot(magnet.x_M - gear.x_M) + magnet.x_M
        x_M_min = rot.T.dot(magnet.x_M - gear.x_M) + magnet.x_M
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
                magnet_tag = add_magnet_segment(model, magnet)
            else:
                raise RuntimeError()
            model.occ.cut([(3, box_entity)], [(3, magnet_tag)], removeObject=True, removeTool=False)
            magnet_entities.append(magnet_tag)

    # cut removed magnets
    for mag in removed_magnets:
        Ri = 0
        if isinstance(gear, mgc.MagneticBallGear):
            Ro = gear.R + gear.r + padding
            t = 2 * (gear.r + padding)
        elif isinstance(gear, (mgc.MagneticBarGear, mgc.SegmentGear)):
            Ro = gear.R + gear.w + padding
            t = 2 * (gear.d + padding)
        else:
            raise RuntimeError()

        x_axis = mag.x_M - gear.x_M
        x_axis /= np.linalg.norm(x_axis)
        segment = add_segment(model, Ri, Ro, t, angle=2 * np.pi / gear.n, x_M=gear.x_M, x_axis=x_axis)
        box_entity = model.occ.cut([(3, box_entity)], [(3, segment)], removeObject=True, removeTool=False)[0][0][1]

    model.occ.synchronize()

    # get boundary entities
    magnet_boundary_entities = [model.getBoundary([(3, magnet_tag)], oriented=False)[0][1] \
        for magnet_tag in magnet_entities]

    # create namedtuple
    magnet_subdomain_tags, magnet_boundary_subdomain_tags, box_subdomain_tag = add_physical_groups(
        model, box_entity, magnet_entities, magnet_boundary_entities
        )

    # set mesh size fields
    set_mesh_size_fields_gmsh(model, magnet_entities, mesh_size_space, mesh_size_magnets)

    # generate mesh
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
    return mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags, box_subdomain_tag

def segment_mesh(Ri, Ro, t, angle, x_M_ref, x_axis, fname, padding, mesh_size, write_to_pvd=False, \
    verbose=False):
    Ri -= padding
    Ro += 3 * padding  # make sure rotating the gear inside the segment is possible
    t += 2 * padding
    angle += 2 * padding / Ri
    assert x_M_ref is not None
    assert len(x_M_ref) == 3
    x_axis /= np.linalg.norm(x_axis)
    print("Meshing segment... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    model = gmsh.model()
    segment = add_segment(model, Ri, Ro, t, angle=angle, x_M=x_M_ref, x_axis=x_axis)
    model.occ.synchronize()

    segment_boundary = model.getBoundary([(3, segment)], oriented=False)[0][1]

    # add physical groups
    segment_boundary_subdomain_tag = model.addPhysicalGroup(2, np.atleast_1d(segment_boundary))
    segment_subdomain_tag = model.addPhysicalGroup(3, [segment])

    # set mesh size
    model.mesh.setSize(model.occ.getEntities(0), mesh_size)

    # generate mesh
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

    return mesh, cell_marker, facet_marker