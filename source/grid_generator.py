import gmsh
import numpy as np
import dolfin as dlf
from source.mesh_tools import generate_mesh_with_markers


def add_physical_groups(model, box_entity, magnet_entities, magnet_boundary_entities):
    """Add box and magnets as physical groups to gmsh model.
    
    The respective tags will later be used by the facet/cell markers.
    """
    # store the physical group tags, they will later be used to reference subdomains
    magnet_subdomain_tags = []
    magnet_boundary_subdomain_tags = []

    # magnet boundary
    for n, tag in enumerate(magnet_boundary_entities):
        physical_tag = model.addPhysicalGroup(2, np.atleast_1d(tag), name="magnet_" + str(n + 1), tag=int("1%.2d" % (n + 1)))
        magnet_boundary_subdomain_tags.append(physical_tag)

    # box volume
    box_subdomain_tag = model.addPhysicalGroup(3, [box_entity], name="box", tag=1)

    # magnet volume
    for n, tag in enumerate(magnet_entities):
        physical_tag = model.addPhysicalGroup(3, [tag], name="magnet_" + str(n + 1), tag=int("3%.2d" % (n + 1)))
        magnet_subdomain_tags.append(physical_tag)

    return magnet_subdomain_tags, magnet_boundary_subdomain_tags, box_subdomain_tag

def set_mesh_size_fields_gmsh(model, magnet_entities, mesh_size_space, mesh_size_magnets):        
    # global mesh size
    model.mesh.setSize(model.occ.getEntities(0), mesh_size_space)

    # mesh size field for magnets
    mag_field_tag = model.mesh.field.add("Constant")
    model.mesh.field.setNumber(mag_field_tag, "VIn", mesh_size_magnets)
    model.mesh.field.setNumbers(mag_field_tag, "VolumesList", magnet_entities)

    # use the minimum of all the fields as the mesh size field
    min_tag = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(min_tag, "FieldsList", [mag_field_tag])
    model.mesh.field.setAsBackgroundMesh(min_tag)

##################################################
################## ball magnets ##################
##################################################

def ball_gear_mesh(gear, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd=False, verbose=False):
    print("Meshing gear... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    box_entity = add_box_ball_gear(model, gear, padding)

    magnet_entities = []
    for magnet in gear.magnets:
        magnet_tag = add_ball_magnet(model, magnet)
        model.occ.cut([(3, box_entity)], [(3, magnet_tag)], removeObject=True, removeTool=False)
        magnet_entities.append(magnet_tag)
    model.occ.synchronize()

    # get boundary entities
    magnet_boundary_entities = [model.getBoundary([(3, magnet_tag)], oriented=False)[0][1] \
        for magnet_tag in magnet_entities]

    # create mesh and markers
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

def add_ball_magnet(model, magnet):
    magnet_tag = model.occ.addSphere(*magnet.x_M, magnet.R)
    return magnet_tag

def add_box_ball_gear(model, gear, padding):
    """Create the surrounding cylindrical box."""
    # create the cylindrical box 
    A = gear.x_M - (gear.r + padding) * gear.axis
    diff = 2. * (gear.r + padding) * gear.axis
    box_entity = model.occ.addCylinder(*A, *diff, gear.R + gear.r + padding, tag=1)
    model.occ.synchronize()
    return box_entity

##################################################
################## bar magnets ###################
##################################################

def bar_gear_mesh(gear, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd=False, verbose=False):
    print("Meshing gear... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    box_entity = add_box_bar_gear(model, gear, padding)

    magnet_entities = []
    for magnet in gear.magnets:
        magnet_tag = add_bar_magnet(model, gear.axis, magnet)
        model.occ.cut([(3, box_entity)], [(3, magnet_tag)], removeObject=True, removeTool=False)
        magnet_entities.append(magnet_tag)
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
    gmsh.write(fname + '.msh')

    # create mesh and markers
    mesh, cell_marker, facet_marker = generate_mesh_with_markers(fname.rstrip("/"), delete_source_files=False)
    if write_to_pvd:
        dlf.File(fname.rstrip("/") + "_mesh.pvd") << mesh
        dlf.File(fname.rstrip("/") + "_cell_marker.pvd") << cell_marker
        dlf.File(fname.rstrip("/") + "_facet_marker.pvd") << facet_marker

    gmsh.finalize()
    print("Done.")
    return mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags, box_subdomain_tag

def add_bar_magnet(model, axis, magnet):
    # add magnet corner points
    p1 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.d, magnet.w, magnet.h]))))
    p2 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.d, - magnet.w, magnet.h]))))
    p3 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.d, - magnet.w, - magnet.h]))))
    p4 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.d, magnet.w, - magnet.h]))))

    # combine points with lines
    l1 = model.occ.addLine(p1, p2)
    l2 = model.occ.addLine(p2, p3)
    l3 = model.occ.addLine(p3, p4)
    l4 = model.occ.addLine(p4, p1)

    # add front surface
    loop = model.occ.addCurveLoop([l1, l2, l3, l4])
    surf = model.occ.addPlaneSurface([loop])
    model.occ.synchronize()

    # extrude front surface to create the bar magnet
    vec = 2 * magnet.d * axis
    dx, dy, dz = vec
    magnet_gmsh = model.occ.extrude([(2, surf)], dx, dy, dz)

    # find entity with dimension 3 (the extruded volume) and save its tag
    index = np.where(np.array(magnet_gmsh)[:, 0] == 3)[0].item()
    magnet_tag = magnet_gmsh[index][1]

    return magnet_tag

def add_box_bar_gear(model, gear, padding):
    """Create the surrounding cylindrical box."""
    # create the cylindrical box 
    A = gear.x_M - (gear.d + padding) * gear.axis
    diff = 2. * (gear.d + padding) * gear.axis
    box_entity = model.occ.addCylinder(*A, *diff, gear.R + gear.w + padding, tag=1)
    model.occ.synchronize()
    return box_entity

##################################################
############### cylinder segment #################
##################################################

def segment_mesh(x_M, x_M_start, axis, Ri, Ro, t, angle, mesh_size, fname, write_to_pvd=False, verbose=False):
    """Mesh a cylinder segment.

    The occ model is created by first creating an initial surface with 
    center point x_M_start. This initial surface is extruded along curve
    around the rotational axis. The surface is extruded by angle / 2 in
    each direction.

    Args:
        x_M (np.ndarray): Center point.
        x_M_start (np.ndarray): Starting point.
        axis (np.ndarray): Rotational axis.
        Ri (float): Inner radius.
        Ro (float): Outer radius.
        t (float): Thickness (height).
        angle (float): Angle.
        mesh_size (float): Global mesh size.
        fname (str): File name (relative or absolute path).
        write_to_pvd (bool, optional): If True write mesh and markers
                                       to .pvd files. Defaults to False.
        verbose (bool, optional): If True print gmsh info. Defaults to False.

    Returns:
        tuple: mesh, cell_marker, facet_marker
    """
    print("Meshing gear... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()
    axis /= np.linalg.norm(axis)

    # some parameters
    dR = Ro - Ri
    d = x_M_start - x_M
    assert np.isclose(np.dot(d, axis), 0.)
    Rm = np.linalg.norm(x_M - x_M_start)
    d /= Rm

    # add first surface
    # add points
    p1 = model.occ.addPoint(*(x_M_start + dR / 2 * d + t / 2 * axis))
    p2 = model.occ.addPoint(*(x_M_start - dR / 2 * d + t / 2 * axis))
    p3 = model.occ.addPoint(*(x_M_start - dR / 2 * d - t / 2 * axis))
    p4 = model.occ.addPoint(*(x_M_start + dR / 2 * d - t / 2 * axis))

    # combine points with lines
    l1 = model.occ.addLine(p1, p2)
    l2 = model.occ.addLine(p2, p3)
    l3 = model.occ.addLine(p3, p4)
    l4 = model.occ.addLine(p4, p1)
    
    # add front surface
    loop = model.occ.addCurveLoop([l1, l2, l3, l4])
    surf = model.occ.addPlaneSurface([loop])

    # add wire
    curve = model.occ.addCircle(*x_M, r=Rm, angle1=-angle / 2, angle2=angle / 2, zAxis=axis, xAxis=d)
    wire = model.occ.addWire([curve])
    pipe = model.occ.addPipe([(2, surf)], wire)[0][1]
    model.occ.synchronize()

    segment_boundary = model.getBoundary([(3, pipe)], oriented=False)[0][1]

    model.addPhysicalGroup(2, [segment_boundary])
    model.addPhysicalGroup(3, [pipe])

    # global mesh size
    model.mesh.setSize(model.occ.getEntities(0), mesh_size)

    # generate mesh
    model.mesh.generate(dim=3)

    # write mesh to msh file
    gmsh.write(fname + '.msh')

    # create namedtuple
    mesh, cell_marker, facet_marker = generate_mesh_with_markers(fname.rstrip("/"), delete_source_files=False)
    if write_to_pvd:
        dlf.File(fname.rstrip("/") + "_mesh.pvd") << mesh
        dlf.File(fname.rstrip("/") + "_cell_marker.pvd") << cell_marker
        dlf.File(fname.rstrip("/") + "_facet_marker.pvd") << facet_marker

    gmsh.finalize()
    print("Done.")
    return mesh, cell_marker, facet_marker
