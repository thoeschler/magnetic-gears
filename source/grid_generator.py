import gmsh
import numpy as np
import dolfin as dlf
from source.tools.mesh_tools import generate_mesh_with_markers
import source.magnetic_gear_classes as mgc


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

def add_ball_magnet(model, magnet):
    magnet_tag = model.occ.addSphere(*magnet.x_M, magnet.R)
    return magnet_tag

def add_box_ball_gear(model, gear, padding):
    """Create the surrounding cylindrical box."""
    # create the cylindrical box 
    A = gear.x_M - np.array([(gear.r + padding), 0., 0.])
    diff = np.array([2. * (gear.r + padding), 0., 0.])
    box_entity = model.occ.addCylinder(*A, *diff, gear.R + gear.r + padding, tag=1)
    model.occ.synchronize()
    return box_entity

def add_bar_magnet(model, magnet):
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
    vec = np.array([2 * magnet.d, 0., 0.])
    dx, dy, dz = vec
    magnet_gmsh = model.occ.extrude([(2, surf)], dx, dy, dz)

    # find entity with dimension 3 (the extruded volume) and save its tag
    index = np.where(np.array(magnet_gmsh)[:, 0] == 3)[0].item()
    magnet_tag = magnet_gmsh[index][1]

    return magnet_tag

def add_box_bar_gear(model, gear, padding):
    """Create the surrounding cylindrical box."""
    # create the cylindrical box 
    A = gear.x_M - np.array([(gear.d + padding), 0., 0.])
    diff = np.array([2. * (gear.d + padding), 0., 0.])
    box_entity = model.occ.addCylinder(*A, *diff, gear.R + gear.w + padding, tag=1)
    model.occ.synchronize()
    return box_entity

def add_magnet_segment(model, magnet):
    # add first surface
    # add points
    p1 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([magnet.d, magnet.w, 0.]))))
    p2 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([magnet.d, - magnet.w, 0.]))))
    p3 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([-magnet.d, - magnet.w, 0.]))))
    p4 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([-magnet.d, magnet.w, 0.]))))

    # combine points with lines
    l1 = model.occ.addLine(p1, p2)
    l2 = model.occ.addLine(p2, p3)
    l3 = model.occ.addLine(p3, p4)
    l4 = model.occ.addLine(p4, p1)
    
    # add front surface
    loop = model.occ.addCurveLoop([l1, l2, l3, l4])
    surf = model.occ.addPlaneSurface([loop])

    # add wire
    # center point
    x_M = magnet.x_M - magnet.Q.dot(np.array([0., magnet.Rm, 0.]))
    curve = model.occ.addCircle(*x_M, r=magnet.Rm, angle1=-magnet.alpha, angle2=magnet.alpha, \
                                zAxis=np.array([1., 0., 0.]), xAxis=magnet.Q.dot(np.array([0., 1., 0.])))
    wire = model.occ.addWire([curve])
    magnet_tag = model.occ.addPipe([(2, surf)], wire)[0][1]

    return magnet_tag


#######################################################
################ the mesh function ####################
#######################################################

def gear_mesh(gear, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd=False, verbose=False):
    print("Meshing gear... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    # add box
    if isinstance(gear, mgc.MagneticBallGear):
        box_entity = add_box_ball_gear(model, gear, padding)
    elif isinstance(gear, (mgc.MagneticBarGear, mgc.SegmentGear)):
        box_entity = add_box_bar_gear(model, gear, padding)
    else:
        raise RuntimeError()
    
    # add magnets
    magnet_entities = []
    for magnet in gear.magnets:
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
