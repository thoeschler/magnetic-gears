import gmsh
import numpy as np
import dolfin as dlf
from source.tools.mesh_tools import generate_mesh_with_markers
import source.magnetic_gear_classes as mgc
import source.magnet_classes as mc 


def add_physical_groups(model, magnet_entities, magnet_boundary_entities):
    """
    Add magnets as physical groups to gmsh model.

    The respective tags will later be used by the facet/cell markers.

    Args:
        model (gmsh.model): A gmsh model.
        magnet_entities (list): Magnet tags in gmsh.occ.
        magnet_boundary_entities (list): Magnet boundary tags in gmsh.occ.

    Returns:
        tuple: magnet_subdomain_tags, magnet_boundary_subdomain_tags
    """
    # store the physical group tags, they will later be used to reference subdomains
    magnet_subdomain_tags = []
    magnet_boundary_subdomain_tags = []

    # magnet boundary
    for n, tag in enumerate(magnet_boundary_entities):
        physical_tag = model.addPhysicalGroup(2, np.atleast_1d(tag), name="magnet_" + str(n + 1), tag=int("1%.2d" % (n + 1)))
        magnet_boundary_subdomain_tags.append(physical_tag)

    # magnet volume
    for n, tag in enumerate(magnet_entities):
        physical_tag = model.addPhysicalGroup(3, [tag], name="magnet_" + str(n + 1), tag=int("3%.2d" % (n + 1)))
        magnet_subdomain_tags.append(physical_tag)

    return magnet_subdomain_tags, magnet_boundary_subdomain_tags

def set_mesh_size_fields_gmsh(model, magnet_entities, mesh_size):  
    """
    Set mesh size fields in gmsh model.

    Args:
        model (gmsh.model): Gmsh model.
        magnet_entities (list): List of magnet tags in gmsh.occ.
        mesh_size (float): Mesh size.
    """          
    # mesh size field for magnets
    mag_field_tag = model.mesh.field.add("Constant")
    model.mesh.field.setNumber(mag_field_tag, "VIn", mesh_size)
    model.mesh.field.setNumbers(mag_field_tag, "VolumesList", magnet_entities)

    # use the minimum of all the fields as the mesh size field
    min_tag = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(min_tag, "FieldsList", [mag_field_tag])
    model.mesh.field.setAsBackgroundMesh(min_tag)

def add_ball_magnet(model, magnet):
    """
    Add ball magnet to gmsh model.

    Args:
        model (gmsh.model): Gmsh model.
        magnet (BallMagnet): Ball magnet.

    Returns:
        int: Magnet tag.
    """
    assert isinstance(magnet, mc.BallMagnet)
    magnet_tag = model.occ.addSphere(*magnet.x_M, magnet.R)
    return magnet_tag

def add_bar_magnet(model, magnet):
    """
    Add bar magnet to gmsh model.

    Args:
        model (gmsh.model): Gmsh model.
        magnet (BarMagnet): Bar magnet.

    Returns:
        int: Magnet tag.
    """
    assert isinstance(magnet, mc.BarMagnet)

    # add magnet corner points
    p1 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.w / 2, magnet.d / 2, magnet.h / 2]))))
    p2 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.w / 2, - magnet.d / 2, magnet.h / 2]))))
    p3 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.w / 2, - magnet.d / 2, - magnet.h / 2]))))
    p4 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.w / 2, magnet.d / 2, - magnet.h / 2]))))

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
    vec = np.array([magnet.w, 0., 0.])
    dx, dy, dz = vec
    magnet_gmsh = model.occ.extrude([(2, surf)], dx, dy, dz)

    # find entity with dimension 3 (the extruded volume) and save its tag
    index = np.where(np.array(magnet_gmsh)[:, 0] == 3)[0].item()
    magnet_tag = magnet_gmsh[index][1]

    return magnet_tag


def add_cylinder_segment_magnet(model, magnet):
    """
    Add cylinder segment magnet to gmsh model.

    Args:
        model (gmsh.model): Gmsh model.
        magnet (CylinderSegment): Cylinder segment magnet.

    Returns:
        int: Magnet tag.
    """
    assert isinstance(magnet, mc.CylinderSegment)
    # add first surface
    # add points
    p1 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([magnet.w / 2, magnet.t / 2, 0.]))))
    p2 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([magnet.w / 2, - magnet.t / 2, 0.]))))
    p3 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([-magnet.w / 2, - magnet.t / 2, 0.]))))
    p4 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([-magnet.w / 2, magnet.t / 2, 0.]))))

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
    curve = model.occ.addCircle(*x_M, r=magnet.Rm, angle1=-magnet.alpha / 2, angle2=magnet.alpha / 2, \
                                zAxis=np.array([1., 0., 0.]), xAxis=magnet.Q.dot(np.array([0., 1., 0.])))
    wire = model.occ.addWire([curve])
    magnet_tag = model.occ.addPipe([(2, surf)], wire)[0][1]

    return magnet_tag


#######################################################
################ the mesh function ####################
#######################################################

def gear_mesh(gear, mesh_size, fname, write_to_pvd=False, verbose=False):
    """
    Create mesh of magnetic gear.

    Args:
        gear (MagneticGear): Magnetic gear.
        mesh_size (float): Mesh size.
        fname (str): File name.
        write_to_pvd (bool, optional): If True write mesh to pvd file. Defaults to False.
        verbose (bool, optional): If True output gmsh info. Defaults to False.

    Returns:
        tuple: Mesh, cell marker, facet marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags
    """
    print("Meshing gear... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    # add magnets
    magnet_entities = []
    for magnet in gear.magnets:
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

    # create namedtuple
    magnet_subdomain_tags, magnet_boundary_subdomain_tags = add_physical_groups(
        model, magnet_entities, magnet_boundary_entities
        )

    # set mesh size fields
    set_mesh_size_fields_gmsh(model, magnet_entities, mesh_size)

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
    return mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags
