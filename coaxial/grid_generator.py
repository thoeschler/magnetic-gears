import gmsh
import dolfin as dlf
import numpy as np
from scipy.spatial.transform import Rotation
from source.grid_generator import add_ball_magnet, add_bar_magnet, add_box_ball_gear, add_box_bar_gear, add_physical_groups
from source.mesh_tools import generate_mesh_with_markers

def set_mesh_size_fields_coaxial(model, gear, x_M_ref, magnet_entities, mesh_size_space, mesh_size_magnets):
    model.mesh.setSize(model.occ.getEntities(0), mesh_size_space)

    # mesh size field for magnets
    mag_field_tag = model.mesh.field.add("Constant")
    model.mesh.field.setNumber(mag_field_tag, "VIn", mesh_size_magnets)

    close_magnets = []
    D_ref = np.linalg.norm(x_M_ref - gear.x_M)
    rot =  Rotation.from_rotvec((2 * np.pi / gear.n) * np.array([1., 0., 0.])).as_matrix()
    for magnet, tag in zip(gear.magnets, magnet_entities):
        x_M_max = rot.dot(magnet.x_M - gear.x_M) + magnet.x_M
        x_M_min = rot.T.dot(magnet.x_M - gear.x_M) + magnet.x_M
        if (np.linalg.norm(x_M_max - x_M_ref) < D_ref) or (np.linalg.norm(x_M_min - x_M_ref) < D_ref):
            close_magnets.append(tag)
    model.mesh.field.setNumbers(mag_field_tag, "VolumesList", close_magnets)

    # use the minimum of all the fields as the mesh size field
    min_tag = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(min_tag, "FieldsList", [mag_field_tag])
    model.mesh.field.setAsBackgroundMesh(min_tag)

##################################################
################## ball magnets ##################
##################################################

def ball_gear_mesh(gear, x_M_ref, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd=False, \
    verbose=False):
    assert x_M_ref is not None
    assert len(x_M_ref) == 3
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

    # create namedtuple
    magnet_subdomain_tags, magnet_boundary_subdomain_tags, box_subdomain_tag = add_physical_groups(
        model, box_entity, magnet_entities, magnet_boundary_entities
        )

    # set mesh size fields
    set_mesh_size_fields_coaxial(model, gear, x_M_ref, magnet_entities, mesh_size_space, \
        mesh_size_magnets)

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

##################################################
################## bar magnets ##################
##################################################

def bar_gear_mesh(gear, x_M_ref, mesh_size_space, mesh_size_magnets, fname, padding, write_to_pvd=False, verbose=False):
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
    set_mesh_size_fields_coaxial(model, gear, x_M_ref, magnet_entities, mesh_size_space, mesh_size_magnets)

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
    return mesh, cell_marker, facet_marker, magnet_subdomain_tags, magnet_boundary_subdomain_tags, box_subdomain_tag