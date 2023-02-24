import gmsh
import dolfin as dlf
from source.tools.mesh_tools import generate_mesh_with_markers, generate_xdmf_mesh, read_mesh


def create_single_magnet_mesh(magnet, mesh_size, verbose=True):
    """Create finite element mesh of a single ball magnet inside a box.

    Args:
        magnet (BallMagnet): The spherical magnet.
        mesh_size (float): The global mesh size.
        verbose (bool, optional): If True print gmsh info.
                                  Defaults to True.

    Returns:
        tuple: A tuple containing mesh, cell_marker, facet_marker,
               magnet_volume_tag, magnet_boundary_tag.
    """
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    # file name
    fname = "mesh"
    model = gmsh.model()

    # create magnet
    magnet_tag = model.occ.addSphere(*magnet.x_M, magnet.R)
    model.occ.synchronize()

    # add physical groups
    magnet_boundary = model.getBoundary([(3, magnet_tag)], oriented=False)[0][1]
    magnet_boundary_tag = model.addPhysicalGroup(2, [magnet_boundary])
    magnet_volume_tag = model.addPhysicalGroup(3, [magnet_tag])

    # set mesh size
    model.mesh.setSize(model.occ.getEntities(0), mesh_size)

    # generate mesh
    model.mesh.generate(dim=3)
    gmsh.write(f"{fname}.msh")

    # create mesh and markers
    generate_xdmf_mesh(f"{fname}.msh", delete_source_files=True)
    mesh = read_mesh(fname + ".xdmf")
    """dlf.File(fname + "_mesh.pvd") << mesh
    dlf.File(fname + "_cell_marker.pvd") << cell_marker
    dlf.File(fname + "_facet_marker.pvd") << facet_marker
    """
    gmsh.finalize()
    return mesh#, cell_marker, facet_marker, magnet_volume_tag, magnet_boundary_tag
