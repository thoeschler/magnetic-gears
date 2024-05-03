import meshio
import subprocess
import dolfin as dlf
# https://fenicsproject.org/olddocs/dolfinx/dev/python/demos/gmsh/demo_gmsh.py.html
import os


def generate_xdmf_mesh(filename, delete_source_files=True):
    """
    Script generating two xdmf-files from a geo-file or a msh-file. The two
    xdmf-files contain the mesh and the associated facet markers. Facet markers
    refer to the markers on entities of codimension one.
    The mesh is generated by calling gmsh to a generate an msh-file (only if a
    geo-file is passed) and the two xmdf-files are generated using the meshio package.
    """
    # input check
    assert isinstance(filename, str)
    assert os.path.exists(filename)

    # generate msh file
    msh_file = filename.replace(".geo", ".msh")
    assert msh_file.endswith(".msh")
    if filename.endswith(".geo"):
        subprocess.run(["gmsh", "-v", "0", "-3", filename], stdout=subprocess.DEVNULL, check=True)

    # read msh file
    assert os.path.exists(msh_file)
    mesh = meshio.read(msh_file)

    # determine dimension
    if "triangle" in mesh.cells_dict and "tetra" not in mesh.cells_dict:
        assert "line" in mesh.cell_data_dict["gmsh:physical"]
        dim = 2
        prune_z = True
    elif "triangle" in mesh.cells_dict and "tetra" in mesh.cells_dict:  # pragma: no cover
        assert "triangle" in mesh.cell_data_dict["gmsh:physical"]
        dim = 3
        prune_z = False
    else:  # pragma: no cover
        raise RuntimeError()

    # specify cell types
    if dim == 2:
        facet_type = "line"
        cell_type = "triangle"
    elif dim == 3:  # pragma: no cover
        facet_type = "triangle"
        cell_type = "tetra"

    # extract facet mesh (codimension one)
    facet_mesh = create_meshio_mesh(mesh, facet_type, prune_z=prune_z)
    xdmf_facet_marker_file = msh_file.replace(".msh", "_facet_markers.xdmf")
    meshio.write(xdmf_facet_marker_file, facet_mesh, data_format="XML")

    # extract cell mesh (codimension zero)
    cell_mesh = create_meshio_mesh(mesh, cell_type, prune_z=prune_z)
    xdmf_file = msh_file.replace(".msh", ".xdmf")
    meshio.write(xdmf_file, cell_mesh, data_format="XML")

    if delete_source_files:
        # delete msh file
        subprocess.run(["rm", msh_file], check=True)

def create_meshio_mesh(mesh, cell_type, prune_z=False):
    """
    Create a meshio mesh object from a meshio mesh where only cells of
    `cell_type` are taken into account.
    """
    # input check
    assert isinstance(mesh, meshio.Mesh)
    assert isinstance(cell_type, str)
    assert isinstance(prune_z, bool)
    assert cell_type in ("line", "triangle", "tetra")
    # extract cells
    cells = mesh.get_cells_type(cell_type)
    # extract physical regions
    assert "gmsh:physical" in mesh.cell_data_dict
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    # specify data name
    if "triangle" in mesh.cells_dict and "tetra" not in mesh.cells_dict:
        if cell_type == "triangle":
            data_name = "cell_markers"
        elif cell_type == "line":
            data_name = "facet_markers"
        else:  # pragma: no cover
            raise RuntimeError()
    elif "triangle" in mesh.cells_dict and "tetra" in mesh.cells_dict:
        if cell_type == "tetra":
            data_name = "cell_markers"
        elif cell_type == "triangle":
            data_name = "facet_markers"
        else:  # pragma: no cover
            raise RuntimeError()
    else:  # pragma: no cover
        raise RuntimeError()
    # create mesh object
    if prune_z:
        out_mesh = meshio.Mesh(points=mesh.points[:, :2], cells={cell_type: cells},
                            cell_data={data_name: [cell_data]})
    else:  # pragma: no cover
        out_mesh = meshio.Mesh(points=mesh.points, cells={cell_type: cells},
                            cell_data={data_name: [cell_data]})
    return out_mesh

def generate_mesh_with_markers(file_name, delete_source_files=False):
    # transform msh file to xdmf
    # generates both, the cell mesh as well as the facet mesh
    generate_xdmf_mesh(file_name + '.msh')

    # load xdmf meshes
    mesh_file = dlf.XDMFFile(file_name + '.xdmf')
    facet_file = dlf.XDMFFile(file_name + '_facet_markers.xdmf')
    # and then write (or 'read') it into an empty mesh entity
    mesh = dlf.Mesh()
    mesh_file.read(mesh)

    # Mesh Value Collection Cells
    mvc_vol = dlf.MeshValueCollection("size_t", mesh, 3)
    mesh_file.read(mvc_vol, "cell_markers")
    cell_marker = dlf.cpp.mesh.MeshFunctionSizet(mesh, mvc_vol)

    # Mesh Value Collection Facets
    mvc_surf = dlf.MeshValueCollection("size_t", mesh, 2)
    facet_file.read(mvc_surf, "facet_markers")
    facet_marker = dlf.cpp.mesh.MeshFunctionSizet(mesh, mvc_surf)

    if delete_source_files:
        # delete geo and xdmf files
        subprocess.run(["rm", file_name + '.geo_unrolled'], check=True)
        subprocess.run(["rm", file_name + '.xdmf'], check=True)
        subprocess.run(["rm", file_name + '_facet_markers.xdmf'], check=True)

    return mesh, cell_marker, facet_marker

def read_mesh_and_markers(file_name):
    """Read mesh and markers.

    Args:
        file_name (str): File name.

    Returns:
        tuple: Mesh, cell marker and facet marker.
    """
    file_name = file_name.rstrip(".xdmf")
    # load xdmf meshes
    mesh_file = dlf.XDMFFile(file_name + '.xdmf')
    facet_file = dlf.XDMFFile(file_name + '_facet_markers.xdmf')
    # and then write (or 'read') it into an empty mesh entity
    mesh = dlf.Mesh()
    mesh_file.read(mesh)

    # Mesh Value Collection Cells
    mvc_vol = dlf.MeshValueCollection("size_t", mesh, 3)
    mesh_file.read(mvc_vol, "cell_markers")
    cell_marker = dlf.cpp.mesh.MeshFunctionSizet(mesh, mvc_vol)

    # Mesh Value Collection Facets
    mvc_surf = dlf.MeshValueCollection("size_t", mesh, 2)
    facet_file.read(mvc_surf, "facet_markers")
    facet_marker = dlf.cpp.mesh.MeshFunctionSizet(mesh, mvc_surf)

    return mesh, cell_marker, facet_marker

def read_mesh(fname_xdmf):
    """Read mesh from xdmf file.

    Args:
        fname_xdmf (str): Mesh file name (xmdf).
    
    Returns:
        dlf.Mesh: Finite element mesh.
    """
    assert fname_xdmf.endswith(".xdmf")
    mesh_file = dlf.XDMFFile(fname_xdmf)
    mesh = dlf.Mesh()
    mesh_file.read(mesh)
    return mesh
