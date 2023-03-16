import gmsh
import dolfin as dlf
import numpy as np
import source.magnet_classes as mc
from source.tools.fenics_tools import CustomScalarExpression, CustomVectorExpression
from source.tools.mesh_tools import generate_xdmf_mesh
from source.grid_generator import add_ball_magnet, add_bar_magnet, add_cylinder_segment_magnet


def create_reference_mesh(reference_magnet, domain_radius, mesh_size_min, mesh_size_max, \
    shape="sphere", thickness=None, fname="reference_mesh", verbose=False):
    """
    Create a mesh of a reference sphere.

    Args:
        reference_magnet (PermanentMagnet): Reference magnet.
        domain_radius (float): Radius of domain.
        mesh_size_min (float): Minimum mesh size.
        mesh_size_max (float): Maximum mesh size.
        shape (str): Shape of reference mesh.
        thickness (float): Thickness of cylinder reference mesh. Only specify for
                           shape="cylinder".
        verbose (bool, optional): Whether to display gmsh info. Defaults to False.
    """
    fname = fname.split(".")[0]
    print("Creating reference mesh... ", end="")
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    model = gmsh.model()

    # add magnet
    if isinstance(reference_magnet, mc.BallMagnet):
        magnet = add_ball_magnet(model, magnet=reference_magnet)
        t_min = 2 * reference_magnet.R
    elif isinstance(reference_magnet, mc.BarMagnet):
        magnet = add_bar_magnet(model, magnet=reference_magnet)
        t_min = 2 * reference_magnet.w
    elif isinstance(reference_magnet, mc.CylinderSegment):
        magnet = add_cylinder_segment_magnet(model, magnet=reference_magnet)
        t_min = 2 * reference_magnet.w
    else:
        raise RuntimeError()

    # create surrounding box sphere
    x_M = reference_magnet.x_M
    if shape.lower() == "sphere":
        box = model.occ.addSphere(*x_M, domain_radius)
    elif shape.lower() == "cylinder":
        if thickness is None:
            t = t_min + 1e-3  # some padding
        else:
            t = max(t_min, thickness) + 1e-3 
        box = model.occ.addCylinder(-t / 2, 0., 0., t, 0., 0., r=domain_radius)

    # cut magnet from surrounding box
    model.occ.cut([(3, box)], [(3, magnet)], removeObject=True, removeTool=False)
    model.occ.synchronize()

    # add physical groups
    bndry_ids = np.array(model.getBoundary([(3, box)]))[:, 1]
    model.addPhysicalGroup(2, bndry_ids)
    model.addPhysicalGroup(3, [box])
    model.addPhysicalGroup(3, [magnet])

    # add distance field
    mid_point = model.occ.addPoint(*reference_magnet.x_M)
    model.occ.synchronize()
    distance_tag = model.mesh.field.add("Distance")
    model.mesh.field.setNumbers(distance_tag, "PointsList", [mid_point])

    # add MathEval field that depends on distance
    math_eval_tag = model.mesh.field.add("MathEval")
    model.mesh.field.setString(math_eval_tag, "F", f"F{distance_tag} / {domain_radius} * "\
        + f"{mesh_size_max - mesh_size_min} + {mesh_size_min}")

    # use the minimum of all the fields as the mesh size field
    min_tag = model.mesh.field.add("Min")
    model.mesh.field.setNumbers(min_tag, "FieldsList", [math_eval_tag])
    model.mesh.field.setAsBackgroundMesh(min_tag)

    # generate mesh
    # use parallel "HXT" algorithm
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    model.mesh.generate(3)

    gmsh.write(f"{fname}.msh")

    # generate xdmf mesh
    generate_xdmf_mesh(f"{fname}.msh", delete_source_files=True)

    gmsh.finalize()
    print("Done.")

def interpolate_field(field, mesh, cell_type, p_deg, fname=None, write_pvd=False, scale=1.0):
    """Interpolate a field on a mesh.

    Args:
        field (callable): The field to interpolate.
        mesh (dlf.Mesh): A finite element mesh.
        cell_type (str): Finite element type.
        p_deg (int): Polynomial degree.
        fname (str or None, optional): Output file name. Defaults to None.
        write_pvd (bool, optional): If true write mesh and markers to paraview (pvd)
                                    file. Defaults to False.
        scale (float, optional): Scaling parameter. Defaults to 1.0.

    Returns:
        dlf.Function: The interpolated field.
    """
    if write_pvd:
        assert isinstance(fname, str)

    # test whether the field is scalar or vector-valued (evaluate at random point (0, 0, 0))
    vector_valued = np.atleast_1d(field(np.zeros(3))).size > 1
    print("Interpolating reference field ...", end="")

    if vector_valued:
        V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)
        field_expr = CustomVectorExpression(field, degree=p_deg)
    else:
        V = dlf.FunctionSpace(mesh, cell_type, p_deg)
        field_expr = CustomScalarExpression(field, dim=1, degree=p_deg)

    field_interpolated = dlf.Function(V)
    dlf.LagrangeInterpolator.interpolate(field_interpolated, field_expr)
    if write_pvd:
        assert fname is not None
        dlf.File(f"{fname}.pvd") << field_interpolated

    field_interpolated.vector()[:] *= scale

    print("Done.")
    return field_interpolated

def write_hdf5_file(field, mesh, fname, field_name):
    """Write field to hdf5 file.

    Args:
        field (dlf.Function): The interpolated field.
        mesh (dlf.Mesh): A finite element mesh.
        fname (str): Output file name.
        field_name (str): Name of the interpolated field.
    """
    print(f"Writing hdf5 file... ", end="")
    f = dlf.HDF5File(mesh.mpi_comm(), fname, "w")
    f.write(field, field_name)
    print("Done")

def read_hd5f_file(fname, field_name, mesh, cell_type, p_deg, vector_valued=False):
    """Read field from hdf5 file.

    Args:
        fname (str): File name (hdf5).
        field_name (str): Name of the field that should be read from the file.
        vector_valued (bool, optional): Set to True if the field that should be
                                        read is vector valued. Defaults to False.

    Returns:
        dlf.Function: The field.
    """
    assert fname.endswith(".h5")
    print(f"Reading hdf5 file... ", end="")
    f = dlf.HDF5File(mesh.mpi_comm(), fname, "r")
    if vector_valued:
        V = dlf.VectorFunctionSpace(mesh, cell_type, p_deg)
    else:
        V = dlf.FunctionSpace(mesh, cell_type, p_deg)
    u = dlf.Function(V)

    f.read(u, field_name)
    print("Done.")
    return u
