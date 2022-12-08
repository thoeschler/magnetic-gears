import gmsh
import dolfin as dlf
import numpy as np
import os
from magnet_classes import BallMagnet, BarMagnet, CustomVectorExpression, CustomScalarExpression
from mesh_generator import generate_xdmf_mesh


class FieldInterpolator:
    def __init__(self, domain_radius, cell_type, p_deg, mesh_size_min, mesh_size_max, main_dir=None):
        """_summary_

        Args:
            domain_radius (float): When interpolating a field a spherical reference mesh with
                                   this radius will be used. 
            cell_type (str): Finite element cell type.
            p_deg (int): Polynomial degree of finite element.
            mesh_size_min (float): Minimum mesh size of reference field. This value
                                   will be used as the starting mesh size at the center
                                   of the reference mesh.
            mesh_size_max (float): Maximum mesh size of reference mesh. The reference
                                   mesh size is increased up to this value with increasing
                                   distance from the center of the reference mesh.
            path (str, optional): The main path. Defaults to None.
        """
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert main_dir.exists(main_dir)
            self._main_dir = main_dir
        self._domain_radius = domain_radius
        self._cell_type = cell_type
        self._p_deg = p_deg
        self._mesh_size_min = mesh_size_min
        self._mesh_size_max = mesh_size_max

    @property
    def domain_radius(self):
        return self._domain_radius
        
    @property
    def p_deg(self):
        return self._p_deg
    
    @property
    def cell_type(self):
        return self._cell_type
    
    @property
    def mesh_size_min(self):
        return self._mesh_size_min

    @property
    def mesh_size_max(self):
        return self._mesh_size_max

    def set_mesh(self, mesh):
        self.mesh = mesh

    def create_reference_mesh(self, fname="reference_mesh", type_classifier="Ball", verbose=False):
        """Create a reference mesh.

        Args:
            fname (str, optional): Output file name. Defaults to "reference_mesh".
            type_classifier (str, optional): Classifies magnet type. Defaults to "Ball".
            verbose (bool, optional): If true the gmsh meshing information will be
                                      displayed. Defaults to False.

        """
        print("Creating reference mesh... ", end="")
        gmsh.initialize()
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model()

        # create surrounding box sphere
        x_M = np.zeros(3)
        box = model.occ.addSphere(*x_M, self.domain_radius)
        if type_classifier == "Ball":
            magnet = model.occ.addSphere(0., 0., 0., 1.)
        elif type_classifier == "Bar":
            raise NotImplementedError()
        else:
            raise RuntimeError()

        # cut magnet from surrounding box
        model.occ.cut([(3, box)], [(3, magnet)], removeObject=True, removeTool=False)
        model.occ.synchronize()

        # add physical groups
        bndry_ids = np.array(model.getBoundary([(3, box)]))[:, 1]
        model.addPhysicalGroup(2, bndry_ids)
        model.addPhysicalGroup(3, [box])
        model.addPhysicalGroup(3, [magnet])

        # add distance field
        mid_point = model.occ.addPoint(0., 0., 0.)
        model.occ.synchronize()
        distance_tag = model.mesh.field.add("Distance")
        model.mesh.field.setNumbers(distance_tag, "PointsList", [mid_point])
        
        # add MathEval field that depends on distance
        math_eval_tag = model.mesh.field.add("MathEval")
        model.mesh.field.setString(math_eval_tag, "F", f"F{distance_tag} / {self.domain_radius} * "\
            + f"{self.mesh_size_max - self.mesh_size_min} + {self.mesh_size_min}")

        # use the minimum of all the fields as the mesh size field
        min_tag = model.mesh.field.add("Min")
        model.mesh.field.setNumbers(min_tag, "FieldsList", [math_eval_tag])
        model.mesh.field.setAsBackgroundMesh(min_tag)

        # generate mesh
        model.mesh.generate(3)

        # create directory if it does not exist
        if not os.path.exists(self._main_dir + "/meshes/reference"):
            os.mkdir(self._main_dir + "/meshes/reference")

        gmsh.write(self._main_dir + "/meshes/reference/" + fname + ".msh")

        # generate xdmf mesh
        generate_xdmf_mesh(self._main_dir + "/meshes/reference/" + fname + ".msh", delete_source_files=True)

        gmsh.finalize()
        print("Done.")

    def read_reference_mesh(self, fname_xdmf):
        """Read reference mesh from xdmf file.

        Args:
            fname_xdmf (str): Mesh file name (xmdf).
        """
        assert fname_xdmf.endswith(".xdmf")
        mesh_file = dlf.XDMFFile(self._main_dir + "/meshes/reference/" + fname_xdmf)
        self.mesh = dlf.Mesh()
        mesh_file.read(self.mesh)

    def interpolate_reference_field(self, field, fname=None, write_pvd=False):
        """Interpolate a given field on the reference mesh.

        Args:
            field (callable): _description_
            fname (str, optional): Output file name. Defaults to None.
            write_pvd (bool, optional): If true write mesh and markers to paraview (pvd)
                                        file. Defaults to False.

        Returns:
            _type_: _description_
        """
        assert isinstance(fname, str)
        assert hasattr(self, "mesh")

        # test whether the field is scalar or vector-valued (evaluate at random point (0, 0, 0))
        vector_valued = np.atleast_1d(field(np.zeros(3))).size > 1 
        print("Interpolating reference field ...", end="")

        if vector_valued:
            V = dlf.VectorFunctionSpace(self.mesh, self.cell_type, self.p_deg)
            field_expr = CustomVectorExpression(field)
        else:
            V = dlf.FunctionSpace(self.mesh, self.cell_type, self.p_deg)
            field_expr = CustomScalarExpression(field, dim=1)

        field_interpolated = dlf.interpolate(field_expr, V)
        if write_pvd:
            assert fname is not None
            dlf.File(self._main_dir + "/results/" + fname + ".pvd") << field_interpolated

        print("Done.")
        return field_interpolated

    def write_hdf5_file(self, field, fname, field_name):
        """Write field to hdf5 file.

        Args:
            field (dlf.Function): The interpolated field.
            fname (str): Output file name.
            field_name (str): Name of the interpolated field.
        """
        print(f"Writing {fname}... ", end="")
        f = dlf.HDF5File(self.mesh.mpi_comm(), self._main_dir + "/data/" + fname, "w")
        f.write(field, field_name)
        print("Done")

    def read_hd5f_file(self, fname, field_name, vector_valued=False):
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
        print(f"Reading {fname}... ", end="")
        f = dlf.HDF5File(self.mesh.mpi_comm(), self._main_dir + "/data/" + fname, "r")
        if vector_valued:
            V = dlf.VectorFunctionSpace(self.mesh, self.cell_type, self.p_deg)
        else:
            V = dlf.FunctionSpace(self.mesh, self.cell_type, self.p_deg)
        u = dlf.Function(V)

        f.read(u, field_name)
        print("Done.")
        return u

    
if __name__ == "__main__":
    fs = FieldInterpolator(pad=5, p_deg=1, cell_type="DG", mesh_size=0.3)
    
    # file names
    mesh_fname = "reference_mesh"
    
    # create mesh, write it to xdmf file and read it
    fs.create_reference_mesh(mesh_fname)
    fs.read_reference_mesh(mesh_fname)
    print(fs.mesh)

    # create reference magnet
    ref_mag = BallMagnet(1., 1., np.zeros(3), np.eye(3))
    B_interpol = fs.interpolate_reference_field(ref_mag.V_m, "V_m", write_pvd=True)
    field_name = "V_m"
    field_file_name = "V_m.h5"

    fs.write_hdf5_file(B_interpol, field_file_name, field_name)
    B = fs.read_hd5f_file(field_file_name, field_name)
