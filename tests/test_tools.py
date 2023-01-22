from source.tools import create_reference_mesh, read_hd5f_file, write_hdf5_file, interpolate_field
from source.mesh_tools import read_mesh
from source.magnet_classes import BallMagnet
import numpy as np
import os
import subprocess

if __name__ == "__main__":
    # create test directory
    test_dir = os.getcwd() + "/test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # create field interpolator
    domain_radius = 5
    cell_type = "CG"
    p_deg = 1
    mesh_size_min = 0.3
    mesh_size_max = 1.0
    
    # file names
    mesh_fname = "reference_mesh.xdmf"

    # reference magnet
    ref_mag = BallMagnet(1., 1., np.zeros(3), np.eye(3))

    # create mesh, write it to xdmf file and read it
    create_reference_mesh(ref_mag, domain_radius, mesh_size_min, mesh_size_max, f"{test_dir}/{mesh_fname}.xdmf")
    mesh = read_mesh(f"{test_dir}/{mesh_fname}")

    # interpolate reference field
    B_interpol = interpolate_field(ref_mag.B, mesh, cell_type, p_deg, "B", write_pvd=False)
    field_name = "B"
    field_file_name = "B.h5"

    # write interpolated field to hdf5 file and read it
    write_hdf5_file(B_interpol, mesh, f"{test_dir}/{field_file_name}", field_name)
    B = read_hd5f_file(f"{test_dir}/{field_file_name}", field_name, mesh, cell_type, p_deg, vector_valued=True)

    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)
