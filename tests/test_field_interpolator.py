from source.field_interpolator import FieldInterpolator
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
    fi = FieldInterpolator(domain_radius=5, cell_type="CG", p_deg=1, mesh_size_min=0.3, mesh_size_max=1.0, main_dir=test_dir)

    # file names
    mesh_fname = "reference_mesh.xdmf"

    # reference magnet
    ref_mag = BallMagnet(1., 1., np.zeros(3), np.eye(3))

    # create mesh, write it to xdmf file and read it
    fi.create_reference_mesh(ref_mag, mesh_fname)
    fi.read_reference_mesh(mesh_fname)

    # interpolate reference field
    B_interpol = fi.interpolate_reference_field(ref_mag.B, "B", write_pvd=False)
    field_name = "B"
    field_file_name = "B.h5"

    # write interpolated field to hdf5 file and read it
    fi.write_hdf5_file(B_interpol, field_file_name, field_name)
    B = fi.read_hd5f_file(field_file_name, field_name, vector_valued=True)

    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)
