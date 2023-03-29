from source.tools.tools import create_reference_mesh, read_hd5f_file, write_hdf5_file, interpolate_field
from source.tools.mesh_tools import read_mesh
from source.magnet_classes import BallMagnet, BarMagnet, CylinderSegment
import numpy as np
import os
import subprocess


def test_ball_magnet(test_dir):
    # create field interpolator
    domain_radius = 5
    cell_type = "CG"
    p_deg = 1
    mesh_size_min = 0.3
    mesh_size_max = 1.0

    # file names
    mesh_fname = "reference_mesh"

    # reference magnet
    ref_mag = BallMagnet(1., 1., np.zeros(3), np.eye(3))

    # create mesh, write it to xdmf file and read it
    create_reference_mesh(ref_mag, domain_radius, mesh_size_min, mesh_size_max, fname=f"{test_dir}/{mesh_fname}")
    mesh = read_mesh(f"{test_dir}/{mesh_fname}.xdmf")

    # interpolate reference field
    Vm_interpol = interpolate_field(ref_mag.Vm, mesh, cell_type, p_deg, fname=f"{test_dir}/Vm_ball", write_pvd=True)
    field_name = "Vm"
    field_file_name = "Vm_ball.h5"

    # write interpolated field to hdf5 file and read it
    write_hdf5_file(Vm_interpol, mesh, f"{test_dir}/{field_file_name}", field_name)
    _ = read_hd5f_file(f"{test_dir}/{field_file_name}", field_name, mesh, cell_type, p_deg, vector_valued=False)

    # interpolate reference field
    B_interpol = interpolate_field(ref_mag.B, mesh, cell_type, p_deg, fname=f"{test_dir}/B_ball", write_pvd=True)
    field_name = "B"
    field_file_name = "B_ball.h5"

    # write interpolated field to hdf5 file and read it
    write_hdf5_file(B_interpol, mesh, f"{test_dir}/{field_file_name}", field_name)
    _ = read_hd5f_file(f"{test_dir}/{field_file_name}", field_name, mesh, cell_type, p_deg, vector_valued=True)

def test_bar_magnet(test_dir):
    # create field interpolator
    domain_radius = 7
    cell_type = "CG"
    p_deg = 1
    mesh_size_min = 0.3
    mesh_size_max = 1.0

    # file names
    mesh_fname = "reference_mesh_bar"

    # reference magnet
    ref_mag = BarMagnet(width=2., depth=1., height=5., magnetization_strength=1., \
                        position_vector=np.zeros(3), rotation_matrix=np.eye(3))

    # create mesh, write it to xdmf file and read it
    create_reference_mesh(ref_mag, domain_radius, mesh_size_min, mesh_size_max, fname=f"{test_dir}/{mesh_fname}")
    mesh = read_mesh(f"{test_dir}/{mesh_fname}.xdmf")

    # interpolate reference field
    Vm_interpol = interpolate_field(ref_mag.Vm, mesh, cell_type, p_deg, fname=f"{test_dir}/Vm_bar", write_pvd=True)
    field_name = "Vm"
    field_file_name = "Vm_bar.h5"

    # write interpolated field to hdf5 file and read it
    write_hdf5_file(Vm_interpol, mesh, f"{test_dir}/{field_file_name}", field_name)
    _ = read_hd5f_file(f"{test_dir}/{field_file_name}", field_name, mesh, cell_type, p_deg, vector_valued=False)

    # interpolate reference field
    B_interpol = interpolate_field(ref_mag.B, mesh, cell_type, p_deg, fname=f"{test_dir}/B_bar", write_pvd=True)
    field_name = "B"
    field_file_name = "B_bar.h5"

    # write interpolated field to hdf5 file and read it
    write_hdf5_file(B_interpol, mesh, f"{test_dir}/{field_file_name}", field_name)
    _ = read_hd5f_file(f"{test_dir}/{field_file_name}", field_name, mesh, cell_type, p_deg, vector_valued=True)

def test_magnet_segment(test_dir):
    # create field interpolator
    domain_radius = 13.
    mesh_size_min = 0.1
    mesh_size_max = 1.0

    # file names
    mesh_fname = "reference_mesh_segment"

    # reference magnet
    ref_mag = CylinderSegment(radius=5., width=1., thickness=1., alpha=np.pi / 4, magnetization_strength=1.0, \
                            position_vector=np.zeros(3), rotation_matrix=np.eye(3))

    # create mesh, write it to xdmf file and read it
    create_reference_mesh(ref_mag, domain_radius, mesh_size_min, mesh_size_max, fname=f"{test_dir}/{mesh_fname}")
    _ = read_mesh(f"{test_dir}/{mesh_fname}.xdmf")


if __name__ == "__main__":
    # create test directory
    test_dir = os.getcwd() + "/test_tools"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    test_ball_magnet(test_dir)
    test_bar_magnet(test_dir)
    test_magnet_segment(test_dir)

    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)
