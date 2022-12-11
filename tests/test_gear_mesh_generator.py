import dolfin as dlf
import numpy as np
from source.gear_mesh_generator import GearWithBallMagnetsMeshGenerator
from source.magnetic_gear_classes import MagneticGearWithBallMagnets
import subprocess
import os


def test_mesh_generator_directly():
     # create test directory
    test_dir = os.getcwd() + "/test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # create gear
    gear = MagneticGearWithBallMagnets(5, 1, 10, np.zeros(3), 1., 0., 0, main_dir=test_dir)

    # create mesh generator
    mg = GearWithBallMagnetsMeshGenerator(gear, main_dir=test_dir)

    # generate mesh and markers
    mesh_and_marker, tags = mg.generate_mesh(mesh_size_space=1.0, mesh_size_magnets=0.5, fname="test_mesh.xdmf", \
        write_to_pvd=True)

    # unpack mesh, marker and tags
    mesh = mesh_and_marker.mesh
    cell_marker, facet_marker = mesh_and_marker.cell_marker, mesh_and_marker.facet_marker
    magnet_subdomain_tags, magnet_boundary_subdomain_tags = tags.magnet_subdomain, tags.magnet_boundary_subdomain
    box_subdomain_tag = tags.box_subdomain

    # check output
    assert isinstance(mesh, dlf.Mesh)
    assert isinstance(cell_marker, dlf.cpp.mesh.MeshFunctionSizet)
    assert isinstance(facet_marker, dlf.cpp.mesh.MeshFunctionSizet)
    assert isinstance(magnet_subdomain_tags, list)
    assert isinstance(magnet_boundary_subdomain_tags, list)
    assert isinstance(box_subdomain_tag, int)
    
    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)

def test_mesh_generator_indirectly():
     # create test directory
    test_dir = os.getcwd() + "/test_dir"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # create gear
    gear = MagneticGearWithBallMagnets(5, 1, 10, np.zeros(3), 1., 0., 0, main_dir=test_dir)

    # generate mesh and markers
    gear.generate_mesh_and_markers(mesh_size_space=1.0, mesh_size_magnets=0.5, \
        fname="test_mesh.xdmf", write_to_pvd=True)

    # check output
    assert isinstance(gear._mesh, dlf.Mesh)
    assert isinstance(gear._cell_marker, dlf.cpp.mesh.MeshFunctionSizet)
    assert isinstance(gear._facet_marker, dlf.cpp.mesh.MeshFunctionSizet)
    assert isinstance(gear._magnet_subdomain_tags, list)
    assert isinstance(gear._magnet_boundary_subdomain_tags, list)
    assert isinstance(gear._box_subdomain_tag, int)
    
    # remove test directory
    subprocess.run(["rm", "-rf",  test_dir], stdout=subprocess.DEVNULL, check=True)


if __name__ == "__main__":
    test_mesh_generator_directly()
    test_mesh_generator_indirectly()