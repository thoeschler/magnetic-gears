import gmsh
import dolfin as dlf
import numpy as np
from source.mesh_tools import generate_mesh_with_markers
import os
from collections import namedtuple


class GearMeshGenerator:
    def __init__(self, magnetic_gear, main_dir=None):
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert os.path.exists(main_dir)
            self._main_dir = main_dir

        self._magnetic_gear = magnetic_gear
        self._model = gmsh.model()
        # some padding value for mesh generation
        self._pad = self.gear.R / 10.

    @property
    def gear(self):
        return self._magnetic_gear

    @property
    def pad(self):
        return self._pad

    def _add_box(self):
        "Purely virtual method."
        pass

    def _add_magnet(self):
        "Purely virtual method."
        pass

    def _add_physical_groups_gmsh(self):
        """Add box and magnets as physical groups to gmsh model.
        
        The respective tags will later be used by the facet/cell markers.
        """
        # box and magnet tags
        assert hasattr(self, "_box_entity")

        # store the physical group tags, they will later be used to reference subdomains
        magnet_subdomain_tags = []
        magnet_boundary_subdomain_tags = []

        # magnet boundary
        for n, tag in enumerate(self._mag_boundary_entities):
            physical_tag = self._model.addPhysicalGroup(2, np.atleast_1d(tag), name="magnet_" + str(n + 1), tag=int("1%.2d" % (n + 1)))
            magnet_boundary_subdomain_tags.append(physical_tag)

        # box volume
        box_subdomain_tag = self._model.addPhysicalGroup(3, [self._box_entity], name="box", tag=1)

        # magnet volume
        for n, tag in enumerate(self._mag_entities):
            physical_tag = self._model.addPhysicalGroup(3, [tag], name="magnet_" + str(n + 1), tag=int("3%.2d" % (n + 1)))
            magnet_subdomain_tags.append(physical_tag)

        return magnet_subdomain_tags, magnet_boundary_subdomain_tags, box_subdomain_tag

    def _set_mesh_size_fields_gmsh(self, mesh_size_space, mesh_size_magnets):
        assert hasattr(self, "_mag_entities")

        # global mesh size
        self._model.mesh.setSize(self._model.occ.getEntities(0), mesh_size_space)

        # mesh size field for magnets
        mag_field_tag = self._model.mesh.field.add("Constant")
        self._model.mesh.field.setNumber(mag_field_tag, "VIn", mesh_size_magnets)
        self._model.mesh.field.setNumbers(mag_field_tag, "VolumesList", self._mag_entities)

        # use the minimum of all the fields as the mesh size field
        min_tag = self._model.mesh.field.add("Min")
        self._model.mesh.field.setNumbers(min_tag, "FieldsList", [mag_field_tag])
        self._model.mesh.field.setAsBackgroundMesh(min_tag)

    def get_differential_measures(self, mesh, cell_marker, facet_marker):
        # include normal vector as well
        normal_vector = dlf.FacetNormal(mesh)
        dV = dlf.Measure('dx', domain=mesh, subdomain_data=cell_marker)
        dA = dlf.Measure('dS', domain=mesh, subdomain_data=facet_marker)

        return normal_vector, dV, dA

    def generate_mesh(self, mesh_size_space, mesh_size_magnets, fname, write_to_pvd=True, verbose=False):
        # check input
        assert isinstance(mesh_size_space, float)
        assert isinstance(mesh_size_magnets, float)
        assert isinstance(fname, str)
        fname = fname.rstrip(".xdmf")

        print("Meshing gear... ", end="")
        gmsh.initialize()
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        
        # add surrounding box
        assert hasattr(self, "_add_box")
        self._add_box()

        # add magnets
        assert hasattr(self, "_add_magnet")
        self._mag_entities = []
        for magnet in self.gear.magnets:
            magnet_tag = self._add_magnet(self._model, magnet)
            self._model.occ.cut([(3, self._box_entity)], [(3, magnet_tag)], removeObject=True, removeTool=False)
            self._mag_entities.append(magnet_tag)
        self._model.occ.synchronize()

        # get boundary entities
        self._mag_boundary_entities = [self._model.getBoundary([(3, magnet_tag)], oriented=False)[0][1] \
            for magnet_tag in self._mag_entities]

        # create namedtuple
        magnet_subdomain_tags, magnet_boundary_subdomain_tags, box_subdomain_tag = self._add_physical_groups_gmsh()
        Tags = namedtuple("Tags", ["magnet_subdomain", "magnet_boundary_subdomain", "box_subdomain"])
        tags = Tags(magnet_subdomain=magnet_subdomain_tags, magnet_boundary_subdomain=magnet_boundary_subdomain_tags, \
            box_subdomain=box_subdomain_tag)
        
        # set mesh size fields
        self._set_mesh_size_fields_gmsh(mesh_size_space, mesh_size_magnets)

        # generate mesh
        self._model.mesh.generate(dim=3)

        # write mesh to msh file
        if not os.path.exists(self._main_dir + "/meshes/gears"):
            os.makedirs(self._main_dir + "/meshes/gears")
        gmsh.write(self._main_dir + "/meshes/gears/" + fname + '.msh')

        # create namedtuple
        mesh, cell_marker, facet_marker = generate_mesh_with_markers(self._main_dir + "/meshes/gears/" + fname, delete_source_files=False)
        Mesh = namedtuple("Mesh", ["mesh", "cell_marker", "facet_marker"])
        mesh_and_marker = Mesh(mesh=mesh, cell_marker=cell_marker, facet_marker=facet_marker)

        if write_to_pvd:
            dlf.File(self._main_dir + "/meshes/gears/" + fname + "_mesh.pvd") << mesh
            dlf.File(self._main_dir + "/meshes/gears/" + fname + "_cell_marker.pvd") << cell_marker
            dlf.File(self._main_dir + "/meshes/gears/" + fname + "_facet_marker.pvd") << facet_marker

        gmsh.finalize()
        print("Done.")

        return mesh_and_marker, tags


class GearWithBallMagnetsMeshGenerator(GearMeshGenerator):
    def _add_box(self):
        """Create the surrounding cylindrical box."""
        # create the cylindrical box 
        A = self.gear.x_M - np.array([self.gear.r + self._pad, 0., 0.])
        diff = np.array([2. * (self.gear.r + self._pad), 0., 0.])
        self._box_entity = self._model.occ.addCylinder(*A, *diff, self.gear.R + self.gear.r + self._pad, tag=1)
        self._model.occ.synchronize()

    @staticmethod
    def _add_magnet(model, magnet):
        magnet_tag = model.occ.addSphere(*magnet.x_M, magnet.R)
        return magnet_tag

    def get_padded_radius(self):
        return self.gear.R + self.gear.r + self._pad


class GearWithBarMagnetsMeshGenerator(GearMeshGenerator):
    def _add_box(self):
        """Create the surrounding cylindrical box."""
        A = self.gear.x_M - np.array([self.gear.d + self._pad, 0., 0.])
        diff = np.array([2. * (self.gear.d + self._pad), 0., 0.])
        self._box_entity = self._model.occ.addCylinder(*A, *diff, self.gear.R + self.gear.w + self._pad, tag=1)
        self._model.occ.synchronize()

    @staticmethod
    def _add_magnet(model, magnet):
        # add magnet corner points
        p1 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.d, magnet.w, magnet.h]))))
        p2 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.d, - magnet.w, magnet.h]))))
        p3 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.d, - magnet.w, - magnet.h]))))
        p4 = model.occ.addPoint(*(magnet.x_M + magnet.Q.dot(np.array([- magnet.d, magnet.w, - magnet.h]))))
        
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
        magnet_gmsh = model.occ.extrude([(2, surf)], 2 * magnet.d, 0, 0)

        # find entity with dimension 3 (the extruded volume) and save its tag
        index = np.where(np.array(magnet_gmsh)[:, 0] == 3)[0].item()
        magnet_tag = magnet_gmsh[index][1]

        return magnet_tag

    def get_padded_radius(self):
        return self.gear.R + self.gear.w + self._pad
