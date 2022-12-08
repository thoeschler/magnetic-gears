import dolfin as dlf
import gmsh
import os
from os import path
import numpy as np
from scipy.spatial.transform import Rotation
from source.magnet_classes import BallMagnet, BarMagnet
from source.magnetic_field import free_current_potential_bar_magnet
from source.mesh_generator import generate_mesh_with_markers


class MagneticGear:
    def __init__(self, n, R, x_M, magnetization_strength, initial_angle, index, main_dir=None):
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert path.exists(main_dir)
            self._main_dir = main_dir
        self._n = n  # the number of magnets
        self._R = R  # the radius of the gear
        self._x_M = x_M  # the mid point
        self._M_0 = magnetization_strength  # the magnetization strength
        self._angle = initial_angle  # angle in direction of spin
        self._index = index
        self._pad = self.R / 10.  # some padding value
    
    @property
    def n(self):
        return self._n

    @property
    def R(self):
        return self._R
    
    @property
    def x_M(self):
        return self._x_M

    @property
    def M_0(self):
        return self._M_0

    @property
    def alpha(self):
        return self._angle
    
    @property
    def index(self):
        return self._index
    
    @property
    def mesh(self):
        assert hasattr(self, "_mesh")
        return self._mesh
    
    @property
    def domain_radius(self):
        assert hasattr(self, "_domain_radius")
        return self._domain_radius

    def _create_magnets(self):
        "Purely virtual method."
        pass

    def _add_physical_groups_gmsh(self, model):
        """Add box and magnets as physical groups to gmsh model. The
        respective tags will later be used by the facet/cell markers.

        Args:
            model: the gmsh model
        """
        # box and magnet tags 
        assert hasattr(self, "box_entity")
        assert hasattr(self, "mag_entities")
        assert hasattr(self, "mag_boundary_entities")

        # store the physical group tags, they will later be used to reference subdomains
        self.boundary_subdomain_tags = []
        self.subdomain_tags = []

        # magnet surface
        for n, tag in enumerate(self.mag_boundary_entities):
            self.boundary_subdomain_tags.append(model.addPhysicalGroup(2, [tag], name="magnet_1_" + str(n + 1), tag=int("1%.2d" % (n + 1))))

        # volume
        self.box_subdomain = model.addPhysicalGroup(3, [self.box_entity], name="Box", tag=1)

        for n, tag in enumerate(self.mag_entities):
            self.subdomain_tags.append(model.addPhysicalGroup(3, [tag], name="magnet_1_" + str(n + 1), tag=int("3%.2d" % (n + 1))))

    def _set_differential_measures(self):
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_cell_marker")
        assert hasattr(self, "_facet_marker")

        self.normal_vector = dlf.FacetNormal(self._mesh)
        self.dV = dlf.Measure('dx', domain=self._mesh, subdomain_data=self._cell_marker)
        self.dA = dlf.Measure('dS', domain=self._mesh, subdomain_data=self._facet_marker)

    def _set_mesh_size_fields_gmsh(self, model, mesh_size_space, mesh_size_magnets):
        assert hasattr(self, "mag_entities")

        # global mesh size
        model.mesh.setSize(model.occ.getEntities(0), mesh_size_space)

        # mesh size field for magnets
        mag_field_tag = model.mesh.field.add("Constant")
        model.mesh.field.setNumber(mag_field_tag, "VIn", mesh_size_magnets)
        model.mesh.field.setNumbers(mag_field_tag, "VolumesList", self.mag_entities)

        # use the minimum of all the fields as the mesh size field
        min_tag = model.mesh.field.add("Min")
        model.mesh.field.setNumbers(min_tag, "FieldsList", [mag_field_tag])
        model.mesh.field.setAsBackgroundMesh(min_tag)

    def update_parameters(self, d_angle):
        # update the angle
        self._angle += d_angle
        # then update the rest
        self.update_magnets()
        self.update_mesh(d_angle)

    def update_magnets(self):
        assert hasattr(self, "magnets")
        for k, mag in enumerate(self.magnets):
            mag._Q = Rotation.from_rotvec((2 * np.pi / self.n * k + self._angle) * \
                np.array([1., 0., 0.])).as_matrix()
            mag._xM = self.x_M + np.array([0., self.R * np.cos(2 * np.pi / self.n * k + self._angle),
                                           self.R * np.sin(2 * np.pi / self.n * k + self._angle)])
            mag._M = mag._Q.dot(np.array([0., 0., 1.]))
    
    def update_mesh(self, d_angle):
        """Update mesh coordinates.

        Args:
            d_angle (float): Angle increment.
        """
        # rotate mesh around axis 0 (x-axis) through gear midpoint by angle d_angle
        self._mesh.rotate(d_angle * 180 / np.pi, 0, dlf.Point(*self.x_M))


class MagneticGearWithBallMagnets(MagneticGear):
    def __init__(self, n, r, R, x_M, magnetization_strength, initial_angle, index, main_dir=None):
        super().__init__(n, R, x_M, magnetization_strength, initial_angle, index, main_dir)
        self._r = r  # the magnet radius
        self._create_magnets()
        
    @property
    def r(self):
        return self._r

    def _create_cylinder_box_gmsh(self, model):
        # create the cylinder box 
        A = self.x_M - np.array([self.r + self._pad, 0., 0.])
        diff = np.array([2. * (self.r + self._pad), 0., 0.])
        self.box_entity = model.occ.addCylinder(*A, *diff, self.R + self.r + self._pad, tag=1)
        model.occ.synchronize()

        # set domain size
        self._domain_radius = self.R + self.r + self._pad

    def create_mesh(self, mesh_size_space, mesh_size_magnets, fname='ball_gear', write_to_file=False, verbose=False):
        print("Meshing gear... ", end="")
        gmsh.initialize()
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        model = gmsh.model()

        self._create_cylinder_box_gmsh(model)

        # add magnets
        self.mag_entities = []
        
        for n, mag in enumerate(self.magnets):
            mag_gmsh = model.occ.addSphere(*(mag.x_M), self.r)
            model.occ.cut([(3, self.box_entity)], [(3, mag_gmsh)], removeObject=True, removeTool=False)
            self.mag_entities.append(mag_gmsh)

        model.occ.synchronize()

        # get boundary entities
        self.mag_boundary_entities = [model.getBoundary([(3, mag)], oriented=False)[0][1] for mag in self.mag_entities]

        self._add_physical_groups_gmsh(model)
        self._set_mesh_size_fields_gmsh(model, mesh_size_space, mesh_size_magnets)

        # generate mesh
        model.mesh.generate(dim=3)

        # write mesh to msh file
        gmsh.write(self._main_dir + "/meshes/gears/" + fname + '.msh')

        self._mesh, self._cell_marker, self._facet_marker = generate_mesh_with_markers(self._main_dir + "/meshes/gears/" + fname, delete_source_files=False)
        
        if write_to_file:
            dlf.File(self._main_dir + "/meshes/gears/" + fname + "_mesh.pvd") << self._mesh
            dlf.File(self._main_dir + "/meshes/gears/" + fname + "_cell_markers.pvd") << self._cell_marker
            dlf.File(self._main_dir + "/meshes/gears/" + fname + "_facet_marker.pvd") << self._facet_marker

        gmsh.finalize()

        self._set_differential_measures()
        print("Done.")

    def _create_magnets(self):
        # info text
        print("Creating magnets... ", end="")

        self.magnets = []

        for k in range(self.n):
            # compute position and rotation matrix
            x_M = self.x_M + np.array([0.,
                                       self.R * np.cos(2 * np.pi / self.n * k + self.alpha),
                                       self.R * np.sin(2 * np.pi / self.n * k + self.alpha)])
            Q = Rotation.from_rotvec((2 * np.pi / self.n * k + self.alpha) * \
                    np.array([1., 0., 0.])).as_matrix()
            self.magnets.append(BallMagnet(self.r, self.M_0, x_M, Q))
        
        print("Done.")

    def B(self, x_0, limit_direction=-1):
        # takes 56s for n1=7, n2=9, mesh_size_space=2.0, mesh_size_magnets=0.3
        B = np.zeros(3)
        for mag in self.magnets:
            B += np.dot(mag.Q, mag.B_eigenfield_dimless(mag.Q.T.dot(x_0 - mag.x_M)))
        return B

class MagneticGearWithBarMagnets(MagneticGear):
    def __init__(self, n, h, w, d, R, x_M, magnetization_strength, initial_angle, index, main_dir=None):
        super().__init__(n, R, x_M, magnetization_strength, initial_angle, main_dir)
        self._h = h  # the magnet height
        self._w = w  # the magnet width
        self._d = d  # the magnet depth
        self._create_magnets()
        self.H = free_current_potential_bar_magnet(self.h, self.w, self.d)
        
    @property
    def h(self):
        return self._h

    @property
    def w(self):
        return self._w

    @property
    def d(self):
        return self._d

    def _create_magnets(self):
        # info text
        print("Creating magnets... ")

        self.magnets = []

        for k in range(self.n):
            # compute position and rotation matrix
            pos = self.x_M + np.array([0.,
                                       self.R * np.cos(2 * np.pi / self.n * k + self.alpha),
                                       self.R * np.sin(2 * np.pi / self.n * k + self.alpha)])
            rot = Rotation.from_rotvec((2 * np.pi / self.n * k + self.alpha) * \
                np.array([1., 0., 0.])).as_matrix()
            self.magnets.append(BarMagnet(self.h, self.w, self.d, self.M_0, pos, rot))
        
        print("Done.")

    def B(self, x_0, limit_direction=-1):
        # takes 98s for n1=1, n2=1, mesh_size_space=2.0, mesh_size_magnets=0.3
        B = np.zeros(3)
        for mag in self.magnets:
            B += np.dot(mag.Q, mag.B_eigenfield_dimless(np.dot(mag.Q.T, x_0 - mag.x_M)))
        return B


if __name__ == "__main__":
    testGear = MagneticGearWithBallMagnets(10, 1, 20, np.zeros(3), 1., 0.)
    testGear.create_mesh(1.0, 0.3, write_to_file=True, verbose=True)