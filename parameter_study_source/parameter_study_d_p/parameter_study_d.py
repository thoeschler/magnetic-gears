import numpy as np
import itertools as it
import sys
import os
from source.magnetic_gear_classes import SegmentGear
from parameter_study_source.sample_base import SpurGearsSampling, write_parameter_file


def sample_segment(n_iterations, par_number):
    d_values = np.array([0.05, 0.1, 0.2, 0.4])
    p_values = np.arange(4, 40, 2)

    t1 = 1.0
    t2 = t1
    R1 = 10.
    R2 = R1
    w1 = 3.0
    w2 = w1

    # create sample directory
    sample_dir = "parameter_study_d"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    mesh_size = 0.25
    p_deg = 2

    par_list = list(it.product(d_values, p_values))
    par = par_list[par_number]
    d, p = par
    p1 = p
    p2 = p

    # create directory
    d_dir = f"d_{str(d).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = os.path.join(d_dir, pole_nb_dir)
    target_dir = os.path.join(sample_dir, data_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    gear_1 = SegmentGear(p1, R1, w1, t1, np.zeros(3))
    gear_2 = SegmentGear(p2, R2, w2, t2, np.array([0., 1., 0.]))
    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    D = gear_1.outer_radius + gear_2.outer_radius + d

    # use same main dir for all parameter combination
    # this way all reference fields can be used in each iteration
    sampling = SpurGearsSampling(gear_1, gear_2, D, main_dir=sample_dir)

    # mesh smaller gear
    sampling.mesh_gear(sampling.sg, mesh_size=mesh_size, 
                    fname=f"gear_{1 if sampling.sg is sampling.gear_1 else 2}_{id(sampling)}", \
                    write_to_pvd=False)

    # mesh the segment
    sampling.load_reference_field(sampling.lg, "Vm", "CG", p_deg=p_deg, mesh_size_min=mesh_size, \
                                    mesh_size_max=mesh_size, domain_size=sampling.domain_size, \
                                    analytical_solution=False, write_to_pvd=False)
    sampling.mesh_reference_segment(mesh_size, write_to_pvd=False)
    sampling.interpolate_to_reference_segment(p_deg=p_deg, interpolate="twice", use_Vm=True)

    write_parameter_file(sampling, target_dir)

    sampling.sample(n_iterations, data_dir=data_dir, p_deg=p_deg)

if __name__ == "__main__":
    par_number = int(sys.argv[1])
    sample_segment(5, par_number)