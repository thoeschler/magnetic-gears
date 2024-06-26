import numpy as np
import os
import itertools as it
from math import ceil
from source.magnetic_gear_classes import MagneticBallGear, MagneticBarGear, SegmentGear
from parameter_study_source.sample_base import SpurGearsSampling, write_parameter_file


def sample_ball(n_iterations, par_number):
    # create sample directory
    sample_dir = "sample_ball_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    R1 = 10.
    mesh_size = 0.25
    p_deg = 2
    d = 0.1

    # parameters
    gear_ratio_values = np.array([1.0, 1.4, 2.0, 2.4])
    p1_values = list(range(6, 72, 2))
    par_list = list(it.product(gear_ratio_values, p1_values))
    par = par_list[par_number]
    gear_ratio, p1 = par

    # compute rest of values
    R2 = R1 * gear_ratio
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()
    p2 = int(p2)

    r1 = R1 * np.sin(np.pi / p1)
    r2 = R2 * np.sin(np.pi / p2)

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = f"{gear_ratio_dir}/{pole_nb_dir}"
    target_dir = os.path.join(sample_dir, data_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    gear_1 = MagneticBallGear(p1, R1, r1, np.zeros(3))
    gear_2 = MagneticBallGear(p2, R2, r2, np.array([0., 1., 0.]))
    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    D = gear_1.outer_radius + gear_2.outer_radius + d

    # use same main dir for all parameter combination
    # this way all reference fields can be used in each iteration
    sampling = SpurGearsSampling(gear_1, gear_2, D, main_dir=sample_dir)

    # mesh larger gear
    sampling.mesh_gear(sampling.lg, mesh_size=mesh_size, fname=f"gear_{1 if sampling.lg is sampling.gear_1 else 2}", \
                       write_to_pvd=False)

    # set angles for sampling
    if sampling.gear_1 is sampling.sg:
        angle_1_min = 0.
        angle_1_max = np.pi / sampling.gear_1.p
        angle_2_min = - np.pi / sampling.gear_2.p
        angle_2_max = np.pi / sampling.gear_2.p
        phi_sg_min = angle_1_min
        phi_sg_max = angle_1_max
        phi_lg_min = angle_2_min
        phi_lg_max = angle_2_max
        n1 = ceil(n_iterations / 2)
        n2 = n_iterations
    elif sampling.gear_2 is sampling.sg:
        angle_1_min = - np.pi / sampling.gear_1.p
        angle_1_max = np.pi / sampling.gear_1.p
        angle_2_min = 0.
        angle_2_max = np.pi / sampling.gear_2.p
        phi_sg_min = angle_2_min
        phi_sg_max = angle_2_max
        phi_lg_min = angle_1_min
        phi_lg_max = angle_1_max
        n1 = n_iterations
        n2 = ceil(n_iterations / 2)
    angles_1 = np.linspace(angle_1_min, angle_1_max, num=n1, endpoint=True)
    angles_2 = np.linspace(angle_2_min, angle_2_max, num=n2, endpoint=True)

    sampling.load_reference_field(sampling.sg, "Vm", "CG", p_deg=p_deg, mesh_size_min=mesh_size, \
                                    mesh_size_max=mesh_size, domain_size=sampling.domain_size, \
                                    analytical_solution=True, write_to_pvd=False)
    # create reference segment
    sampling.mesh_reference_segment(mesh_size, phi_sg_min=phi_sg_min, phi_sg_max=phi_sg_max,
                                    phi_lg_min=phi_lg_min, phi_lg_max=phi_lg_max, write_to_pvd=False)
    sampling.interpolate_to_reference_segment(p_deg=p_deg, interpolate="twice", use_Vm=True)

    write_parameter_file(sampling, target_dir)

    sampling.sample(angles_1, angles_2, data_dir=data_dir, p_deg=p_deg)


def sample_bar(n_iterations, par_number):
    t1 = 1.0
    R1 = 10.
    d_ref = 0.1
    w_ref = 3.
    # create sample directory
    sample_dir = "sample_bar_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    mesh_size = 0.25
    p_deg = 2

    # parameters
    gear_ratio_values = np.array([1.0, 1.4, 2.0, 2.4])
    R_ref_values = np.array([6., 10., 20.])
    scaling_factors = R1 / R_ref_values

    w1_values = w_ref * scaling_factors
    p1_values = list(range(6, 72, 2))
    par_list = list(it.product(gear_ratio_values, w1_values, p1_values))
    par = par_list[par_number]
    gear_ratio, w1, p1 = par
    w2 = w1

    # compute rest of values
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()

    # compute d
    d = d_ref * w1 / w_ref

    # compute d
    p2 = int(p2)
    d1 = np.tan(np.pi / p1) * (2 * R1 - t1)
    d2 = d1

    # compute R2 and t2
    R2 = R1 * np.tan(np.pi / p1) / np.tan(np.pi / p2)
    t2 = t1 * np.tan(np.pi / p1) / np.tan(np.pi / p2)

    # assert dimensioning
    assert np.isclose(t2 / (R2 + t2 / 2), t1 / (R1 + t1 / 2))
    assert np.isclose(d2, np.tan(np.pi / p2) * (2 * R2 - t2))

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    width_dir = f"w1_{str(w1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = os.path.join(gear_ratio_dir, width_dir, pole_nb_dir)
    target_dir = os.path.join(sample_dir, data_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    gear_1 = MagneticBarGear(p1, R1, w1, t1, d1, np.zeros(3))
    gear_2 = MagneticBarGear(p2, R2, w2, t2, d2, np.array([0., 1., 0.]))
    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    D = gear_1.outer_radius + gear_2.outer_radius + d

    # use same main dir for all parameter combination
    # this way all reference fields can be used in each iteration
    sampling = SpurGearsSampling(gear_1, gear_2, D, main_dir=sample_dir)

    # mesh larger gear
    sampling.mesh_gear(sampling.lg, mesh_size=mesh_size, \
                       fname=f"gear_{1 if sampling.lg is sampling.gear_1 else 2}_{id(sampling)}", \
                        write_to_pvd=False)

    # set angles for sampling
    if sampling.gear_1 is sampling.sg:
        angle_1_min = 0.
        angle_1_max = np.pi / sampling.gear_1.p
        angle_2_min = - np.pi / sampling.gear_2.p
        angle_2_max = np.pi / sampling.gear_2.p
        phi_sg_min = angle_1_min
        phi_sg_max = angle_1_max
        phi_lg_min = angle_2_min
        phi_lg_max = angle_2_max
        n1 = ceil(n_iterations / 2)
        n2 = n_iterations
    elif sampling.gear_2 is sampling.sg:
        angle_1_min = - np.pi / sampling.gear_1.p
        angle_1_max = np.pi / sampling.gear_1.p
        angle_2_min = 0.
        angle_2_max = np.pi / sampling.gear_2.p
        phi_sg_min = angle_2_min
        phi_sg_max = angle_2_max
        phi_lg_min = angle_1_min
        phi_lg_max = angle_1_max
        n1 = n_iterations
        n2 = ceil(n_iterations / 2)
    angles_1 = np.linspace(angle_1_min, angle_1_max, num=n1, endpoint=True)
    angles_2 = np.linspace(angle_2_min, angle_2_max, num=n2, endpoint=True)

    # load reference field
    sampling.load_reference_field(sampling.sg, "Vm", "CG", p_deg=p_deg, mesh_size_min=mesh_size, \
                                    mesh_size_max=mesh_size, domain_size=sampling.domain_size, \
                                    analytical_solution=True, write_to_pvd=False)

    # create reference segment
    sampling.mesh_reference_segment(mesh_size, phi_sg_min=phi_sg_min, phi_sg_max=phi_sg_max,
                                    phi_lg_min=phi_lg_min, phi_lg_max=phi_lg_max, write_to_pvd=False)
    sampling.interpolate_to_reference_segment(p_deg=p_deg, interpolate="twice", use_Vm=True)

    write_parameter_file(sampling, target_dir)

    sampling.sample(angles_1, angles_2, data_dir=data_dir, p_deg=p_deg)


def sample_segment(n_iterations, par_number):
    t1 = 1.0
    R1 = 10.
    d_ref = 0.1
    w_ref = 3.
    # create sample directory
    sample_dir = "sample_cylinder_segment_gear"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    # set parameters
    mesh_size = 0.65
    p_deg = 2

    # parameters
    gear_ratio_values = np.array([1.0, 1.4, 2.0, 2.4])
    R_ref_values = np.array([6., 10., 20.])
    scaling_factors = R1 / R_ref_values

    w1_values = w_ref * scaling_factors
    p1_values = list(range(6, 72, 2))
    par_list = list(it.product(gear_ratio_values, w1_values, p1_values))
    par = par_list[par_number]
    gear_ratio, w1, p1 = par
    w2 = w1

    # compute rest of values
    R2 = R1 * gear_ratio
    p2 = p1 * gear_ratio
    if not p2.is_integer():
        exit()
    if int(p2) % 2 != 0:
        exit()
    p2 = int(p2)
    # scale d accordingly
    d = d_ref * w1 / w_ref
    # compute t2
    t2 = R2 / R1 * t1

    # create directory
    gear_ratio_dir = f"gear_ratio_{str(gear_ratio).replace('.', 'p')}"
    width_dir = f"w1_{str(w1).replace('.', 'p')}"
    pole_nb_dir = f"p1_{str(p1).replace('.', 'p')}"
    data_dir = os.path.join(gear_ratio_dir, width_dir, pole_nb_dir)
    target_dir = os.path.join(sample_dir, data_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    gear_1 = SegmentGear(p1, R1 - t1 / 2, w1, t1, np.zeros(3))
    gear_2 = SegmentGear(p2, R2 - t2 / 2, w2, t2, np.array([0., 1., 0.]))

    gear_1.create_magnets(1.0)
    gear_2.create_magnets(1.0)
    D = gear_1.outer_radius + gear_2.outer_radius + d

    # use same main dir for all parameter combination
    # this way all reference fields can be used in each iteration
    sampling = SpurGearsSampling(gear_1, gear_2, D, main_dir=sample_dir)

    # mesh larger gear
    sampling.mesh_gear(sampling.lg, mesh_size=mesh_size,
                       fname=f"gear_{1 if sampling.sg is sampling.gear_1 else 2}_{id(sampling)}",
                        write_to_pvd=False)

    # set angles for sampling
    if sampling.gear_1 is sampling.sg:
        angle_1_min = - np.pi / sampling.gear_1.p
        angle_1_max = np.pi / sampling.gear_1.p
        angle_2_min = - np.pi / sampling.gear_2.p
        angle_2_max = np.pi / sampling.gear_2.p
        phi_sg_min = angle_1_min
        phi_sg_max = angle_1_max
        phi_lg_min = angle_2_min
        phi_lg_max = angle_2_max
        n1 = ceil(n_iterations / 2)
        n2 = n_iterations
    elif sampling.gear_2 is sampling.sg:
        angle_1_min = - np.pi / sampling.gear_1.p
        angle_1_max = np.pi / sampling.gear_1.p
        angle_2_min = - np.pi / sampling.gear_2.p
        angle_2_max = np.pi / sampling.gear_2.p
        phi_sg_min = angle_2_min
        phi_sg_max = angle_2_max
        phi_lg_min = angle_1_min
        phi_lg_max = angle_1_max
        n1 = n_iterations
        n2 = ceil(n_iterations / 2)
    angles_1 = np.linspace(angle_1_min, angle_1_max, num=n1, endpoint=True)
    angles_2 = np.linspace(angle_2_min, angle_2_max, num=n2, endpoint=True)

    # load reference field
    sampling.load_reference_field(sampling.sg, "Vm", "CG", p_deg=p_deg, mesh_size_min=mesh_size,
                                    mesh_size_max=mesh_size, domain_size=sampling.domain_size,
                                    analytical_solution=False, write_to_pvd=False)

    # create reference segment
    sampling.mesh_reference_segment(mesh_size, phi_sg_min=phi_sg_min, phi_sg_max=phi_sg_max,
                                    phi_lg_min=phi_lg_min, phi_lg_max=phi_lg_max, write_to_pvd=False)
    sampling.interpolate_to_reference_segment(p_deg=p_deg, interpolate="twice", use_Vm=True)

    write_parameter_file(sampling, target_dir)

    sampling.sample(angles_1, angles_2, data_dir=data_dir, p_deg=p_deg)


if __name__ == "__main__":
    import sys
    magnet_type = sys.argv[1]
    it_nb = int(sys.argv[2])

    if magnet_type == "ball":
        sample_ball(25, it_nb)
    elif magnet_type == "bar":
        sample_bar(5, it_nb)
    elif magnet_type == "cylinder_segment":
        sample_segment(5, it_nb)
    else:
        raise RuntimeError()
