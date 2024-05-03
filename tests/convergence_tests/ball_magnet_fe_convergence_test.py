from source.fe import compute_magnetic_potential
from source.magnet_classes import BallMagnet
from source.tools.fenics_tools import compute_current_potential

import logging
logging.basicConfig(level=logging.INFO)
import numpy as np


def mesh_size_convergence_test(magnet: BallMagnet, mesh_size_values, p_deg):
    for mesh_size in mesh_size_values[::-1]:
        R_domain = 3. + mesh_size
        R_inf = 80 * magnet.size
        assert R_inf >= magnet.size
        # compute solution
        logging.info("Computing magnetic potential ...")
        Vm_num = compute_magnetic_potential(magnet, R_domain=R_domain, R_inf=R_inf,
                                        mesh_size_magnet=mesh_size, mesh_size_domain_min=mesh_size,
                                        mesh_size_domain_max=mesh_size, mesh_size_space=None, p_deg=p_deg,
                                        cylinder_mesh_size_field=False, fname="Vm", autorefine=False,
                                        write_to_pvd=False)
        H_num = compute_current_potential(Vm_num, project=True, cell_type="DG")

        # create some points
        n = 10
        r_vals = np.linspace(1.1, R_domain, num=n)[:, None, None]
        theta_vals = np.linspace(0., np.pi, num=n + 1, endpoint=False)[1:][None, :, None]
        phi_vals = np.linspace(0., 2 * np.pi, num=2 * n, endpoint=False)[None, None]

        points_x = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
        points_y = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
        points_z = r_vals * np.cos(theta_vals) * np.ones_like(phi_vals)
        points = np.vstack((points_x.flatten(), points_y.flatten(), points_z.flatten())).T

        # evaluate solution at some points
        H_ana_vals = np.empty_like(points)
        H_num_vals = np.empty_like(points)
        for k, p in enumerate(points):
            H_ana_vals[k] = magnet.H(p)
            H_num_vals[k] = H_num(*p)
        error_H = np.linalg.norm(H_ana_vals - H_num_vals, axis=1).sum() / \
            np.linalg.norm(H_ana_vals, axis=1).sum()

        n_dofs = Vm_num.function_space().dim()
        with open(f"fe_error_H_p_deg_{p_deg}.csv", "a+") as f:
            f.write(f"{mesh_size} {n_dofs} {error_H}\n")

def R_inf_test(magnet: BallMagnet, R_inf_values, mesh_size, p_deg):
    assert isinstance(magnet, BallMagnet)
    R_domain = 2 * magnet.R
    for R_inf in R_inf_values:
        assert R_inf >= magnet.size
        # compute solution
        logging.info("Computing magnetic potential ...")
        Vm_num = compute_magnetic_potential(magnet, R_domain=R_domain, R_inf=R_inf,
                                            mesh_size_magnet=mesh_size, mesh_size_domain_min=mesh_size,
                                            mesh_size_domain_max=mesh_size, mesh_size_space=None,
                                            p_deg=p_deg, cylinder_mesh_size_field=False,
                                            autorefine=False, check_input=False, fname="Vm", write_to_pvd=False)
        H_num = compute_current_potential(Vm_num, project=True, cell_type="DG")

        # create some points
        n = 10
        r_vals = np.linspace(1.1, np.mean((R_inf_values.min(), magnet.R)), num=n)[:, None, None]
        theta_vals = np.linspace(0., np.pi, num=n + 1, endpoint=False)[1:][None, :, None]
        phi_vals = np.linspace(0., 2 * np.pi, num=2 * n, endpoint=False)[None, None]

        points_x = r_vals * np.sin(theta_vals) * np.cos(phi_vals)
        points_y = r_vals * np.sin(theta_vals) * np.sin(phi_vals)
        points_z = r_vals * np.cos(theta_vals) * np.ones_like(phi_vals)
        points = np.vstack((points_x.flatten(), points_y.flatten(), points_z.flatten())).T

        # evaluate solution at some points
        H_ana_vals = np.empty_like(points)
        H_num_vals = np.empty_like(points)
        for k, p in enumerate(points):
            H_ana_vals[k] = magnet.H(p)
            H_num_vals[k] = H_num(*p)
        error_H = np.linalg.norm(H_ana_vals - H_num_vals, axis=1).sum() / \
            np.linalg.norm(H_ana_vals, axis=1).sum()

        n_dofs = Vm_num.function_space().dim()
        with open(f"fe_R_inf_error_H_p_deg_{p_deg}.csv", "a+") as f:
            f.write(f"{mesh_size} {n_dofs} {R_inf} {error_H}\n")

if __name__ == "__main__":
    ball_magnet = BallMagnet(1., 1., np.zeros(3), np.eye(3))
    R_inf_values = ball_magnet.R + np.hstack((np.linspace(3., 20., num=8, endpoint=False), np.linspace(20., 100., num=5)))
    mesh_size_values = np.geomspace(1e-1, 1.0, num=6)
    for p_deg in (1, 2):
        R_inf_test(ball_magnet, R_inf_values, mesh_size=0.1, p_deg=p_deg)
        mesh_size_convergence_test(ball_magnet, mesh_size_values=mesh_size_values, p_deg=p_deg)
