import dolfin as dlf
import numpy as np
from source.fe import compute_magnetic_potential, magnet_mesh
from source.magnet_classes import BallMagnet, BarMagnet, PermanentMagnet
from source.tools.fenics_tools import compute_current_potential


def main(magnet: PermanentMagnet, ms):
    #R_domain_values = np.linspace(1., 20., num=3) * magnet.R
    R_domain = magnet.size
    #for R_domain in R_domain_values:
    R_inf_values = R_domain + np.linspace(0., 300., num=10)
    for i, R_inf in enumerate(R_inf_values):
        assert R_inf >= R_domain
        # compute solution
        Vm_num = compute_magnetic_potential(magnet, R_domain=R_domain, R_inf=R_inf, \
                                        mesh_size_magnet=ms, mesh_size_domain_min=ms, \
                                            mesh_size_domain_max=4 * ms, mesh_size_max=20., p_deg=2,\
                                                fname="Vm", write_to_pvd=True)
        print("DONE!.")
        # compute current potential
        dlf.File("Vm_num.pvd") << Vm_num

        # compute analytical solution
        Vm_ana = magnet.Vm_as_expression(degree=Vm_num.ufl_element().degree() + 2)
 
        # compute error
        if magnet.type == "Ball":
            error_Vm = dlf.errornorm(Vm_ana, Vm_num, norm_type="l2", degree_rise=2)
            norm_Vm = dlf.norm(Vm_ana, norm_type="L2", mesh=Vm_num.function_space().mesh())
            error_Vm /= norm_Vm
            """error_H = dlf.errornorm(H_ana, H_num, norm_type="l2", degree_rise=1)
            norm_H = dlf.norm(H_ana, norm_type="L2", mesh=H_num.function_space().mesh())
            error_H /= norm_H"""
        elif magnet.type == "Bar":
            if i > 0:
                error_Vm = dlf.errornorm(old_Vm, Vm_num, norm_type="l2", mesh=Vm_num.function_space().mesh())
                norm_Vm = dlf.norm(old_Vm, norm_type="L2")
                error_Vm /= norm_Vm
            else:
                error_Vm = 0
            old_Vm = Vm_num
        else:
            raise RuntimeError()
        print(f"error_Vm = {error_Vm}")
        with open(f"fe_test_{magnet.type}_error_Vm.csv", "a+") as f:
            f.write(f"{ms}{magnet.size} {R_domain} {R_inf} {error_Vm}\n")


if __name__ == "__main__":
    ball_magnet = BallMagnet(1., 1., np.zeros(3), np.eye(3))
    bar_magnet = BarMagnet(1., 1., 1., 1., np.zeros(3), np.eye(3))
    for magnet in (bar_magnet,):
        main(magnet, 0.1)
