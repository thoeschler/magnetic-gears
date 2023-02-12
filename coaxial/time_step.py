import numpy as np
from scipy.integrate import RK45


def rhs(tau_1, tau_2, theta_1, theta_2):
    """Right hand side for integration of law of motion
    of a single gear.

    Args:
        tau_1 (callable): Torque function of first gear.
        tau_2 (callable): Torque function of second gear.
        theta_1 (float): Moment of intertia of first gear.
        theta_2 (float): Moment of intertia of seocnd gear.

    Returns:
        callable: The right hand side. Two input parameters
                  are time and the y (the solution array).
    """
    # using RK45: convert two second order scalar equation into first
    # order vector equation of dimension 4
    # the array y contains: phi_1, phi_2, phip_1, phip_2

    # the functions tau_1 and tau_2 are expected to have four input arguments
    return lambda t, y: np.array([y[2], y[3], tau_1(*y) / theta_1, tau_2(*y) / theta_2])


def get_integrator(tau_1, tau_2, theta_1, theta_2, phi_1_init=0., phi_2_init=0., phip_1_init=0., \
                    phip_2_init=0., t_init=0., t_end=1.0):
    """Get integrator for equation of motion of a single gear.

    Args:
        tau_1 (): _description_
        tau_2 (_type_): _description_
        theta_1 (_type_): _description_
        theta_2 (_type_): _description_
        phi_1_init (_type_, optional): _description_. Defaults to 0..
        phi_2_init (_type_, optional): _description_. Defaults to 0..
        phip_1_init (_type_, optional): _description_. Defaults to 0..
        t_init (_type_, optional): _description_. Defaults to 0..
        t_end (float, optional): _description_. Defaults to 1.0.

    Returns:
        scipy.integrate.RK45: The scipy integrator.
    """

    y0 = np.array([phi_1_init, phi_2_init, phip_1_init, phip_2_init])  # initial values
    integrator = RK45(rhs(tau_1, tau_2, theta_1, theta_2), t0=0, y0=y0, t_bound=t_end)

    return integrator

