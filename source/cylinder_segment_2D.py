import numpy as np
import scipy as sp


"""
In this file the 2d field of a hollow cylinder with p pole pairs
is implemented. The field solution is taken from this article
https://iopscience.iop.org/article/10.1088/0022-3727/33/1/305.
"""

def integrand_force(theta_edge, phi_1, D, p, Ri, Ro, nb_terms):
    func = lambda r: (
        - Bz_cylinder_segment_2D(r, theta_edge, phi_1, D, p, Ri, Ro, nb_terms), \
        By_cylinder_segment_2D(r, theta_edge, phi_1, D, p, Ri, Ro, nb_terms)
        )
    return func

def integrand_torque(theta_edge, phi_1, D, p, Ri, Ro, nb_terms):
    func = lambda r: r * (np.cos(theta_edge) * By_cylinder_segment_2D(r, theta_edge, phi_1, D, p, Ri, Ro, nb_terms) + \
        np.sin(theta_edge) * Bz_cylinder_segment_2D(r, theta_edge, phi_1, D, p, Ri, Ro, nb_terms))
    return func

def integrate(integrand_func, theta_2_edge, phi_1, D, p, Ri1, Ro1, Ri2, Ro2, nb_terms):
    assert callable(integrand_func)
    integrand = integrand_func(theta_2_edge, phi_1, D, p, Ri1, Ro1, nb_terms)
    if isinstance(integrand(0), (tuple, list)):
        result = []
        for i in range(len(integrand(0))):
            integrand_comp = lambda r: integrand(r)[i]
            result_comp, _ = sp.integrate.quad(integrand_comp, Ri2, Ro2)
            result.append(result_comp)
        return np.array(result)
    result, _ = sp.integrate.quad(integrand, Ri2, Ro2)
    return result

def force_torque_cylinder_segment_2D(w2, t1, t2, p1, p2, Rm1, Rm2, phi_1, phi_2, D, nb_terms=40):
    """
    Compute torque from gear 1 on gear 2.
    """
    # sum over all magnets
    f = np.zeros(2)
    tau = 0.
    Ri1 = Rm1 - t1 / 2
    Ro1 = Rm1 + t1 / 2
    Ri2 = Rm2 - t2 / 2
    Ro2 = Rm2 + t2 / 2
    for n2 in range(p2):
        theta_2_edge = phi_2 + np.pi / p2 * (1 + 2 * n2)

        f += (-1) ** (n2 + 1) * integrate(integrand_force, theta_2_edge, phi_1, D, p1, Ri1, Ro1, Ri2, Ro2, nb_terms)
        tau += (-1) ** (n2 + 1) * integrate(integrand_torque, theta_2_edge, phi_1, D, p1, Ri1, Ro1, Ri2, Ro2, nb_terms)

    return 2 * w2 * f, 2 * w2 * tau

def By_cylinder_segment_2D(r, theta, phi_1, D, p, Ri, Ro, nb_terms):
    rp = np.sqrt(r ** 2 + 2 * r * D * np.cos(theta) + D ** 2)
    thetap = np.arctan2(r * np.sin(theta), r * np.cos(theta) + D)
    By = Br_cylinder_segment_2D_eigen(rp, thetap - phi_1, p, Ri, Ro, nb_terms) * np.cos(thetap) - \
        Btheta_cylinder_segment_2D_eigen(rp, thetap - phi_1, p, Ri, Ro, nb_terms) * np.sin(thetap)
    return By

def Bz_cylinder_segment_2D(r, theta, phi_1, D, p, Ri, Ro, nb_terms):
    rp = np.sqrt(r ** 2 + 2 * r * D * np.cos(theta) + D ** 2)
    thetap = np.arctan2(r * np.sin(theta), r * np.cos(theta) + D)
    Bz = Br_cylinder_segment_2D_eigen(rp, thetap - phi_1, p, Ri, Ro, nb_terms) * np.sin(thetap) + \
        Btheta_cylinder_segment_2D_eigen(rp, thetap - phi_1, p, Ri, Ro, nb_terms) * np.cos(thetap)
    return Bz

def Br_cylinder_segment_2D_eigen(rp, thetap, p, Ri, Ro, nb_terms):
    k = np.arange(nb_terms) + 1
    Mk = M(k, p)
    return np.sum(k * rp ** (- (k + 1)) * U2(Mk, Ri, Ro, k) * np.cos(k * thetap))

def Btheta_cylinder_segment_2D_eigen(rp, thetap, p, Ri, Ro, nb_terms):
    k = np.arange(nb_terms) + 1
    Mk = M(k, p)
    return np.sum(k * rp ** (- (k + 1)) * U2(Mk, Ri, Ro, k) * np.sin(k * thetap))

def U2(M, Ri, Ro, k):
    return - H2(M, Ri, Ro, k) / 4

def H2(M, Ri, Ro, k):
    return Ro ** (k + 1) * (
        2 / Ro * M / k / (k + 1) * (Ro - (Ri / Ro) ** k * Ri) - \
        2 * M / k + 2 * M / k * (Ri / Ro) ** (k + 1)
    )

def M(k, p):
    assert isinstance(p, int)
    k = np.atleast_1d(k)[np.newaxis]
    m = np.arange(p) + 1
    m = m[:, np.newaxis]
    return np.sum(1 / k / np.pi * (-1) ** m * (np.sin((2 * m - 1) / p * k * np.pi) - np.sin((2 * m + 1) / p * k * np.pi)), axis=0)
