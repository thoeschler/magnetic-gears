from coaxial.time_step import get_integrator
from source.tools.math_tools import get_interpolater, interpolate
import pandas as pd
import numpy as np
import os

def pad_data(phi_1, phi_2, torque_1, torque_2):
    # resort data
    id_1 = np.argsort(phi_1)
    id_2 = np.argsort(phi_2)

    # sort angle values (have to be in ascending order)
    phi_1.sort()
    phi_2.sort()

    # pad angle values
    phi_1 = np.append(phi_1, phi_1[-1] + (phi_1[1] - phi_1[0]))
    phi_2 = np.append(phi_2, phi_2[-1] + (phi_2[1] - phi_2[0]))
    
    # sort torque values accordingly
    torque_1 = torque_1[:, id_1][id_2]
    torque_2 = torque_2[:, id_1][id_2]

    # pad torque values
    torque_1 = np.vstack((torque_1, torque_1[0]))
    torque_1 = np.hstack((torque_1, torque_1[:, 0][:, None]))
    torque_2 = np.vstack((torque_2, torque_2[0]))
    torque_2 = np.hstack((torque_2, torque_2[:, 0][:, None]))

    return phi_1, phi_2, torque_1, torque_2


def get_phi_torque(fname):
    # load torque data
    data = pd.read_csv(fname, index_col=None, header=None, delimiter="\s", engine="python").to_numpy()

    n_rows = data.shape[0]
    assert np.isclose(np.sqrt(n_rows), int(np.sqrt(n_rows)))
    n = int(np.sqrt(n_rows))
    phi_1 = data[:n, 0]
    phi_2 = data[::n, 1]
    torque_1 = data[:, 2].reshape(n, n)
    torque_2 = data[:, 3].reshape(n, n)
    assert len(torque_1) == n
    assert len(torque_2) == n

    phi_1, phi_2, torque_1, torque_2 = pad_data(phi_1, phi_2, torque_1, torque_2)

    return phi_1, phi_2, torque_1, torque_2


def main():
    fname = os.path.dirname(__file__) + "/torque_ball_gears.csv"
    print(fname)

    phi_1_vals, phi_2_vals, torque_1_vals, torque_2_vals = get_phi_torque(fname)

    # get torque interpolators
    torque_1_interp = get_interpolater(torque_1_vals, phi_1_vals, phi_2_vals)
    torque_2_interp = get_interpolater(torque_2_vals, phi_1_vals, phi_2_vals)

    tau_in = -1.
    tau_out = 0.

    # torque functions
    tau1 = lambda phi1, phi2, phip1, phip2: interpolate(torque_1_interp, (phi1, phi2)) + tau_in
    tau2 = lambda phi1, phi2, phip1, phip2: interpolate(torque_2_interp, (phi1, phi2)) + tau_out

    # initial values
    phi_1 = 0.
    phi_2 = 0.
    phip_1 = 0.
    phip_2 = 0.

    # final time
    t_end = 10.0

    # get integrator
    integrator = get_integrator(
        tau_1=tau1, tau_2=tau2, theta_1=1., theta_2=1., \
            phi_1_init=phi_1, phi_2_init=phi_2, phip_1_init=phip_1, phip_2_init=phip_2, \
                t_end=t_end
    )
    t = 0.
    t_end = 1.0
    i = 0
    t_values = []
    y_values = []

    while integrator.t < integrator.t_bound:
        i +=1
        print(i, tau1(*integrator.y), tau2(*integrator.y))
        t_values += [integrator.t]
        y_values += [integrator.y]
        # perform step
        integrator.step()

    return t_values, y_values


if __name__ == "__main__":
    t_values, y_values = main()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    phi_1_values = [y[0] for y in y_values]
    phi_2_values = [y[1] for y in y_values]
    ax.plot(t_values, phi_1_values)
    ax.plot(t_values, phi_2_values)
    plt.show()