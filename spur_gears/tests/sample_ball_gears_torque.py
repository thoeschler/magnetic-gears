import numpy as np
from source.magnetic_gear_classes import MagneticBallGear
from spur_gears.spur_gears_problem import SpurGearsProblem
from tests.convergence_tests.ball_magnets_force_torque import compute_force_ana
from spur_gears.tests.convergence_tests.ball_gear_convergence_test_base import compute_torque_ana

# create two gears
n = 10
R = 5
r = 1.
gear_1 = MagneticBallGear(n, R, r, np.zeros(3))
gear_2 = MagneticBallGear(n, R, r, np.array([0., 1., 0.]))
gear_1.create_magnets(1.0)
gear_2.create_magnets(1.0)
D = gear_1.outer_radius + gear_2.outer_radius + 0.1
spur_gears = SpurGearsProblem(gear_1, gear_2, D)
spur_gears.align_gears()

offset = 0.#np.pi / n
spur_gears.gear_2.update_parameters(offset)
num = 25
angles_1 = np.linspace(0., 4 * np.pi / n, num=num)
angles_2 = np.linspace(0., 4 * np.pi / n, num=num)
d_angle = angles_1[1] - angles_1[0]

tau_12_values = np.zeros((num, num))
tau_21_values = np.zeros((num, num))

for i in range(num):
    for j in range(num):
        f_12 = np.zeros(3)
        tau_12 = np.zeros(3)
        tau_21 = np.zeros(3)
        for mag_1 in gear_1.magnets:
            for mag_2 in gear_2.magnets:
                # force
                f_21_mag = compute_force_ana(mag_2, mag_1)
                f_12_mag = compute_force_ana(mag_1, mag_2)
                assert np.allclose(f_12_mag, -f_21_mag)
                f_12 += f_12_mag

                # torque
                tau_12 += compute_torque_ana(mag_1, mag_2, f_12_mag, gear_2.x_M)
                tau_21 += compute_torque_ana(mag_1, mag_2, f_21_mag, gear_2.x_M)
        tau_12_values[i, j] = tau_12[0]
        tau_21_values[i, j] = tau_21[0]
        gear_2.update_parameters(d_angle)
    gear_2.update_parameters(-gear_2.angle + angles_2[0])
    gear_1.update_parameters(d_angle)

print(tau_12_values)
print(tau_21_values)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
idcs = np.arange(num)
ax.plot(angles_1, tau_12_values[idcs, idcs])
print(np.argmax(tau_12_values, axis=0))
plt.show()