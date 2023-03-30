import pandas as pd

tau_12 = pd.read_csv("sample_cylinder_segment_gear/gear_ratio_1p0/R1_6p0/p1_12/tau_12.csv", index_col=0).to_numpy()
tau_21 = pd.read_csv("sample_cylinder_segment_gear/gear_ratio_1p0/R1_6p0/p1_12/tau_21.csv", index_col=0).to_numpy()

print(tau_12.min())
print(tau_21.min())

tau_12 = pd.read_csv("sample_bar_gear/gear_ratio_1p0/R1_6p0/p1_12/tau_12.csv", index_col=0).to_numpy()
tau_21 = pd.read_csv("sample_bar_gear/gear_ratio_1p0/R1_6p0/p1_12/tau_21.csv", index_col=0).to_numpy()

print(tau_12.max())
print(tau_21.max())