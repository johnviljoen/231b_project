import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

data_num = 0
experimentalData = np.genfromtxt ('data/run_{0:03d}.csv'.format(data_num), delimiter=',')
data_length = len(experimentalData)
print("data_length: ", data_length)

B = 0.8
true_x = experimentalData[-1, 5]
true_y = experimentalData[-1, 6]
true_theta = experimentalData[-1, 7]
nominal_obs_x = true_x + 0.5 * B * np.cos(true_theta)
nominal_obs_y = true_y + 0.5 * B * np.sin(true_theta)

wxs = []
wys = []
for i in range(data_length):
    data = experimentalData[i, :]
    is_obs_x_nan = np.isnan(data[3])
    is_obs_y_nan = np.isnan(data[4])

    if is_obs_x_nan or is_obs_y_nan:
        continue

    obs_x = data[3]
    obs_y = data[4]
    wx = obs_x - nominal_obs_x # measurement noise for x
    wy = obs_y - nominal_obs_y # measurement noise for y
    wxs.append(wx)
    wys.append(wy)

print("nominal_obs_x: ", nominal_obs_x)
print("nominal_obs_y: ", nominal_obs_y)
print("true_x: ", true_x)
print("true_y: ", true_y)
print("true_theta: ", true_theta)
print("mean_wx: ", np.mean(wxs))
print("cov_wx: ", np.cov(wxs))
print("mean_wy: ", np.mean(wys))
print("cov_wy: ", np.cov(wys))

