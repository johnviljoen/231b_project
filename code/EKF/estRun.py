import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

# nominal dynamics - assuming euler integration
def q(x, u, dt, B=0.8, r=0.425):
    # B +- 10%
    # r +- 5%
    # x = {x, y, theta}
    # u = {omega, gamma}
    return np.array([
        x[0] + 5 * r * u[0] * np.cos(x[2]) * dt,
        x[1] + 5 * r * u[0] * np.sin(x[2]) * dt,
        x[2] + 5 * r * u[0] / B * np.tan(u[1]) * dt
    ])

def h(x, B=0.8):
    # B +- 10%
    return np.array([
        [x[0] + 0.5 * B * np.cos(x[2])],
        [x[1] + 0.5 * B * np.sin(x[2])]
    ])

def linmod(x, u, dt, B=0.8, r=0.425):
    # jacobian of f -> linmod basically
    A_mat = np.array([
        [1, 0, - 5 * r * u[0] * np.sin(x[2]) * dt],
        [0, 1, 5 * r * u[0] * np.cos(x[2]) * dt],
        [0, 0, 1]
    ])
    B_mat = np.array([
        [5 * r * np.cos(x[2]) * dt, 0],
        [5 * r * np.sin(x[2]) * dt, 0],
        [5 * r / B * np.tan(u[1]) * dt, 5 * r * u[0] / B * 1/np.cos(u[1])**2 * dt]
    ])
    H_mat = np.array([
        [1, 0, - 0.5 * B * np.sin(x[2])],
        [0, 1, 0.5 * B * np.cos(x[2])]
    ])
    L_mat = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    M_mat = np.array([
        [1., 0.],
        [0., 1.]
    ])
    return A_mat, B_mat, H_mat, L_mat, M_mat


def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s] 
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  x: your current best estimate for the bicycle's x-position
    #  y: your current best estimate for the bicycle's y-position
    #  theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # unpack internal state
    xm = np.array(internalStateIn[:3]) # {x, y, theta}
    um = np.array([pedalSpeed, steeringAngle]) # {omega, gamma}
    z = np.array(measurement) # {x, y}
    Pm = internalStateIn[3] # {3x3}

    # Sigma_vv = np.array([
    #     [ 0.20586454,  0.02667422, -0.00290327],
    #     [ 0.02667422,  0.2355549 ,  0.00036677],
    #     [-0.00290327,  0.00036677,  0.13376413]
    # ])
    # Sigma_ww = np.array([
    #     [2.24469592, 2.03553876],
    #     [2.03553876, 4.55707782]
    # ])

    B = 0.8
    Sigma_vv = np.zeros((3, 3))
    Sigma_vv[0,0] = 25 * um[0] * um[0] * dt * dt * np.cos(xm[2]) * np.cos(xm[2]) * 0.0002 + 0.01
    Sigma_vv[1,1] = 25 * um[0] * um[0] * dt * dt * np.sin(xm[2]) * np.sin(xm[2]) * 0.0002 + 0.01
    Sigma_vv[2,2] = 25 * um[0] * um[0] * dt * dt * 0.0002 / 0.2133 * np.tan(um[1]) * np.tan(um[1]) + 0.01

    Sigma_vv[0,1] = 25 * um[0] * um[0] * dt * dt * np.cos(xm[2]) * np.sin(xm[2]) * 0.0002
    Sigma_vv[1,0] = Sigma_vv[0,1]

    Sigma_vv[0,2] = 25 * um[0] * um[0] * dt * dt * 0.0002 / 0.8 * np.tan(um[1]) * np.cos(xm[2])
    Sigma_vv[2,0] = Sigma_vv[0,2]

    Sigma_vv[1,2] = 25 * um[0] * um[0] * dt * dt * 0.0002 / 0.8 * np.tan(um[1]) * np.sin(xm[2])
    Sigma_vv[2,1] = Sigma_vv[1,2]

    Sigma_ww = np.array([[1.0893397308015538,  0.0],
                        [0.0, 2.9879548591140996]])
    mean_ww = np.array([[0.0028458850141975336],
                        [0.041453958735999615]])
    
    # mean_ww = np.zeros([2,1])

    # Update
    A_km1, _, _, L_km1, _ = linmod(xm, um, dt)
    xp = q(xm, um, dt)
    Pp_k = A_km1 @ Pm @ A_km1.T + L_km1 @ Sigma_vv @ L_km1.T

    # Measurement
    _, _, H_k, _, M_k = linmod(xp, um, dt)
    K_k = Pp_k @ H_k.T @  np.linalg.inv(H_k @ Pp_k @ H_k.T + M_k @ Sigma_ww @ M_k.T)
    if not np.isnan(z).any():
        xm = xp[:, None] + K_k @ (z[:, None] + mean_ww - h(xp))
    else:
        xm = xp[:, None]
    Pm = (np.eye(3) - K_k @ H_k) @ Pp_k @ (np.eye(3) - K_k @ H_k).T + K_k @ Sigma_ww @ K_k.T

    #### OUTPUTS ####
    x, y, theta = xm.flatten()
    internalStateOut = [x, y, theta, Pm]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


