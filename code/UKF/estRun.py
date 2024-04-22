import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)
from scipy.linalg import sqrtm

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

def get_sig(xm, P, n=3):
    sp_x = []
    sqrt_nP = sqrtm(n*P)
    for i in range(n):
        sp_x.append(xm + sqrt_nP[:, i])
    for i in range(n):
        sp_x.append(xm - sqrt_nP[:, i])
    return np.vstack(sp_x).T

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
    Sigma_vv = internalStateIn[4] # {3x3}
    Sigma_ww = internalStateIn[5] # {2x2}

    # Update
    A_km1, _, _, L_km1, _ = linmod(xm, um, dt)
    xp = q(xm, um, dt)
    Pp_k = A_km1 @ Pm @ A_km1.T + L_km1 @ Sigma_vv @ L_km1.T

    # Measurement
    _, _, H_k, _, M_k = linmod(xp, um, dt)
    K_k = Pp_k @ H_k.T @  np.linalg.inv(H_k @ Pp_k @ H_k.T + M_k @ Sigma_ww @ M_k.T)
    if not np.isnan(z).any():
        xm = xp[:, None] + K_k @ (z[:, None] - h(xp))
        residual = z - h(xp).flatten()
    else:
        xm = xp[:, None]
        residual = np.zeros_like(z)
    Pm = (np.eye(3) - K_k @ H_k) @ Pp_k @ (np.eye(3) - K_k @ H_k).T + K_k @ Sigma_ww @ K_k.T

    # Update covariances based on EWMA
    alpha_vv = 0.01 # learning rate vv
    alpha_ww = 0.01 # learning rate ww
    Sigma_vv = (1 - alpha_vv) * Sigma_vv + alpha_vv * np.outer(xp[:, None] - q(xm, um, dt), xp[:, None] - q(xm, um, dt))
    Sigma_ww = (1 - alpha_ww) * Sigma_ww + alpha_ww * np.outer(residual, residual)

    #### OUTPUTS ####
    x, y, theta = xm.flatten()
    internalStateOut = [x, y, theta, Pm, Sigma_vv, Sigma_ww]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


