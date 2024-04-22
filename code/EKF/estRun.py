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
    Sigma_vv = internalStateIn[4] # {3x3}
    Sigma_ww = internalStateIn[5] # {2x2}

    Sigma_vv = np.eye(3)
    Sigma_ww = np.eye(2)

    # Sigma_vv = np.array([[ 0.5287062 ,  0.20239834, -0.01566376],
    #    [ 0.20239834,  0.72856973,  0.08223604],
    #    [-0.01566376,  0.08223604,  0.28171361]])
    # Sigma_ww = np.array([[5.00751546, 2.59393855],
    #    [2.59393855, 7.45516744]])
    
    # Sigma_vv = np.array([[0.19199799, 0.03592953, 0.00370974],
    #    [0.03592953, 0.24384159, 0.01312044],
    #    [0.00370974, 0.01312044, 0.04701103]])
    # Sigma_ww = np.array([[1.64948702, 2.05031815],
    #    [2.05031815, 4.57723283]])
    
    # Sigma_vv = np.array([[0.05395144, 0.01174522, 0.00102412],
    #    [0.01174522, 0.07927587, 0.01053843],
    #    [0.00102412, 0.01053843, 0.0095018 ]])
    # Sigma_ww = np.array([[1.42896293, 1.99171226],
    #    [1.99171226, 4.80647754]])
    
    Sigma_vv = np.array([[ 4.03852023e-01, -4.53941322e-03, -1.78433646e-03],
       [-4.53941322e-03,  3.96567534e-01,  3.25160704e-04],
       [-1.78433646e-03,  3.25160704e-04,  3.17160995e-02]])
    Sigma_ww = np.array([[2.45192888, 2.15958191],
       [2.15958191, 4.71962838]])

    # used for manually gathering statistics through the example runs on the Sigma_vv and Sigma_ww
    cumulative_vv = internalStateIn[6]
    cumulative_ww = internalStateIn[7] 
    count_vv = internalStateIn[8]
    count_ww = internalStateIn[9]

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
        cumulative_ww += np.outer(residual, residual)
        count_ww += 1
    else:
        xm = xp[:, None]
        residual = np.zeros_like(z)
    Pm = (np.eye(3) - K_k @ H_k) @ Pp_k @ (np.eye(3) - K_k @ H_k).T + K_k @ Sigma_ww @ K_k.T

    # Update cumulative sums for process noise covariance
    process_residual = xp[:, None] - q(xm, um, dt)
    cumulative_vv += np.outer(process_residual, process_residual)
    count_vv += 1

    # outdated, seems the covariances fairly constant, no need for online adaptation
    # Update covariances based on EWMA
    # alpha_vv = 0.01 # learning rate vv
    # alpha_ww = 0.01 # learning rate ww
    # Sigma_vv = (1 - alpha_vv) * Sigma_vv + alpha_vv * np.outer(xp[:, None] - q(xm, um, dt), xp[:, None] - q(xm, um, dt))
    # Sigma_ww = (1 - alpha_ww) * Sigma_ww + alpha_ww * np.outer(residual, residual)

    #### OUTPUTS ####
    x, y, theta = xm.flatten()
    internalStateOut = [x, y, theta, Pm, Sigma_vv, Sigma_ww, cumulative_vv, cumulative_ww, count_vv, count_ww]

    if time == 99.9:
        print('fin')

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


