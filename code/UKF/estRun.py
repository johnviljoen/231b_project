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
        x[0] + 1 * r * u[0] * np.cos(x[2]) * dt,
        x[1] + 1 * r * u[0] * np.sin(x[2]) * dt,
        x[2] + 1 * r * u[0] / B * np.tan(u[1]) * dt
    ])

def h(x, B=0.8):
    # B +- 10%
    return np.array([
        [x[0] + 0.5 * B * np.cos(x[2])],
        [x[1] + 0.5 * B * np.sin(x[2])]
    ])

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3.real

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
    Pm = internalStateIn[3]
    cumulative_vv = internalStateIn[4]
    cumulative_ww = internalStateIn[5]
    count_vv = internalStateIn[6]
    count_ww = internalStateIn[7]

    # covariances found from statistical analysis - hardcoded here
    Sigma_vv = np.array([
        [ 0.20586454,  0.02667422, -0.00290327],
        [ 0.02667422,  0.2355549 ,  0.00036677],
        [-0.00290327,  0.00036677,  0.13376413]
    ])
    Sigma_ww = np.array([
        [2.24469592, 2.03553876],
        [2.03553876, 4.55707782]
    ])

    # Sigma_vv = np.array([[ 0.18197149, -0.01098326,  0.00094567],
    #    [-0.01098326,  0.17309458, -0.00113221],
    #    [ 0.00094567, -0.00113221,  0.09461396]])
    
    # Sigma_ww = np.array([[10.82315937,  5.79229707],
    #    [ 5.79229707, 17.56257188]])

    ###### Prior update ######
    sm = get_sig(xm, Pm) # get 2n sigma points
    sp = q(sm, um, dt) # transform for prior sigma points

    # compute prior statistics
    xp_hat_k = np.mean(sp, axis=1, keepdims=True)
    Pp_k = (sp - xp_hat_k) @ (sp - xp_hat_k).T / sp.shape[1] + Sigma_vv

    ###### Posteriori update ######
    sp = q(get_sig(xp_hat_k.flatten(), Pp_k), um, dt) # recompute 2n sigma points and transform

    sz_k = h(sp)[:,0,:] # compute sigma points for the measurements
    z_hat_k = np.mean(sz_k, axis=1, keepdims=True) # expected measurement
    Pzz_k = (sz_k - z_hat_k) @ (sz_k - z_hat_k).T / sz_k.shape[1] + Sigma_ww # associated covariance

    # cross covariance
    Pxz_k = (sp - xp_hat_k) @ (sz_k - z_hat_k).T / sz_k.shape[1]

    # apply the kalman filter gain
    K_k = Pxz_k @ np.linalg.inv(Pzz_k)
    if not np.isnan(z).any():
        xm_k = xp_hat_k + K_k @ (z[:, None] - z_hat_k)
        # Update cumulative covariances and counts for measurement noise
        cumulative_ww += (z[:, None] - z_hat_k) @ (z[:, None] - z_hat_k).T
        count_ww += 1
    else:
        xm_k = xp_hat_k

    # Update cumulative covariances and counts for process noise
    cumulative_vv += (xp_hat_k - q(xm_k, um, dt)) @ (xp_hat_k - q(xm_k, um, dt)).T
    count_vv += 1

    Pm_k = Pp_k - K_k @ Pzz_k @ K_k.T

    #### OUTPUTS ####
    x, y, theta = xm_k.flatten()
    internalStateOut = [x, y, theta, Pm_k, cumulative_vv, cumulative_ww, count_vv, count_ww]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


