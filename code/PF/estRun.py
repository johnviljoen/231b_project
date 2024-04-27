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

    # get the last time step state x1, y1, theta
    x1 = internalStateIn[0]
    y1 = internalStateIn[1]
    theta = internalStateIn[2]

    myColor = internalStateIn[3]

    # get the particle number
    N = internalStateIn[4]

    # get B and r
    B = internalStateIn[5]
    r = internalStateIn[6]

    # get the measurements of x and y
    x_meas = measurement[0]
    y_meas = measurement[1]

    # calculate the linear velocity
    v = 5 * r * pedalSpeed

    # prior update of x1, y1, theta
    x1p = x1 + v * np.cos(theta) * dt + np.random.normal(loc = 0, scale = np.sqrt(0.001), size = [N,1])
    y1p = y1 + v * np.sin(theta) * dt + np.random.normal(loc = 0, scale =np.sqrt(0.003), size = [N,1])
    thetap = theta + v / B * np.tan(steeringAngle) * dt + np.random.normal(loc = 0, scale = np.sqrt(0.001), size = [N,1])

    # get the priors  of bicycle position x, y from x1, y2
    xp = x1p + 0.5 * B * np.cos(thetap)
    yp = y1p + 0.5 * B * np.sin(thetap)

    if not (np.isnan(measurement[0]) and np.isnan(measurement[1])):
        # if we have a valid measurement
        
        xy = np.concatenate([xp, yp], axis=1)

        if np.isnan(measurement[0]):
            # in case only the measurement of y is valid
            beta = sp.stats.norm.pdf(xy[:, 1], loc=y_meas, scale=np.sqrt(2.98))
        elif np.isnan(measurement[1]):
            # in case only the measurement of x is valid
            beta = sp.stats.norm.pdf(xy[:, 0], loc=x_meas, scale=np.sqrt(1.09))
        else:
            # both measurements of x, y are valid
            beta = sp.stats.multivariate_normal.pdf(
                xy, mean = [x_meas,y_meas],
                cov = np.array([[1.08933973, 1.53329122],[1.53329122, 2.98795486]])
            )

        # Particles with too low probability may have NaN probability values
        beta = np.where(np.isnan(beta), np.zeros_like(beta), beta)
        beta = beta / np.sum(beta)
        particle_id = np.random.choice(N, size=N, replace=True, p=beta)
        x1m = x1p[particle_id]
        y1m = y1p[particle_id]
        thetam = thetap[particle_id]
        xm = xp[particle_id]
        ym = yp[particle_id]
    else:
        # If there are no valid measurements,we just use the prior estimate
        x1m = x1p
        y1m = y1p
        thetam = thetap
        xm = xp
        ym = yp

    # #we're unreliable about our favourite colour: 
    # if myColor == 'green':
    #     myColor = 'red'
    # else:
    #     myColor = 'green'


    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalStateOut = [x1m,
                     y1m,
                     thetam,
                     myColor,
                     N,
                     B,
                     r
                     ]
    xm = np.mean(xm)
    ym = np.mean(ym)
    thetam = np.mean(thetam)

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return xm, ym, thetam, internalStateOut 


