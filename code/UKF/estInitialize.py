import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your estRun() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    # 
    # The third return variable must be a string with the estimator type

    #we make the internal state a list, with the first three elements the position
    # x, y; the angle theta; and our favorite color.
    x = 0
    y = 0
    theta = np.pi/4
    Pm = np.diag([10, 10, 0.5]) # identity for now
    Sigma_vv = np.eye(3)
    Sigma_ww = np.eye(2)
    cumulative_vv = np.zeros_like(Sigma_vv)
    cumulative_ww = np.zeros_like(Sigma_ww)
    count_vv = 0
    count_ww = 0

    # note that there is *absolutely no prescribed format* for this internal state.
    # You can put in it whatever you like. Probably, you'll want to keep the position
    # and angle, and probably you'll remove the color.
    internalState = [
        x,
        y,
        theta, 
        Pm,
        cumulative_vv,
        cumulative_ww,
        count_vv,
        count_ww
    ]

    # replace these names with yours. Delete the second name if you are working alone.
    studentNames = ['John Viljoen',
                    'Yutaka Shimizu',
                    'Constance Angelopolous']
    
    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'UKF'  
    
    return internalState, studentNames, estimatorType

