
# This is the main function, which will initilize your estimator, and run it using data loaded from a text file. 
#

import numpy as np
from PF.estRun import estRun
from PF.estInitialize import estInitialize
from tqdm import tqdm

Err_x, Err_y, Err_theta = [], [], []

for experimentalRun in tqdm(range(100)):

    experimentalData = np.genfromtxt ('data/run_{0:03d}.csv'.format(experimentalRun), delimiter=',')

    #===============================================================================
    # Here, we run your estimator's initialization
    #===============================================================================
    internalState, studentNames, estimatorType = estInitialize()
    numDataPoints = experimentalData.shape[0]

    #Here we will store the estimated position and orientation, for later plotting:
    estimatedPosition_x = np.zeros([numDataPoints,])
    estimatedPosition_y = np.zeros([numDataPoints,])
    estimatedAngle = np.zeros([numDataPoints,])

    # print('Running the system')
    dt = experimentalData[1,0] - experimentalData[0,0]
    for k in range(numDataPoints):
        t = experimentalData[k,0]
        gamma = experimentalData[k,1]
        omega = experimentalData[k,2]
        measx = experimentalData[k,3]
        measy = experimentalData[k,4]
        
        #run the estimator:
        x, y, theta, internalState = estRun(t, dt, internalState, gamma, omega, (measx, measy))

        #keep track:
        estimatedPosition_x[k] = x
        estimatedPosition_y[k] = y
        estimatedAngle[k] = theta

    #make sure the angle is in [-pi,pi]
    estimatedAngle = np.mod(estimatedAngle+np.pi,2*np.pi)-np.pi

    posErr_x = estimatedPosition_x - experimentalData[:,5]
    posErr_y = estimatedPosition_y - experimentalData[:,6]
    angErr   = np.mod(estimatedAngle - experimentalData[:,7]+np.pi,2*np.pi)-np.pi

    Err_x.append(posErr_x[-1])
    Err_y.append(posErr_y[-1])
    Err_theta.append(angErr[-1])

print('Final error: ')
print('   pos x =',np.mean(Err_x),'m')
print('   pos y =',np.mean(Err_y),'m')
print('   angle =',np.mean(Err_theta),'rad')





