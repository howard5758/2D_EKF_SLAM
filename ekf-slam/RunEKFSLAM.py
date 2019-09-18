import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from EKFSLAM import EKFSLAM


if __name__ == '__main__':

    # This function should be called with one argument:
    #    sys.argv[1]: Pickle file defining problem setup
    if (len(sys.argv) == 2):
        Data = pickle.load(open(sys.argv[1], 'rb'), encoding='latin1')
    else:
        print ("usage: RunEKF.py Data.pickle")
        sys.exit(2)

    # Load data

    # 3 x T array of control inputs, where each column is of the form
    # [t; d; deltaTheta] and corresponds to the control at time t
    U = Data['U']

    # 4 x n array of observations, where each column is of the form
    # [t; id; x; y] and corresponds to a measurement of the relative
    # position of landmark id acquired at time step t
    Z = Data['Z']

    # Motion and measurement covariance matrices
    R = Data['R']
    Q = Data['Q']

    # 3 x 1 array specifying the initial pose
    X0 = Data['X0']

    # 4 x T array specifying the ground-truth pose,
    # where each column is of the form [t; x; y; theta]
    # and indicates the (x,y) position and orientation
    XGT = Data['XGT']

    # 3 x M array specifying the map, where each column is of the form
    # [id; x; y] and indicates the (x,y) position of landmark with id
    MGT = Data['MGT']

    mu0 = X0 #np.array([[-4.0, -4.0, math.pi/2

    # You can also try setting this to a 3x3 matrix of zeros
    Sigma0 = 0.01*np.eye((3))

    # Instantiate the EKFSLAM class
    ekfslam = EKFSLAM(mu0, Sigma0, R, Q, True)

    # Here's where we run the filter
    ekfslam.run(U, Z, XGT, MGT)
