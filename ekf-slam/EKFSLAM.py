import sys
import time
import numpy as np
from numpy.linalg import inv
import matplotlib
import matplotlib.pyplot as plt
from Visualization import Visualization

class EKFSLAM(object):
    # Construct an EKF instance with the following set of variables
    #    mu:                 The initial mean vector
    #    Sigma:              The initial covariance matrix
    #    R:                  The process noise covariance
    #    Q:                  The measurement noise covariance
    #    visualize:          Boolean variable indicating whether to visualize
    #                        the filter
    def __init__(self, mu, Sigma, R, Q, visualize=True):
        self.mu = mu
        self.Sigma = Sigma
        self.R = R
        self.Q = Q

        # You may find it useful to keep a dictionary that maps a
        # a feature ID to the corresponding index in the mean_pose_handle
        # vector and covariance matrix
        self.mapLUT = {}

        self.visualize = visualize
        if self.visualize == True:
            self.vis = Visualization()
        else:
            self.vis = None




    # Visualize filter strategies
    #   deltat:  Step size
    #   XGT:     Array with ground-truth pose
    def render(self, XGT=None):
        deltat = 0.1
        self.vis.drawEstimates(self.mu, self.Sigma)
        if XGT is not None:
            #print XGT
            self.vis.drawGroundTruthPose (XGT[0], XGT[1], XGT[2])
        plt.pause(deltat/10)





    # Perform the prediction step to determine the mean and covariance
    # of the posterior belief given the current estimate for the mean
    # and covariance, the control data, and the process model
    #    u:                 The forward distance and change in heading
    def prediction(self, u):

        F = np.zeros((3, 3))
        noise = np.random.normal(0, self.R[0, 0])

        F[0, 0] = 1
        F[1, 1] = 1
        F[2, 2] = 1
        F[0, 2] = -1 * (u[0]+noise) * np.sin(self.mu[2])
        F[1, 2] = (u[0]+noise) * np.cos(self.mu[2])

        u1 = u[0]
        u2 = u[1]
        self.mu[0] = self.mu[0] + u1*np.cos(self.mu[2])
        self.mu[1] = self.mu[1] + u1*np.sin(self.mu[2])
        self.mu[2] = self.mu[2] + u2

        upper_left = self.Sigma[0:3, 0:3]
        upper_right = self.Sigma[0:3, 3:]
        bot_left = self.Sigma[3:, 0:3]
        bot_right = self.Sigma[3:, 3:]

        temp = np.matmul(F, upper_left)
        temp = np.matmul(temp, np.transpose(F))
        temp[0, 0] = temp[0, 0] + self.R[0, 0]
        temp[1, 1] = temp[1, 1] + self.R[0, 0]
        temp[2, 2] = temp[2, 2] + self.R[1, 1]
        upper_left = temp

        upper_right = np.matmul(F, upper_right)

        bot_left = np.matmul(bot_left, np.transpose(F))

        up = np.concatenate((upper_left, upper_right), axis=1)
        bot = np.concatenate((bot_left, bot_right), axis=1)
        if not bot.shape[1] == 0:
            self.Sigma = np.concatenate((up, bot), axis=0)

        



    # Perform the measurement update step to compute the posterior
    # belief given the predictive posterior (mean and covariance) and
    # the measurement data
    #    z:     The (x,y) position of the landmark relative to the robot
    #    i:     The ID of the observed landmark
    def update(self, z, i):

        mIdx = self.mapLUT[i]
        xt = self.mu[0]
        yt = self.mu[1]
        thetat = self.mu[2]
        xm = self.mu[mIdx]
        ym = self.mu[mIdx+1]

        H = np.zeros((2, self.mu.shape[0]))
        H[0, 0] = (-1)*np.cos(thetat)
        H[0, 1] = (-1)*np.sin(thetat)
        H[0, 2] = (-1)*xm*np.sin(thetat) + xt*np.sin(thetat) + ym*np.cos(thetat) - yt*np.cos(thetat)
        H[0, mIdx] = np.cos(thetat)
        H[0, mIdx+1] = np.sin(thetat)

        H[1, 0] = np.sin(thetat)
        H[1, 1] = (-1)*np.cos(thetat)
        H[1, 2] = (-1)*xm*np.cos(thetat) + xt*np.cos(thetat) - ym*np.sin(thetat) + yt*np.sin(thetat)
        H[1, mIdx] = (-1)*np.sin(thetat)
        H[1, mIdx+1] = np.cos(thetat)

        inv_temp = np.matmul(H, self.Sigma)
        inv_temp = np.matmul(inv_temp, np.transpose(H))
        inv_temp = inv_temp + self.Q
        temp = np.matmul(self.Sigma, np.transpose(H))
        k = np.matmul(temp, inv(inv_temp))

        temp0 = xm*np.cos(thetat) - xt*np.cos(thetat) + ym*np.sin(thetat) - yt*np.sin(thetat)
        temp1 = (-1)*xm*np.sin(thetat) + xt*np.sin(thetat) + ym*np.cos(thetat) - yt*np.cos(thetat)
        hu = np.array([temp0[0], temp1[0]])
        gain = np.matmul(k, z - hu)
        gain = np.reshape(gain, (self.mu.shape[0], 1))

        self.mu = self.mu + gain

        I = np.eye(self.mu.shape[0])
        temp = np.matmul(k, H)
        temp = I - temp
        self.Sigma = np.matmul(temp, self.Sigma)



    # Augment the state vector to include the new landmark
    #    z:     The (x,y) position of the landmark relative to the robot
    #    i:     The ID of the observed landmark
    def augmentState(self, z, i):

        self.mapLUT[i] = 3 + 2*len(self.mapLUT)
        G = np.zeros((2, self.mu.shape[0]))
        G[0, 0] = 1
        G[0, 2] = (-1)*z[0] * np.sin(self.mu[2]) - z[1] * np.cos(self.mu[2])
        G[1, 1] = 1
        G[1, 2] = z[0] * np.cos(self.mu[2]) - z[1] * np.sin(self.mu[2])

        xm = np.array([self.mu[0] + z[0] * np.cos(self.mu[2]) - z[1] * np.sin(self.mu[2])])
        ym = np.array([self.mu[1] + z[0] * np.sin(self.mu[2]) + z[1] * np.cos(self.mu[2])])

        self.mu = np.concatenate((self.mu, xm), axis=0)
        self.mu = np.concatenate((self.mu, ym), axis=0)

        
        upper_left = self.Sigma

        upper_right = np.matmul(self.Sigma, np.transpose(G))

        bot_left = np.matmul(G, self.Sigma)

        temp = np.matmul(G, self.Sigma)
        temp = np.matmul(temp, np.transpose(G))
        bot_right = temp + self.Q

        up = np.concatenate((upper_left, upper_right), axis=1)
        bot = np.concatenate((bot_left, bot_right), axis=1)
        if not bot.shape[1] == 0:
            self.Sigma = np.concatenate((up, bot), axis=0)


        # Update mapLUT to include the new landmark



    # Runs the EKF SLAM algorithm
    #   U:        Array of control inputs, one column per time step
    #   Z:        Array of landmark observations in which each column
    #             [t; id; x; y] denotes a separate measurement and is
    #             represented by the time step (t), feature id (id),
    #             and the observed (x, y) position relative to the robot
    #   XGT:      Array of ground-truth poses (may be None)
    def run(self, U, Z, XGT=None, MGT=None):
   
        # Draws the ground-truth map
        if MGT is not None:
            self.vis.drawMap (MGT)

        print("init")
        print(self.mu)
        print(self.R)
        print(self.Q)
        # Iterate over the data
        zIdx = 0
        for t in range(U.shape[1]):
        #for t in range(0, 1):
            print(t)
            u = U[:,t]
            self.prediction(u)
            if zIdx < U.shape[1]:
                while Z[0, zIdx] == t:
                    if Z[1, zIdx] in self.mapLUT:
                        self.update(Z[2:, zIdx], Z[1, zIdx])
                    else:
                        self.augmentState(Z[2:, zIdx], Z[1, zIdx])
                    zIdx = zIdx + 1


            # You may want to call the visualization function
            # between filter steps
            if self.visualize:
                if XGT is None:
                    self.render (None)
                else:
                    self.render (XGT[:,t])
        plt.savefig("small_noise")