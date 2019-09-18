import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as sla
from numpy import linalg as nla
from matplotlib.collections import LineCollection


class Visualization(object):
    # Visualization tools
    def __init__(self):

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')

        self.ax.set_xlim(-8.0, 8.0, True, False)
        self.ax.set_ylim(-8.0, 8.0, True, False)

        plt.ion()
        plt.tight_layout()

        self.true_pose_handle = None
        self.mean_pose_handle = None
        self.pose_ellipse_handle = None
        self.map_pose_handle = {}
        self.map_ellipse_handle = {}



    # Function to draw filter estimate
    #   mu:     An n x 1 array denoting the mean
    #   sigma:  An n x n array denoting the covariance matrix
    def drawEstimates (self, mu, Sigma):

        if self.mean_pose_handle == None:
            self.mean_pose_handle, = self.ax.plot(mu[0], mu[1], 'ro')
        else:
            self.mean_pose_handle.set_xdata(mu[0])
            self.mean_pose_handle.set_ydata(mu[1])

        xc = mu[0:2]
        XY = self.getEllipse (xc, Sigma[0:2][:,0:2])

        if XY is not None:
            if self.pose_ellipse_handle:
                self.pose_ellipse_handle.set_xdata(XY[0,:])
                self.pose_ellipse_handle.set_ydata(XY[1,:])
            else:
                self.pose_ellipse_handle, = self.ax.plot(XY[0,:], XY[1,:])


        # Plot the pose of each landmark
        for i in range(3,len(mu),2):
            self.ax.plot(mu[i], mu[i+1],'k.')

            xc = xc = mu[i:i+2]

            # if self.map_pose_handle.has_key(i):
            if i in self.map_pose_handle:
                handle = self.map_pose_handle[i]
                handle.set_xdata(xc[0])
                handle.set_ydata(xc[1])
            else:
                handle, = self.ax.plot(xc[0], xc[1])
                self.map_pose_handle[i] = handle


            XY = self.getEllipse (xc, Sigma[i:i+2][:,i:i+2])

            if XY is not None:
                # if self.map_ellipse_handle.has_key(i):
                if i in self.map_ellipse_handle:
                    handle = self.map_ellipse_handle[i]
                    handle.set_xdata(XY[0,:])
                    handle.set_ydata(XY[1,:])
                else:
                    handle, = self.ax.plot(XY[0,:], XY[1,:])
                    self.map_ellipse_handle[i] = handle

        self.fig.canvas.draw()


    # Function that draws the ground-truth map
    #   M:  3 x N array in which each column denotes an individual
    #       landmark and is of the form [id; x; y]
    def drawMap (self, M):

        for i in range(M.shape[1]):
            self.ax.plot (M[1,i], M[2,i], 'gx')

        self.fig.canvas.draw()

        

    # Function that returns the coordinates of an ellipse corresponding
    # to the level set for a given covariance matrix
    #
    #   Sigma:    Covariance matrix as a 2 x 2 array
    #   xc:       Center of the ellipse as a 2 x 1 array
    #   nSigma:   Multiple of standard deviations (default: 2)
    #
    # Returns:
    #   XY:       2 x m array where each column defines an [x; y] coordinate
    def getEllipse (self, xc, Sigma, nSigma=2):

        if nla.det(Sigma) == 0:
            return None

        w, v = nla.eig(Sigma)
        D = np.diag(w, 0)

        theta = np.linspace(0,2*np.pi, 100, endpoint=True)
        circle = nSigma*np.vstack((np.cos(theta),np.sin(theta)))


        el = sla.sqrtm(D)
        el = el.dot(circle)
        el = v.dot(el)

        XY = xc + el

        return XY




    # Function to draw ground-truth pose
    #   x, y, theta:   Position and orientation
    def drawGroundTruthPose (self, x, y, theta):

        if self.true_pose_handle == None:
            self.true_pose_handle, = self.ax.plot(x, y, 'g.')
        else:
            self.true_pose_handle.set_xdata(x)
            self.true_pose_handle.set_ydata(y)

        self.fig.canvas.draw()
