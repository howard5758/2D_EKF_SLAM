import numpy as np
from Gridmap import Gridmap

class Laser(object):
    # Construct an Laser instance with the following set of variables,
    # which are described in Section 6.3.1 of Probabilistic Robotics
    #   numBeams:   Number of beams that comprise the scan
    def __init__(self, numBeams = 41):
        self.pHit = 0.9800
        self.pShort = 0.01
        self.pMax = 0.005
        self.pRand = 0.005
        self.sigmaHit = 0.02
        self.lambdaShort = 1
        self.zMax = 20
        self.zMaxEps = 0.02
        self.Angles = np.linspace(-np.pi, np.pi, numBeams) # array of angles



    # The following computes the likelihood of a given LIDAR scan from
    # a given pose in the provided map according to the algorithm given
    # in Table 6.1 of Probabilistic Robotics
    #
    #   Ranges:        An array of ranges (the angles are defined by self.Angles)
    #   x, y, thta:    The robot's position (x,y) and heading
    #   gridmap:       An instance of the Gridmap class that specifies
    #                  an occupancy grid representation of the map
    #                  where 1: occupied and 0: free
    #
    # Returns:
    #   likelihood:     Scan likelihood
    def scanProbability (self, z, x, gridmap):

        # Your code goes here
        # Implement the algorithm given in Table 6.1
        # You are provided with an implementation (albeit slow) of ray tracing below

        print "Please add code"



    # Function to convert range and bearing to (x,y) in LIDAR frame
    #   range:   1xn array of range measurements
    #   bearing: 1xn array of bearings
    #
    # Returns:
    #   XY:      2xn array, where each column is an (x,y) pair
    def getXY (self, range, bearing):

        CosSin = np.vstack((np.cos(bearing[:]),np.sin(bearing[:])))
        XY = np.tile(range,(2,1))*CosSin

        return XY



    # An implementation of ray tracing
    #   (xr, yr, thetar):   The robot's pose
    #   lAngle:             The LIDAR angle (in the LIDAR reference frame)
    #   gridmap:            An instance of the Gridmap class that specifies
    #                       an occupancy grid representation of the map
    #                       where 1: occupied and 0: free
    #
    # Returns:
    #   d:                  Range
    #   coords:             Array of (x,y) coordinates
    def rayTracing (self, xr, yr, thetar, lAngle, gridmap):

        angle = thetar + lAngle
        x0 = xr/gridmap.xres
        y0 = yr/gridmap.yres

        (m,n) = gridmap.getShape()
        if gridmap.inCollision(int(np.floor(x0)), int(np.floor(y0)), True):
            d = 0
            coords = np.array([[x0,y0]]).transpose()
            return (d, coords)

        if x0 == np.floor(x0):
            x0 = x0 + 0.001

        if y0 == np.floor(y0):
            y0 = y0 + 0.001


        eps = 0.0001


        # Intersection with horizontal lines
        x = x0
        y = y0
        found = False

        if np.mod(angle, np.pi) != np.pi/2:
            tanangle = np.tan(angle)
            if np.cos(angle) >= 0:
                while (x <= (n - 1)) and (found == False):
                    x = np.floor(x+1)
                    y = y0 + tanangle*(x-x0)

                    if (y > ((m - 1))) or (y < 0):
                        break

                    if gridmap.inCollision(int(np.floor(x+eps)), int(np.floor(y)), True) == 1:
                        xfound_hor = x
                        yfound_hor = y
                        found = True
            else:
                while (x >= 1) and (found == False):
                    x = int(np.ceil(x-1))
                    y = y0 + tanangle*(x-x0)

                    if (y > ((m - 1))) or (y < 0):
                        break

                    if gridmap.inCollision(int(np.floor(x-eps)), int(np.floor(y)),True):
                        xfound_hor = x
                        yfound_hor = y
                        found = True


        found_hor = found

        # Intersection with vertical lines
        x = x0;
        y = y0;
        found = False

        if np.mod(angle, np.pi) != 0:
            cotangle = 1/np.tan(angle)
            if np.sin(angle) >= 0:
                while (y <= (m - 1)) and (found == False):
                    y = np.floor(y+1)
                    x = x0 + cotangle*(y-y0)

                    if (x > ((n - 1))) or (x < 0):
                        break

                    if gridmap.inCollision(int(np.floor(x)), int(np.floor(y+eps)), True):
                        xfound_ver = x
                        yfound_ver = y
                        found = True
            else:
                while (y >= 1) and (found == False):
                    y = int(np.floor(y-1))
                    x = x0 + cotangle*(y-y0)

                    if (x > ((n - 1))) or (x < 0):
                        break

                    if gridmap.inCollision(int(np.floor(x)), int(np.floor(y-eps)), True):
                        xfound_ver = x
                        yfound_ver = y
                        found = True

        found_ver = found

        if (found_hor == False) and (found_ver == False):
            print 'rayTracing: Error finding return'


        # Check which one was first
        if (found_ver == True) and (found_hor == False):
            d_ver = np.sqrt(np.square(xfound_ver - x0) + np.square(yfound_ver - y0))
            d = d_ver
            coords = np.array([[xfound_ver, yfound_ver]]).transpose()
        elif (found_hor == True) and (found_ver == False):
            d_hor = np.sqrt(np.square(xfound_hor - x0) + np.square(yfound_hor - y0))
            d = d_hor
            coords = np.array([[xfound_hor, yfound_hor]]).transpose()
        else:
            d_ver = np.sqrt(np.square(xfound_ver - x0) + np.square(yfound_ver - y0))
            d_hor = np.sqrt(np.square(xfound_hor - x0) + np.square(yfound_hor - y0))

            if d_hor <= d_ver:
                coords = np.array([[xfound_hor, yfound_hor]]).transpose()
                d = d_hor
            else:
                coords = np.array([[xfound_ver, yfound_ver]]).transpose()
                d = d_ver

        return (d, coords)
