from typing import List

import numpy as np
import math


def circle_fit_by_taubin(xy: np.ndarray) -> [List[float], float]:
    """
    Fits a circle to the data points in the input array xy
    :param xy: each row contains the (x, y) coordinates of the data point
    :return: [center(x,y), radius] of the fitted circle
    """
    n = xy.shape[0]                 # number of data points
    centroid = np.mean(xy, axis=0)  # the centroid of the data set

    # computing moments (note: all moments will be normalized, i.e. divided by n)

    Mxx = Myy = Mxy = Mxz = Myz = Mzz = 0

    for i in range(0, n):
        x = xy[i][0] - centroid[0]
        y = xy[i][1] - centroid[1]
        z = x**2 + y**2
        Mxy += x*y
        Mxx += x**2
        Myy += y**2
        Mxz += x*z
        Myz += y*z
        Mzz += z**2

    Mxx = Mxx/n
    Myy = Myy/n
    Mxy = Mxy/n
    Mxz = Mxz/n
    Myz = Myz/n
    Mzz = Mzz/n

    # computing the coefficients of the characteristic polynomial

    Mz = Mxx + Myy
    Cov_xy = Mxx*Myy - Mxy**2
    A3 = 4*Mz
    A2 = -3*Mz*Mz - Mzz
    A1 = Mzz*Mz + 4*Cov_xy*Mz - Mxz**2 - Myz**2 - Mz**3
    A0 = Mxz*Mxz*Myy + Myz*Myz*Mxx - Mzz*Cov_xy - 2*Mxz*Myz*Mxy + Mz*Mz*Cov_xy
    A22 = A2 + A2
    A33 = A3 + A3 + A3

    xnew = 0
    ynew = 1e+20
    epsilon = 1e-12
    IterMax = 20

    # Newton's method starting at x = 0

    for j in range(1, IterMax + 1):
        yold = ynew
        ynew = A0 + xnew*(A1 + xnew*(A2 + xnew*A3))
        if abs(ynew) > abs(yold):
            print("Newton-Taubin goes wrong direction: |ynew| > |yold|")
            break
        Dy = A1 + xnew*(A22 + xnew*A33)
        xold = xnew
        xnew = xold - ynew/Dy

        if abs((xnew - xold)/xnew) < epsilon:
            break
        if j >= IterMax:
            print("Newton-Taubin will not converge")
            xnew = 0
        if xnew < 0:
            print("Newton-Taubin negative root: x =", xnew)
            xnew = 0

    # computing the circle parameters

    DET = xnew**2 - xnew*Mz + Cov_xy
    Center = ((Mxz * (Myy - xnew) - Myz * Mxy), (Myz * (Mxx - xnew) - Mxz * Mxy)) / DET / 2

    return [Center+centroid, math.sqrt(Center@np.transpose(Center) + Mz)]
