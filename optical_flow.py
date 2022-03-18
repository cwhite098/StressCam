import cv2 as cv
from cv2 import NORM_L2
import numpy as np
import matplotlib.pyplot as plt

def optical_flow(frame1, frame2):
    delay = 23 # 60*(13/60)*30
    # frame1, frame2 = Capture(delay)
    # frame1 = cv.imread('flow_frame_1.jpg')
    # frame2 = cv.imread('flow_frame_2.jpg')

    if not (frame1.shape == frame2.shape):
        print("The two image sizes do not match!")
        width = frame1.shape[1]
        height = frame1.shape[0]
        frame2 = cv.resize(frame2, (width, height), interpolation=cv.INTER_AREA)


    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)

    # flow_image = np.zeros_like(frame1)
    flow_image = frame1.copy()


    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    row, col , _ = flow.shape
    U, V = flow.T
    U = cv.normalize(U, None, 1, 0, NORM_L2)
    V = cv.normalize(V, None, 1, 0, NORM_L2)
    # U = np.round(U)
    # V = np.round(V)

    # U_NotZero = np.logical_or(U > 1e-3, U < -1e-3)
    # V_NotZero = np.logical_or(V > 1e-3, V < -1e-3)
    V_Zero = np.logical_and(V < 1e-3, V > -1e-3)
    idx_x, idx_y = np.meshgrid(np.arange(row), np.arange(col))
    grid = np.indices((row, col), sparse=True)

    X = idx_x[V_Zero]
    Y = idx_y[V_Zero]
    # U = U[V_Zero]
    # V = V[V_Zero]
    U[V_Zero] = 0
    V[V_Zero] = 0
    U = U.T
    V = V.T

    # plt.quiver(X, Y, U, V, scale=0.4, minlength=0.01)
    # plt.show()
    return (X, Y, U, V)
