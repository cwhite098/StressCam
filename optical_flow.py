#%%
from math import exp
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# win_size = 1
def motion_matrix(win_size):
    cap = cv.VideoCapture()
    cap.open(0, cv.CAP_DSHOW)
    _, prvs = cap.read()
    prvs = cv.cvtColor(prvs, cv.COLOR_BGR2GRAY)
    M = []
    for i in range(win_size):
        _, next = cap.read()
        next = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        _, V = flow.T
        V = np.reshape(V, V.size)
        M.append(V)
        prvs = next
    cap.release()
    cv.destroyAllWindows()
    return np.array(M)

def mid_respiratory_descriptor(M):
    row, col = M.shape
    lamda_list = []
    f = []
    d = []
    for i in range(int(col/row)):
        start = i * row
        end = (i + 1) * row
        m_i = M[:, start:end]
        lamda, v = np.linalg.eig(m_i)
        lamda_list.append(lamda[0])
        f_i = lamda[0]*v[0] 
        d_i = np.full_like(f_i, -1)
        one = f_i >= 0
        d_i[one] = 1
        d.append(d_i)
        f.append(f_i)

    return d, lamda_list, f
#%%
def descriptor_score(d, R, a, b, n, W, bpm_max):
    S = []
    # S.append(d[0])
    for i in range(len(d)):
        if i == 0:
            d_minus1 = np.ones_like(d[i])
        else:
            d_minus1 = d[i-1]
        d_t = d[i]
        R_i = abs(R[i])
        if R_i > bpm_max:
            e = R_i - bpm_max
        else:
            e = 0
        max_bpm = max([i**(b)*exp(-i/n).real for i in range(0, 100)])
        boost_resp_range = ((R_i**b)*exp(-R_i/n))/max_bpm
        penalise_noise = exp(-a*e)
        en_temporal_consistency = sum([d_minus1[k+1]*d_t[k] for k in range(W-1)])
        S_i = boost_resp_range * penalise_noise * en_temporal_consistency
        S.append(S_i)
    S = np.array(S)
    S = (S-min(S))/(max(S)-min(S))
    d_head = S.copy()
    d_head[d_head<0.5] = 0
    return d_head

#%%
def ROI_detection(M, d):
    # ROI_1
    M_norm = normalize(M)
    M_norm_max = np.amax(M_norm*d) # dimension of M_norm does not match d
    ROI_1 = (M_norm*d)/M_norm_max
    # ROI_2
    THR = 0.7 # pre-defined threshold
    One = ROI_1 >= THR
    ROI_2 = np.zeros_like(ROI_1)
    ROI_2[One] = 1
    # ROI_3
    U, S, V = np.linalg.svd(M)

#%%
cycle = 23
M = motion_matrix(cycle)
descriptors, eig_val, features =  mid_respiratory_descriptor(M)
d_head = descriptor_score(descriptors, np.array(eig_val).real, 1.33, 0.39, 50, cycle, 44)

#%%
ROI = ROI_detection(M, d_head)

#%%
# M
# plt.hist(d, bins=20)



#%%
def optical_flow(frame1, frame2):
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

    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    row, col , _ = flow.shape
    U, V = flow.T
    U = cv.normalize(U, None, 1, 0, cv.NORM_L2)
    V = cv.normalize(V, None, 1, 0, cv.NORM_L2)

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

    return (X, Y, U, V)
