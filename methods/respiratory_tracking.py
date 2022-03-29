# %%
from heapq import nlargest
from tkinter import SEL_FIRST
from matplotlib import animation
from scipy import signal
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import perf_counter
from threading import Thread

class Resp_Rate:
    def __init__(self, win_size=300, vid=None, animate=True, ROI=None):
        self.vid = vid
        self.animate = animate
        self.ROI = ROI
        self.First = True
        self.dur = win_size # number of frame
        self.fps = 10
        # if vid is None:
        #     self.p_y_f = np.array([[0]] * win_size)
        #     self.cropped_list = np.array([[0]] * win_size)
        #     self.p_norm = np.array([0] * win_size)
        # else:
        #     self.p_y_f = []
        #     self.p_norm = []
    
    def average_intensity(self, frame):
        p_f = [(sum([sum([frame[y, x, i] for i in range(3)]) for x in range(self.x_len)]))/self.x_len for y in range(self.y_len)]
        p_f_detrend = signal.detrend(p_f, type='constant')
        self.p_y_f.append(p_f_detrend)

    def resp_pattern(self, frames):
        '''Extract respiration pattern'''
        self.y_len, self.x_len, _ = frames[0].shape
        start = perf_counter()
        threads = [Thread(target=self.average_intensity, args=(frame,))
        for frame in frames] # Creat thread for each frame (limit number of thread in the futrue)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(perf_counter()-start)
        self.p_y_f = np.array(self.p_y_f)
        for y in range(self.y_len):
            self.sd_list.append(np.std(self.p_y_f[:, y]))
        self.sd_list = np.array(self.sd_list)
        largest_sd = nlargest(int(0.05*self.y_len), self.sd_list)
        idx = [np.where(self.sd_list==i) for i in largest_sd] 
        p_5per = self.p_y_f[:, idx]
        p_f = [np.mean(i) for i in p_5per]
        b, a = signal.butter(3, (0.05, 2), fs=30, btype='bandpass', analog=False)
        y = signal.lfilter(b, a, p_f)
        self.p_norm = (y-np.mean(y))/np.std(y)
    
    def init_p(self):
        for i in range(10):
            _, cur = self.cap.read()
            cv.imshow('frame', cur)
            if cv.waitKey(1) == ord('q'):
                self.cap.release()
                cv.destroyAllWindows()
            cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
            int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
            self.p_y_f[0:-1,:] = self.p_y_f[1:,:] # move values forward by one row
            p_f = [(sum([sum([cropped[y, x, i] for i in range(3)]) for x in range(self.x_len)]))/self.x_len for y in range(self.y_len)]
            p_f_detrend = signal.detrend(p_f, type='constant')
            self.p_y_f[-1,:] = p_f_detrend # (win_size(n frames) x y_len)
    
    def resp_pattern_per_frame(self, frame):
        # Calculate averaged intensity componenets
        self.p_y_f[0:-1,:] = self.p_y_f[1:,:] # move values forward by one row
        p_f = [(sum([sum([frame[y, x, i] for i in range(3)]) for x in range(self.x_len)]))/self.x_len for y in range(self.y_len)]
        p_f_detrend = signal.detrend(p_f, type='constant')
        self.p_y_f[-1,:] = p_f_detrend # (win_size(n frames) x y_len)
        sd_list = [np.std(self.p_y_f[:, y]) for y in range(self.y_len)] # (y_lne x 1)
        largest_sd = nlargest(int(0.05*self.y_len), sd_list)
        idx = [np.where(sd_list==i) for i in largest_sd]
        # idx = sd_list==largest_sd
        # print(idx, idx[-1])
        p_5per = self.p_y_f[:, idx]
        p_f = [np.mean(i) for i in p_5per]
        b, a = signal.butter(3, (0.05, 2), fs=30, btype='bandpass', analog=False)
        y = signal.lfilter(b, a, p_f)
        self.p_norm = (y-np.mean(y))/np.std(y) # (win_size x 1)
        pass

    def resp_rate(self):
        '''Calculate respiration rate'''
        cross_zero_idx = []
        for i in range(1, self.p_norm.size):
            if self.p_norm[i-1]<0 and self.p_norm[i]>=0: # p_norm[i-1]>0 and p_norm[i+1]<0 or 
                cross_zero_idx.append(i)
        min_idx_list = []
        for i in range(1, len(cross_zero_idx)):
            idx = np.where(self.p_norm==min(self.p_norm[cross_zero_idx[i-1]:cross_zero_idx[i]]))
            min_idx_list.append(idx)
        f_Ri = [60/(min_idx_list[i]-min_idx_list[i-1]) for i in range(1, len(min_idx_list))]
        # ^FIX: min_idx_list[i] return tuple
        return f_Ri
    
    def plot_sig(self):
        '''Plot respiration signal'''
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        for col in range(self.y_len):
            # self.ax0.plot(self.p_y[:, col])
            self.ax1.plot(self.p_y_f[:, col]) 
        self.sig = self.ax2.plot(self.p_norm)[0]
        
    def vid_feed(self):
        '''Process pre recorded video'''
        start = perf_counter()
        self.cap = cv.VideoCapture(self.vid)
        if self.ROI is None:
            # Select ROI manually
            _, cur = self.cap.read()
            ROI = cv.selectROI('ROI', cur)
            cv.destroyWindow('ROI')
        else:
            ROI = self.ROI
        # ROI = (531, 464, 244, 139)
        # ROI = (534, 465, 224, 173)
        cropped_list = []
        n_frame = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) 
        for i in range(n_frame):
            _, cur = self.cap.read()
            if cur is None:
                break
            # cv.imshow('frame', cur)
            cropped = cur[int(ROI[1]):int(ROI[1]+ROI[3]), 
                            int(ROI[0]):int(ROI[0]+ROI[2])]
            cropped_list.append(cropped)
        print('Time spent for cropping: ', perf_counter()-start)
        start = perf_counter()
        cropped_list = np.array(cropped_list)
        self.resp_pattern(cropped_list)
        print('Time spent for pattern extraction: ', perf_counter()-start)
        self.plot_sig()
        # rate = self.resp_rate()
        # print(rate)
        plt.show()
        self.cap.release()
        cv.destroyAllWindows()
    
    def live_feed_per_frame_animate(self, i):
        
        for i in range(150):
            start = perf_counter()
            _, cur = self.cap.read()
            cv.imshow('frame', cur)
            if cv.waitKey(1) == ord('q'):
                self.cap.release()
                cv.destroyAllWindows()
            cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
            int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
            self.resp_pattern_per_frame(cropped)
            print(perf_counter() - start)
        self.sig.set_ydata(self.p_norm)
        return self.sig,
    
    def live_feed_per_frame(self):
        # 17 sec ~ 3 cycle
        while True:
            start = perf_counter()
            _, cur = self.cap.read()
            cv.imshow('frame', cur)
            if cv.waitKey(1) == ord('q'):
                break
            cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
            int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
            self.resp_pattern_per_frame(cropped)
            print(self.p_norm[-1])
    
    def animate_init(self):
        self.sig, = self.ax.plot(self.p_norm)
    
    def live_feed(self):
        '''Process online input'''
        self.cap = cv.VideoCapture()
        self.cap.open(0, cv.CAP_DSHOW)
        _, cur = self.cap.read()
        self.ROI = cv.selectROI('ROI', cur)
        cv.destroyWindow('ROI')
        self.y_len, self.x_len, _ = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
                                int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])].shape
        if self.vid is None:
            self.p_y_f = np.array([[0] * self.y_len] * self.dur)
            # self.cropped_list = np.array([[0]] * win_size)
            self.p_norm = np.array([0] * self.dur)
        # self.cropped_list = np.array()
        if self.animate:
            self.fig = plt.figure()
            self.ax = plt.axes(xlim=(0, self.dur), ylim=(-2.5, 2.5))
            self.init_p()
            ain = FuncAnimation(self.fig, self.live_feed_per_frame_animate, init_func=self.animate_init)
            plt.show()
        else:
            # self.live_feed_loop()
            self.live_feed_per_frame()
        self.cap.release()
        cv.destroyAllWindows()

    def analysis_feed(self):
        if self.vid is None:
            self.live_feed()
        else:
            self.vid_feed()

# Resp = Resp_Rate(vid='Recording2_Trim.mp4', ROI=(534, 465, 224, 173))
Resp = Resp_Rate(animate=True)
p_list = Resp.analysis_feed()

# %%
# rate = self.resp_rate()
# print(rate)