# %%
from heapq import nlargest
from socket import NI_DGRAM
from tkinter import SEL_FIRST
from matplotlib import animation
from scipy import signal
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import perf_counter, time
from threading import Thread

class Resp_Rate:
    def __init__(self, refresh_rate=30, win_size=300, vid=None, ROI=None):
        self.vid = vid
        self.ROI = ROI
        self.First = True
        self.refresh_rate = refresh_rate 
        self.win_size = win_size # number of frame on plot
        self.fps = 15
        self.p_y_f = []

    def resp_pattern(self, frames):
        '''Extract respiration pattern'''
        self.y_len, self.x_len, _ = frames[0].shape
        sd_list = []
        for frame in frames:
            p_f = np.sum(np.sum(frame, axis=2), axis=1)/self.x_len
            p_f_detrend = signal.detrend(p_f, type='constant')
            self.p_y_f.append(p_f_detrend)
        self.p_y_f = np.array(self.p_y_f)
        for y in range(self.y_len):
            sd_list.append(np.std(self.p_y_f[:, y]))
        sd_list = np.array(sd_list)
        largest_sd = nlargest(int(0.05*self.y_len), sd_list)
        idx = [np.where(sd_list==i) for i in largest_sd] 
        p_5per = self.p_y_f[:, idx]
        p_f = [np.mean(i) for i in p_5per]
        b, a = signal.butter(3, (0.05, 2), fs=30, btype='bandpass', analog=False)
        y = signal.lfilter(b, a, p_f)
        self.p_norm = (y-np.mean(y))/np.std(y)

    def resp_pattern_per_frame(self, frame, fps):
        # Calculate averaged intensity componenets
        self.p_y_f[0:-1,:] = self.p_y_f[1:,:] # move values forward by one row
        p_f = np.sum(np.sum(frame, axis=2), axis=1)/self.x_len
        p_f_detrend = signal.detrend(p_f, type='constant')
        self.p_y_f[-1,:] = p_f_detrend # (refresh_rate(n frames) x y_len)
        sd_list = [np.std(self.p_y_f[:, y]) for y in range(self.y_len)] # (y_lne x 1)
        largest_sd = nlargest(int(0.05*self.y_len), sd_list)
        idx = [np.where(sd_list==i) for i in largest_sd]
        p_5per = self.p_y_f[:, idx]
        p_f = [np.mean(i) for i in p_5per]
        b, a = signal.butter(3, (0.05, 2), fs=15, btype='bandpass', analog=False)
        y = signal.lfilter(b, a, p_f)
        self.p_norm = (y-np.mean(y))/np.std(y) # (refresh_rate x 1)
        pass

    def resp_rate(self, n_frame, fps):
        '''Calculate respiration rate from signal'''
        cross_zero_idx = []
        for i in range(1, self.p_norm.size):
            if self.p_norm[i-1]<0 and self.p_norm[i]>=0: # p_norm[i-1]>0 and p_norm[i+1]<0 or 
                cross_zero_idx.append(i)
        min_idx_list = []
        for i in range(1, len(cross_zero_idx)):
            idx = np.where(self.p_norm==min(self.p_norm[cross_zero_idx[i-1]:cross_zero_idx[i]]))
            min_idx_list.append(idx)
        # f_Ri = [60/(min_idx_list[i][0]-min_idx_list[i-1][0]) for i in range(1, len(min_idx_list))]
        # ^FIX: min_idx_list[i] return tuple
        f_R = len(min_idx_list)/((n_frame/fps)/60) # number of breath / ((number of frame / 30 fps) / 60)
        return f_R        

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
        '''Plot fig'''
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        for col in range(self.y_len):
            # self.ax0.plot(self.p_y[:, col])
            self.ax1.plot(self.p_y_f[:, col]) 
        self.sig = self.ax2.plot(self.p_norm)[0]
        self.text = self.ax2.text(0.5, 2, round(self.resp_rate(n_frame), ndigits=5))
        plt.show()
        self.cap.release()
        cv.destroyAllWindows()
    
    def live_feed_per_frame_animate(self, i):
        start = perf_counter()
        for i in range(self.refresh_rate):
            _, cur = self.cap.read()
            cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
            int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
            cv.imshow('frame', cropped)
            if cv.waitKey(1) == ord('q'):
                self.cap.release()
                cv.destroyAllWindows()
            self.resp_pattern_per_frame(cropped, self.fps)
        print(self.fps)
        bpm = self.resp_rate(self.win_size, self.fps)
        self.sig.set_ydata(self.p_norm)
        self.rate.set_text('bpm: %.2f' % bpm)
        self.fps_text.set_text('fps: %.2f' % self.fps)
        self.fps = self.refresh_rate / (perf_counter() - start)
        return self.sig,
    
    def live_feed_per_frame_loop(self, cur, fps):
        cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
        int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
        self.resp_pattern_per_frame(cropped, fps)
        bpm = self.resp_rate(self.win_size, fps)

        self.fig.canvas.restore_region(self.background)
        self.sig.set_ydata(self.p_norm)
        self.rate.set_text('bpm: %.2f' % bpm)
        self.fps_text.set_text('fps: %.2f' % fps)
        self.ax.draw_artist(self.sig)
        self.ax.draw_artist(self.fps_text)
        self.fig.canvas.blit(self.ax.bbox)
        return cropped
    
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
        '''Initial animation plot'''
        plt.ion
        # self.fig = plt.figure()
        # self.ax = plt.axes(xlim=(0, self.win_size), ylim=(-2.5, 2.5))
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim(0, self.win_size)
        self.ax.set_ylim(-2.5, 2.5)
        self.sig, = self.ax.plot(self.p_norm)
        self.rate = self.ax.text(0.5, 2.3, '')
        self.fps_text = self.ax.text(0.5, 2, '')
        self.fig.show()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
    
    def live_feed_init(self, cap):
        '''Initialise ROI'''
        if self.ROI is None:
            _, cur = cap.read()
            self.ROI = cv.selectROI('ROI', cur)
            cv.destroyWindow('ROI')
            self.frame_y, self.frame_x, _ = cur.shape
            self.y_len, self.x_len = (self.ROI[3], self.ROI[2])
        '''Initialise matrices'''
        if self.vid is None:
            self.p_y_f = np.array([[0] * self.y_len] * self.win_size)
            # self.cropped_list = np.array([[0]] * win_size)
            self.p_norm = np.array([0] * self.win_size)
        '''Initialise matrix p to prevent row having the same value'''
        start = perf_counter()
        for i in range(10):
            _, cur = cap.read()
            cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
            int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
            self.p_y_f[0:-1,:] = self.p_y_f[1:,:] # move values forward by one row
            p_f = np.sum(np.sum(cropped, axis=2), axis=1)/self.x_len
            p_f_detrend = signal.detrend(p_f, type='constant')
            self.p_y_f[-1,:] = p_f_detrend # (refresh_rate(n frames) x y_len)
        self.fps = 10 / (perf_counter() - start)

    def live_feed(self):
        '''Process online input'''
        self.cap = cv.VideoCapture()
        self.cap.open(0, cv.CAP_DSHOW)
        self.live_feed_init(self.cap)
        self.animate_init()
        ain = FuncAnimation(self.fig, self.live_feed_per_frame_animate)
        plt.show()
        self.cap.release()
        cv.destroyAllWindows()

    def analysis_feed(self):
        if self.vid is None:
            self.live_feed()
        else:
            self.vid_feed()

# Resp = Resp_Rate(vid='Recording2_Trim.mp4', ROI=(534, 465, 224, 173))
# Resp = Resp_Rate()
# p_list = Resp.analysis_feed()
