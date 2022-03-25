from heapq import nlargest
from matplotlib import animation
from scipy import signal
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time   

class Resp_Rate:
    def __init__(self, win_size=20, vid=None, animate=True):
        self.vid = vid
        self.animate = animate
        self.First = True
        self.dur = win_size
        self.fps = 10
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

    def resp_pattern(self, frames, init=True):
        if init:
            p_y_f = []
            sd_list = []
            for frame in frames:
                self.y_len, self.x_len, _ = frames[0].shape
                p_f = [(sum([sum([frame[y, x, i] for i in range(3)]) for x in range(self.x_len)]))/self.x_len for y in range(self.y_len)]
                p_f_detrend = signal.detrend(p_f, type='constant')
                p_y_f.append(p_f_detrend)
            # self.p_y_f = self.p_y_f/np.linalg.norm(self.p_y_f)
            self.p_y_f = np.array(p_y_f)
            for y in range(self.y_len):
                sd = np.std(self.p_y_f[:, y])
                sd_list.append(sd)
            self.sd_list = np.array(sd_list)
        else:
            self.p_y_f[0:-1,:] = self.p_y_f[1:,:]
            # self.sd_list[0:-1] = self.sd_list[1:]
            p_f = [(sum([sum([frames[y, x, i] for i in range(3)]) for x in range(self.x_len)]))/self.x_len for y in range(self.y_len)]
            p_f_detrend = signal.detrend(p_f, type='constant')
            sd_list = []
            for y in range(self.y_len):
                sd = np.std(self.p_y_f[:, y])
                sd_list.append(sd)
            self.p_y_f[-1,:] = p_f_detrend
            self.sd_list = np.array(sd_list)

        largest_sd = nlargest(int(0.05*self.y_len), self.sd_list)
        idx = [np.where(self.sd_list==i) for i in largest_sd] 
        p_5per = self.p_y_f[:, idx]
        p_f = [np.mean(i) for i in p_5per]
        b, a = signal.butter(3, (0.05, 2), fs=30, btype='bandpass', analog=False)
        y = signal.lfilter(b, a, p_f)
        self.p_norm = (y-np.mean(y))/np.std(y)

    def resp_rate(self, p_norm):
        cross_zero_idx = []
        for i in range(1, p_norm):
            if p_norm[i-1]<0 and p_norm[i]>0: # p_norm[i-1]>0 and p_norm[i+1]<0 or 
                cross_zero_idx.append(i)
        min_idx_list = []
        for i in range(1, len(cross_zero_idx)):
            idx = p_norm==min(p_norm[cross_zero_idx[i-1]:cross_zero_idx[i]])
            min_idx_list.append(idx)
        f_Ri = [60/(min_idx_list[i]-min_idx_list[i-1]) for i in range(1, len(min_idx_list))]
        return f_Ri

    def init_draw(self):
        self.fig, self.ax = plt.subplots(1, 1)
        sig = self.ax.plot(self.p_norm, animated=True)[0]
        self.fig.show()
        self.sig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
    
    def update_draw(self):
        self.fig.canvas.restore_region(self.background)
        self.sig.set_ydata(self.p_norm)
        self.ax.draw_artist(self.sig)
        self.fig.canvas.blit(self.ax.bbox)
    
    def update_fig(self, i):
        # self.ax1.set_ydata(self.p_y_f)
        self.sig.set_ydata(self.p_norm)
        pass
    
    def plot_sig(self):
        # self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        for col in range(self.y_len):
            self.ax1.plot(self.p_y_f[:, col])
        self.sig = self.ax2.plot(self.p_norm)[0]
        
    def vid_feed(self):
        cap = cv.VideoCapture(self.vid)
        _, cur = cap.read()
        ROI = cv.selectROI('ROI', cur)
        cv.destroyWindow('ROI')
        # ROI = (531, 464, 244, 139)
        # ROI = (534, 465, 224, 173)
        cropped_list = []
        n_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        start = time.time()
        for i in range(n_frame):
            _, cur = cap.read()
            if cur is None:
                break
            # cv.imshow('frame', cur)
            cropped = cur[int(ROI[1]):int(ROI[1]+ROI[3]), 
                            int(ROI[0]):int(ROI[0]+ROI[2])]
            cropped_list.append(cropped)
        cropped_list = np.array(cropped_list)
        self.resp_pattern(cropped_list)
        self.plot_sig()
        print(time.time()-start)
        plt.show()
        cap.release()
        cv.destroyAllWindows()
    
    def live_feed_animate(self, i):
        start = time.time()
        if self.First:
            # init window
            while time.time()-start<=self.dur:
                _, cur = self.cap.read()
                cv.imshow('frame', cur)
                if cv.waitKey(1) == ord('q'):
                    break
                cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
                                int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
                self.cropped_list.append(cropped)
            self.cropped_list = np.array(self.cropped_list)
            self.First = False
            self.resp_pattern(self.cropped_list)
            self.plot_sig()
        else:
            _, cur = self.cap.read()
            cv.imshow('frame', cur)
            if cv.waitKey(1) == ord('q'):
                self.cap.release()
                cv.destroyAllWindows()
            cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
                            int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
            self.cropped_list[0:-1] = self.cropped_list[1:]
            self.cropped_list[-1] = cropped
            self.resp_pattern(cropped, init=False)
            self.sig.set_ydata(self.p_norm)
        return self.sig,
    
    def live_feed_loop(self):
        while True:
            start = time.time()
            if self.First:
                # init window
                while time.time()-start<=self.dur:
                    _, cur = self.cap.read()
                    cv.imshow('frame', cur)
                    if cv.waitKey(1) == ord('q'):
                        break
                    cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
                                    int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
                    self.cropped_list.append(cropped)
                self.cropped_list = np.array(self.cropped_list)
                self.First = False
                self.resp_pattern(self.cropped_list)
                self.plot_sig()
            else:
                _, cur = self.cap.read()
                cv.imshow('frame', cur)
                if cv.waitKey(1) == ord('q'):
                    break
                cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
                                int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
                self.cropped_list[0:-1] = self.cropped_list[1:]
                self.cropped_list[-1] = cropped
                self.resp_pattern(cropped, init=False)
    
    def live_feed(self):
        self.cap = cv.VideoCapture()
        self.cap.open(0, cv.CAP_DSHOW)
        _, cur = self.cap.read()
        self.ROI = cv.selectROI('ROI', cur)
        cv.destroyWindow('ROI')
        self.cropped_list = []
        if self.animate:
            ain = FuncAnimation(self.fig, self.live_feed_animate)
            plt.show()
        else:
            self.live_feed_loop()
        self.cap.release()
        cv.destroyAllWindows()

    def analysis_feed(self):
        if self.vid is None:
            self.live_feed()
        else:
            self.vid_feed()

Resp = Resp_Rate(vid='Recording2_Trim.mp4')
# Resp = Resp_Rate(animate=False)
Resp.analysis_feed()
