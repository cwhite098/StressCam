from heapq import nlargest
from shutil import ExecError
from matplotlib import animation
from scipy import signal
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import perf_counter, time
from threading import Thread
import mediapipe as mp

class Resp_Rate:
    def __init__(self, fps,  cap=None, ROI_buffer_y_upper=70, ROI_buffer_y_lower=30, refresh_rate=30, win_size=300, vid=None, ROI=None):
        self.vid = vid
        self.ROI = ROI
        self.First = True
        self.refresh_rate = refresh_rate 
        self.win_size = win_size # number of frame on plot
        self.fps = fps
        self.p_y_f = []
        self.p_norm = np.array([0] * self.win_size)
        self.cap = cap
        self.ROI_buffer_y_upper = ROI_buffer_y_upper
        self.ROI_buffer_y_lower = ROI_buffer_y_lower

        self.b, self.a = signal.butter(3, (0.05, 2), fs=15 if self.fps<4 else self.fps, btype='bandpass', analog=False)

        '''Initialise plotting canvas'''
        # plt.ion
        '''
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlim(0, self.win_size)
        self.ax.set_ylim(-2.5, 2.5)
        self.sig, = self.ax.plot(self.p_norm)
        self.rate = self.ax.text(0.5, 2.3, '')
        self.fps_text = self.ax.text(0.5, 2, '')
        self.fig.show()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        '''

    def find_ROI(self, img, mp_holistic, holistic):
        '''Find ROI'''
        image_height, image_width, _ = img.shape
        results = holistic.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        if results.pose_landmarks:
            shoulder_xs, shoulder_ys = (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width,
            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width), (
                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height,
                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height)

            if max(shoulder_ys)+self.ROI_buffer_y_lower > image_height:
                self.ROI_buffer_y_lower = image_height - max(shoulder_ys)

            self.ROI = [int(shoulder_xs[1]), int(min(shoulder_ys)-self.ROI_buffer_y_upper),
            int(shoulder_xs[0]), int(max(shoulder_ys)+self.ROI_buffer_y_lower)]
            self.y_len, self.x_len = (int(self.ROI[3]-self.ROI[1]), int(shoulder_xs[0]-shoulder_xs[1]))
            self.First = False
        else:
            print('Cannot find shoulders, SKIPPING')
            self.First = True
            self.y_len = 1

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
        #sd_list = [np.std(self.p_y_f[:, y]) for y in range(self.y_len)] # (y_lne x 1)
        sd_list = np.std(self.p_y_f, axis=0)

        #largest_sd = nlargest(int(0.05*self.y_len), sd_list)
        idx = np.argsort(sd_list)[-int(0.05*self.y_len):]

        p_5per = self.p_y_f[:, idx]
        p_f = np.mean(p_5per, 1)
        
        y = signal.lfilter(self.b, self.a, p_f)

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

    def vid_feed_per_frame_analysis(self, frame, fps, frame_num, mp_holistic, holistic):
        if self.First:
            '''Initialise ROI'''
            self.find_ROI(frame, mp_holistic, holistic)
            '''Initialise matrices'''
            self.p_y_f = np.array([[0] * self.y_len] * frame_num)
            self.p_norm = np.array([0] * frame_num)
            self.init, self.f = True, 1
        if not self.First:
            '''Process pre recorded video'''
            cropped = frame[self.ROI[1]:self.ROI[3], self.ROI[0]:self.ROI[2]]
            self.resp_pattern_per_frame(cropped, fps)
    

    
    def live_feed_per_frame_animate(self, i):
        '''Function for live feed with animated bmp graph'''
        start = perf_counter()
        for i in range(self.refresh_rate):
            _, cur = self.cap.read()
            cropped = cur[self.ROI[1]:self.ROI[3], self.ROI[0]:self.ROI[2]]
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
        cropped = cur[self.ROI[1]:self.ROI[3], self.ROI[0]:self.ROI[2]]
        self.resp_pattern_per_frame(cropped, fps)

        if self.init is True and self.f>=self.refresh_rate:
            self.bpm = self.resp_rate(self.win_size, fps)
            text = 'bpm: '+str(round(self.bpm, 2))
            self.init, self.f = False, 1
        elif self.f>=self.refresh_rate:
            self.bpm = self.resp_rate(self.win_size, fps)
            text = 'bpm: '+str(round(self.bpm, 2))
            self.f = 1
        elif self.init is True:
            text = 'Calculating bpm...'
            self.f += 1
        else:
            text = 'bpm: '+str(round(self.bpm, 2))
            self.f += 1
        cropped = cv.putText(cropped, text, (5, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
        # print(self.f)

        self.fig.canvas.restore_region(self.background)
        self.sig.set_ydata(self.p_norm)
        # self.rate.set_text('bpm: %.2f' % bpm)
        # self.fps_text.set_text('fps: %.2f' % fps)
        self.ax.draw_artist(self.sig)
        # self.ax.draw_artist(self.fps_text)
        self.fig.canvas.blit(self.ax.bbox)
        return cropped
    
    def live_feed_per_frame(self):
        # 17 sec ~ 3 cycle
        while 1:
            start = perf_counter()
            _, cur = self.cap.read()
            cv.imshow('frame', cur)
            if cv.waitKey(1) == ord('q'):
                break
            cropped = cur[int(self.ROI[1]):int(self.ROI[1]+self.ROI[3]), 
            int(self.ROI[0]):int(self.ROI[0]+self.ROI[2])]
            self.resp_pattern_per_frame(cropped)
            print(self.p_norm[-1])
    
    def live_feed_init(self):
        _, cur = self.cap.read()
        '''Initialise ROI'''
        self.find_ROI(cur)
        '''Initialise matrices'''
        if self.vid is None:
            self.p_y_f = np.array([[0] * self.y_len] * self.win_size)
        self.init, self.f = True, 1
        '''Initialise matrix p to prevent row having the same value'''
        start = perf_counter()
        for i in range(10):
            _, cur = self.cap.read()
            cropped = cur[self.ROI[1]:self.ROI[3], self.ROI[0]:self.ROI[2]]
            self.p_y_f[0:-1,:] = self.p_y_f[1:,:] # move values forward by one row
            p_f = np.sum(np.sum(cropped, axis=2), axis=1)/self.x_len
            p_f_detrend = signal.detrend(p_f, type='constant')
            self.p_y_f[-1,:] = p_f_detrend # (refresh_rate(n frames) x y_len)
        self.fps = 10 / (perf_counter() - start)

    def live_feed(self):
        '''Process online input'''
        self.cap = cv.VideoCapture()
        self.cap.open(0, cv.CAP_DSHOW)
        self.live_feed_init()
        self.animate_init()
        ani = FuncAnimation(self.fig, self.live_feed_per_frame_animate)
        plt.show()
        self.cap.release()
        cv.destroyAllWindows()

    def analysis_feed(self):
        if self.vid is None:
            self.live_feed()
        else:
            # self.vid_feed()
            pass

# Resp = Resp_Rate(vid='Recording2_Trim.mp4', ROI=(534, 465, 224, 173))
# Resp = Resp_Rate()
# p_list = Resp.analysis_feed()
