import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import heartpy as hp

class Heart_Rate_Monitor:

    def __init__(self, fps, videoWidth, videoHeight):

        # Set params for window
        self.realWidth = 640
        self.realHeight = 480
        # Params for box
        self.videoWidth = videoWidth
        self.videoHeight = videoHeight

        self.videoChannels = 3
        self.videoFrameRate = fps

        # Color Magnification Parameters
        self.levels = 2
        self.alpha = 170
        self.minFrequency = 0.5 # maybe reduce this (1hz = 60bpm)
        self.maxFrequency = 2.0
        self.bufferSize = 100
        self.bufferIndex = 0

        # Output Display Parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.loadingTextLocation = (20, 30)
        self.bpmTextLocation = (self.videoWidth//2 + 5, 30)
        self.fontScale = 1
        self.fontColor = (255,255,255)
        self.lineType = 2
        self.boxColor = (0, 255, 0)
        self.boxWeight = 3

        # Heart Rate Calculation Variables
        self.bpmCalculationBuffer = 600
        self.bpmBufferSize = 20
        self.total_bpm = []

        # POS method variables
        self.project_mat = [[0, 1, -1],
                            [-2, 1, 1]]
        self.bufferSizePOS = 32
        self.bufferPOS = []
        self.totalPOS = []

        self.h = np.linspace(-1,1,150)

        # Bandpass Filter for Specified Frequencies
        self.frequencies = np.linspace(0,4,self.bufferSizePOS)
        mask_freqs = np.linspace(0,4,self.bufferSize)
        self.mask = (mask_freqs >= self.minFrequency) & (mask_freqs <= self.maxFrequency)

        # Create figure with 2 plots
        self.fig, self.axes = plt.subplots(constrained_layout = True, nrows=3 , ncols=1)
        self.fig.suptitle('HEART RATE MONITOR')
        self.ax1 = self.axes[0]
        self.ax2 = self.axes[1]
        self.ax3 = self.axes[2]
        # Set params for BPM graph
        self.ax1.set_ylim([50,120]), self.ax1.set_xlabel('Time'), self.ax1.set_ylabel('BPM')
        self.ax1.set_title('Beats Per Minute')
        # Set params for frequency spectrum
        self.ax2.set_ylim([0,0.1]), self.ax2.set_xlabel('Frequency'), self.ax2.set_ylabel('Magnitude')
        self.ax2.set_title('Fourier Transform')
        # Set params for POS signal
        self.ax3.set_ylim([-0.02,0.02]), self.ax3.set_xlabel('Time'), self.ax3.set_ylabel('POS Signal')
        self.ax3.set_title('POS Signal')
        # Set the lines to be used for the plots
        self.spec = self.ax2.plot(self.frequencies, np.zeros(self.bufferSizePOS), animated=True)[0]
        self.line = self.ax1.plot(np.linspace(0,self.bpmBufferSize,self.bpmBufferSize),np.zeros(self.bpmBufferSize), animated=True)[0]
        self.linePOS = self.ax3.plot(np.linspace(-1,1,self.bufferSizePOS), np.zeros(self.bufferSizePOS), animated=True)[0]
        self.fig.show()
        self.fig.canvas.draw()
        self.background1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        self.background2 = self.fig.canvas.copy_from_bbox(self.ax2.bbox)
        self.background3 = self.fig.canvas.copy_from_bbox(self.ax3.bbox)

        # Lists to store bpm data and HRV data
        self.bpm_list = []
        self.ibi_list = []
        self.sdnn_list = []
        self.sdsd_list = []
        self.rmssd_list = []
        self.pnn20_list = []
        self.pnn50_list = []
        self.hr_mad_list = []
        self.sd1_list = []
        self.sd2_list = []
        self.s_list = []
        self.sd1_sd2_list = []
        self.BR_list = []



    # Helper Methods - modify these to get variable rect size
    def buildGauss(self, frame, levels):
        pyramid = [frame]
        for level in range(levels):
            # Downsamples the fram using a Gaussian pyramid
            # https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstructFrame(self, pyramid, index, levels):
        # Reconstruct the green box to be put in the total frame
        filteredFrame = pyramid[index]
        for level in range(levels):
            # Up sample using Laplacian pyramid
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:self.videoHeight, :self.videoWidth]
        return filteredFrame

    def plot_bpm(self):
        
        self.fig.canvas.restore_region(self.background1)
        self.line.set_ydata(self.bpm_list)  
        self.ax1.draw_artist(self.line) 
        self.fig.canvas.blit(self.ax1.bbox)
        

    def plot_POS(self):

        self.fig.canvas.restore_region(self.background2)
        self.fig.canvas.restore_region(self.background3)

        self.spec.set_ydata(np.real(self.POS_fourier))
        self.linePOS.set_ydata(self.h)

        self.ax2.draw_artist(self.spec)
        self.ax3.draw_artist(self.linePOS)

        self.fig.canvas.blit(self.ax2.bbox)
        self.fig.canvas.blit(self.ax3.bbox)

    def get_bpm(self, frame):

        R = frame[:,:,0].flatten()
        G = frame[:,:,1].flatten()
        B = frame[:,:,2].flatten()
        # Remove black areas and then get colour averages
        R_mean = R[R!=0].mean()
        G_mean = G[G!=0].mean()
        B_mean = B[B!=0].mean()

        C = np.array([R_mean, G_mean, B_mean])
        meanC = C.mean()
        C[:] = C[:] / meanC
        S = np.matmul(self.project_mat, C)
        self.bufferPOS.append(S)
        # Add to list for data saving
        self.totalPOS.append(S)

        filtered_POS = np.array([])

        # if the POS signal buffer is above the set threshold
        if len(self.bufferPOS) > self.bufferSizePOS:

            # This section of code is just to plot the POS and fourier - is it necessary?
            # Could wrap this as a separate function
            signal = np.transpose(self.bufferPOS[-(self.bufferSizePOS+1):-1])

            coeff = (np.std(signal[0,:])/np.std(signal[1,:]))
            self.h = signal[0,:] + (coeff*signal[1,:])
            self.h = self.h - np.mean(self.h)

            # Get POS fourier for plot
            self.POS_fourier = np.fft.fft(self.h)
            self.plot_POS()

            # if the buffer is greater than the min data needed to get bpm
            if len(self.bufferPOS) > self.bpmCalculationBuffer:

                signal = np.transpose(self.bufferPOS[-(self.bpmCalculationBuffer+1):-1])

                coeff = (np.std(signal[0,:])/np.std(signal[1,:]))
                signal2 = signal[0,:] + (coeff*signal[1,:])
                signal2 = signal2 - np.mean(self.h)

                filtered_POS = hp.filtering.filter_signal(signal2, cutoff=(self.minFrequency, self.maxFrequency),
                                                                sample_rate=self.videoFrameRate, order=3, filtertype='bandpass')
                workingdata, measures = hp.process(filtered_POS, sample_rate=self.videoFrameRate)
                bpm_hp = measures['bpm']
                self.ibi_list.append(measures['ibi'])
                self.sdnn_list.append(measures['sdnn'])
                self.sdsd_list.append(measures['sdsd'])
                self.rmssd_list.append(measures['rmssd'])
                self.pnn20_list.append(measures['pnn20'])
                self.pnn50_list.append(measures['pnn50'])
                self.hr_mad_list.append(measures['hr_mad'])
                self.sd1_list.append(measures['sd1'])
                self.sd2_list.append(measures['sd2'])
                self.s_list.append(measures['s'])
                self.sd1_sd2_list.append(measures['sd1/sd2'])
                self.BR_list.append(measures['breathingrate'])

                self.total_bpm.append(bpm_hp)

        # Displays the text if first bpm calc has been done
        if self.total_bpm:
            # takes last calculated bpm value (every 20 frames (1 second))
            try:
                cv2.putText(frame, "BPM: %d" % self.total_bpm[-1], self.bpmTextLocation, self.font, self.fontScale, self.fontColor, self.lineType)
            except ValueError:
                cv2.putText(frame, "BPM: %d" % 0, self.bpmTextLocation, self.font, self.fontScale, self.fontColor, self.lineType)

            # Get BPM subset and update plot
            if len(self.total_bpm) > self.bpmBufferSize:
                self.bpm_list = self.total_bpm[-(self.bpmBufferSize+1):-1]
                self.plot_bpm()

        else:
            cv2.putText(frame, "Calculating BPM...", self.loadingTextLocation, self.font, self.fontScale, self.fontColor, self.lineType)
            bpm = 0


        return frame


    def save_data(self, path):

        # Get POS signal from S
        signal = np.transpose(self.totalPOS)
        coeff = (np.std(signal[0,:])/np.std(signal[1,:]))
        POS_signal = signal[0,:] + (coeff*signal[1,:])
        POS_signal = POS_signal - np.mean(POS_signal)

        # Construct and save dataframe as csv
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df1['BPM'] = self.total_bpm
        df1['ibi'] = self.ibi_list
        df1['sdnn'] = self.sdnn_list 
        df1['sdsd'] = self.sdsd_list 
        df1['rmssd'] = self.rmssd_list 
        df1['pnn20'] = self.pnn20_list 
        df1['pnn50'] = self.pnn50_list 
        df1['hr_mad'] = self.hr_mad_list 
        df1['sd1'] = self.sd1_list 
        df1['sd2'] = self.sd2_list 
        df1['s'] = self.s_list 
        df1['sd1/sd2'] = self.sd1_sd2_list 
        df1['BR'] = self.BR_list

        df2['POS'] = POS_signal

        new = pd.concat([df1, df2], axis=1) 

        new.to_csv(path)
