import numpy as np
import cv2
import matplotlib.pyplot as plt

class Heart_Rate_Monitor:

    def __init__(self):

        # Set params for window
        self.realWidth = 640
        self.realHeight = 480
        # Params for box
        self.videoWidth = 100
        self.videoHeight = 120

        self.videoChannels = 3
        self.videoFrameRate = 20

        # Color Magnification Parameters
        self.levels = 2
        self.alpha = 170
        self.minFrequency = 1.0
        self.maxFrequency = 2.0
        self.bufferSize = 150
        self.bufferIndex = 0

        # Output Display Parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.loadingTextLocation = (20, 30)
        self.bpmTextLocation = (self.videoWidth//2 + 5, 30)
        self.fontScale = 1
        self.fontColor = (0,0,0)
        self.lineType = 2
        self.boxColor = (0, 255, 0)
        self.boxWeight = 3

        # Initialize Gaussian Pyramid
        self.firstFrame = np.zeros((self.videoHeight, self.videoWidth, self.videoChannels))
        self.firstGauss = self.buildGauss(self.firstFrame, self.levels+1)[self.levels]
        # 4D array: time, video_x, video_y, RGB channels
        self.videoGauss = np.zeros((self.bufferSize, self.firstGauss.shape[0], self.firstGauss.shape[1], self.videoChannels))
        self.fourierTransformAvg = np.zeros((self.bufferSize))

        # Bandpass Filter for Specified Frequencies
        self.frequencies = (1.0*self.videoFrameRate) * np.arange(self.bufferSize) / (1.0*self.bufferSize)
        self.mask = (self.frequencies >= self.minFrequency) & (self.frequencies <= self.maxFrequency)

        # Heart Rate Calculation Variables
        self.bpmCalculationFrequency = 15
        self.bpmBufferIndex = 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros((self.bpmBufferSize))

        self.project_mat = [[0, 1, -1],
                            [-2, 1, 1]]

        self.i = 0
        self.h = np.linspace(-1,1,150)

        # Create figure with 2 plots
        self.fig, self.axes = plt.subplots(constrained_layout = True, nrows=3 , ncols=1)
        self.ax1 = self.axes[0]
        self.ax2 = self.axes[1]
        self.ax3 = self.axes[2]
        # Set params for BPM graph
        self.ax1.set_ylim([50,120]), self.ax1.set_xlabel('Time'), self.ax1.set_ylabel('BPM')
        self.ax1.set_title('Beats Per Minute')
        # Set params for frequency spectrum
        self.ax2.set_ylim([0,30]), self.ax2.set_xlabel('Frequency'), self.ax2.set_ylabel('Magnitude')
        self.ax2.set_title('Fourier Transform')
        # Set params for POS signal
        self.ax3.set_ylim([-0.01,0.01]), self.ax3.set_xlabel('Time'), self.ax3.set_ylabel('POS Signal')
        self.ax3.set_title('POS Signal')
        # Set the lines to be used for the plots
        self.spec = self.ax2.plot(self.frequencies, self.fourierTransformAvg, animated=True)[0]
        self.line = self.ax1.plot(np.linspace(0,100,100),np.zeros(100), animated=True)[0]
        self.linePOS = self.ax3.plot(np.linspace(-1,1,150), np.zeros(150), animated=True)[0]
        self.fig.show()
        self.fig.canvas.draw()
        self.background1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        self.background2 = self.fig.canvas.copy_from_bbox(self.ax2.bbox)
        self.background3 = self.fig.canvas.copy_from_bbox(self.ax3.bbox)

        # Lists to store bpm data
        self.bpm_list = []
        self.total_bpm_list = []


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

    def make_plots(self):
        
        self.fig.canvas.restore_region(self.background1)
        self.fig.canvas.restore_region(self.background2)
        self.fig.canvas.restore_region(self.background3)
        self.line.set_ydata(self.bpm_list)
        self.spec.set_ydata(np.real(self.POS_fourier))
        self.linePOS.set_ydata(self.h)
 
        self.ax1.draw_artist(self.line)
        self.ax2.draw_artist(self.spec)
        self.ax3.draw_artist(self.linePOS)
        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)
        self.fig.canvas.blit(self.ax3.bbox)


    def get_bpm(self, frame, start_tuple, rect_dims):

        # Positions the detection box (just where the data is collected from, doesn't move box in display)
        detectionFrame = frame[start_tuple[0]:start_tuple[0]+self.videoHeight, start_tuple[1]:start_tuple[1]+self.videoWidth, :]

        # Construct Gaussian Pyramid
        # add it to the videoGauss buffer
        self.videoGauss[self.bufferIndex] = self.buildGauss(detectionFrame, self.levels+1)[self.levels]

        # Get the average colour signals from buffer
        C = self.videoGauss.mean((1,2))

        # Take the fourier transform of the frames in the buffer
        fourierTransform = np.fft.fft(self.videoGauss, axis=0)

        # Bandpass Filter - using mask defined above
        fourierTransform[self.mask == False] = 0

        # Grab a Pulse
        if self.bufferIndex % self.bpmCalculationFrequency == 0:
            self.i = self.i + 1

            # temporal normalisation for C ???
            mean = C.mean(1)
            C[:,0] = C[:,0] / mean
            C[:,1] = C[:,1] / mean
            C[:,2] = C[:,2] / mean

            S = np.matmul(self.project_mat, np.transpose(C))

            coeff = (np.std(S[0,:])/np.std(S[1,:]))
            self.h = S[0,:] + (coeff*S[1,:])
            self.h = self.h - np.mean(self.h)
            print(self.h)
            self.POS_fourier = np.fft.fft(self.h)
            self.POS_fourier[self.mask == False] = 0

            for buf in range(self.bufferSize):
                # Average the fourier transform for each frame
                self.fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            # Get the max frequency
            #hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
            hz = self.frequencies[np.argmax(self.POS_fourier)]
            bpm = 60.0 * hz
            self.bpmBuffer[self.bpmBufferIndex] = bpm
            self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize

        # Amplify
        # Inverse FT
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * self.alpha # amplify by a factor of alpha

        # Reconstruct Resulting detection Frame
        filteredFrame = self.reconstructFrame(filtered, self.bufferIndex, self.levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        self.bufferIndex = (self.bufferIndex + 1) % self.bufferSize

        # Set the detection region in total frame to the output of the fourier stuff
        frame[start_tuple[0]:start_tuple[0]+self.videoHeight, start_tuple[1]:start_tuple[1]+self.videoWidth, :] = outputFrame
        
        # Displays the rectangle
        cv2.rectangle(frame, (start_tuple[1], start_tuple[0]), (start_tuple[1]+self.videoWidth, start_tuple[0]+self.videoHeight),
                                self.boxColor, self.boxWeight)

        # Displays the text
        if self.i > self.bpmBufferSize:
            cv2.putText(frame, "BPM: %d" % self.bpmBuffer.mean(), self.bpmTextLocation, self.font, self.fontScale, self.fontColor, self.lineType)
            
            # Get BPM
            bpm = self.bpmBuffer.mean()
            self.bpm_list.append(bpm)
            self.total_bpm_list.append(bpm)
            if len(self.bpm_list) == 101:
                self.bpm_list.pop(0)
                self.make_plots()
        else:
            cv2.putText(frame, "Calculating BPM...", self.loadingTextLocation, self.font, self.fontScale, self.fontColor, self.lineType)
            bpm = 0


        return frame, bpm

