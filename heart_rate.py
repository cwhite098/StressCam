"""
Webcam Heart Rate Monitor
Gilad Oved
December 2018
"""

import numpy as np
import cv2
import sys

class Heart_Rate_Monitor:

    def __init__(self):

        # Set params for window
        self.realWidth = 640
        self.realHeight = 480
        # Params for box
        self.videoWidth = 320
        self.videoHeight = 240

        self.videoChannels = 3
        self.videoFrameRate = 15

        # Color Magnification Parameters
        self.levels = 3
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

        self.i = 0


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



    def get_bpm(self, frame, start_tuple, rect_dims):


        # Positions the detection box (just where the data is collected from, doesn't move box in display)
        #detectionFrame = frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2), :]
        detectionFrame = frame[start_tuple[0]:start_tuple[0]+self.videoHeight, start_tuple[1]:start_tuple[1]+self.videoWidth, :]


        # Construct Gaussian Pyramid
        # add it to the videoGauss buffer
        self.videoGauss[self.bufferIndex] = self.buildGauss(detectionFrame, self.levels+1)[self.levels]
        # Take the fourier transform of the frames in the buffer
        fourierTransform = np.fft.fft(self.videoGauss, axis=0)

        # Bandpass Filter - using mask defined above
        fourierTransform[self.mask == False] = 0

        # Grab a Pulse
        if self.bufferIndex % self.bpmCalculationFrequency == 0:
            self.i = self.i + 1
            for buf in range(self.bufferSize):
                # Average the fourier transform for each frame
                self.fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            # Get the max frequency
            hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
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
        #frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2), :] = outputFrame
        frame[start_tuple[0]:start_tuple[0]+self.videoHeight, start_tuple[1]:start_tuple[1]+self.videoWidth, :] = outputFrame
        
        # Displays the rectangle
        #cv2.rectangle(frame, (int(videoWidth/2) , int(videoHeight/2)), (int(realWidth-videoWidth/2), int(realHeight-videoHeight/2)), boxColor, boxWeight)
        # retangle((starting_tuple), (width, height), colour, weight)
        cv2.rectangle(frame, (start_tuple[1], start_tuple[0]), (start_tuple[1]+self.videoWidth, start_tuple[0]+self.videoHeight),
                                self.boxColor, self.boxWeight)

        # Displays the text
        if self.i > self.bpmBufferSize:
            cv2.putText(frame, "BPM: %d" % self.bpmBuffer.mean(), self.bpmTextLocation, self.font, self.fontScale, self.fontColor, self.lineType)
        else:
            cv2.putText(frame, "Calculating BPM...", self.loadingTextLocation, self.font, self.fontScale, self.fontColor, self.lineType)

        # Get BPM
        bpm = self.bpmBuffer.mean()

        return frame, bpm

