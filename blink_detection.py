import numpy as np
import matplotlib.pyplot as plt


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_dist(total_landmarks, idx):
    poi = total_landmarks[idx]
    dist = np.linalg.norm(poi[0]-poi[1])
    return dist

class Blink_Detector:
    def __init__(self):
        # Threshold for the ratio
        self.threshold = 3

        # Counter for number of blinks and total number of frames
        self.blink_counter = 0
        self.frame_counter = 0
        # The frame numbers at which the blinks occur
        self.blink_frames_list = []

        self.ratios_list = []
        self.plotting_xdim = 100

        # init plot
        self.fig, self.axes = plt.subplots(constrained_layout = True, nrows=1, ncols=1)
        self.fig.suptitle('BLINK DETECTOR')
        self.axes.set_ylim([1,5]), self.axes.set_xlabel('Time'), self.axes.set_ylabel('Area')
        self.axes.set_title('Eye Area')
        self.line = self.axes.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), animated=True)[0]
        self.line2 = self.axes.axhline(self.threshold, linestyle='--', c='r')
        self.fig.show()
        self.fig.canvas.draw()
        self.background1 = self.fig.canvas.copy_from_bbox(self.axes.bbox)

        # Points in mesh for horizontal and vertical extents of the eyes
        self.l_eye_vert = [159, 145]
        self.l_eye_hor = [39, 13]
        self.r_eye_vert = [384, 374]
        self.r_eye_hor = [263, 362]


    def get_ratio(self, total_landmarks):
        # Get the areas of both eyes and combine

        r_vert = get_dist(total_landmarks, self.r_eye_vert)
        r_hor = get_dist(total_landmarks, self.r_eye_hor)
        r_ratio = r_hor/r_vert

        l_vert = get_dist(total_landmarks, self.l_eye_vert)
        l_hor = get_dist(total_landmarks, self.l_eye_hor)
        l_ratio = l_hor/l_vert

        ratio = (l_ratio + r_ratio)/2
        self.ratios_list.append(ratio)

        if len(self.ratios_list) > self.plotting_xdim:
            self.update_plot()

        # If threshold exceeded, record blink
        if ratio > self.threshold:
            self.blink_counter += 1
            self.blink_frames_list.append(self.frame_counter)

        self.frame_counter += 1

    def update_plot(self):
        # Update the plot showing the eye area
        self.fig.canvas.restore_region(self.background1)
        self.line.set_ydata(self.ratios_list[-(self.plotting_xdim+1):-1])  
        self.axes.draw_artist(self.line) 
        self.fig.canvas.blit(self.axes.bbox)



    