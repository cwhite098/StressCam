import numpy as np
import matplotlib.pyplot as plt


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def get_dist(total_landmarks, idx):
    poi = total_landmarks[idx]
    dist = np.linalg.norm(poi[0]-poi[1])
    return dist

def get_avg_position(total_landmarks, idx):
    # Get the avg position of a selection of landmarks
    landmarks = total_landmarks[idx]
    avg_x = landmarks[:,0].mean()
    avg_y = landmarks[:,1].mean()

    return [avg_x, avg_y]


class Eyes_Mouth_Detector:
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

        self.mouth_ratios_list = []

        self.l_eyebrow_ratio_list = []
        self.r_eyebrow_ratio_list = []

        # init plot
        self.fig, self.axes = plt.subplots(constrained_layout = True, nrows=3, ncols=1)
        self.fig.suptitle('EYE(BROW)S and MOUTH MONITORING')
        # Set up eye ratio tracking plot
        self.ax1 = self.axes[0]
        self.ax1.set_ylim([1,5]), self.ax1.set_xlabel('Time'), self.ax1.set_ylabel('Eye Ratio')
        self.ax1.set_title('Eye Ratio')
        self.line = self.ax1.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), animated=True)[0]
        self.line2 = self.ax1.axhline(self.threshold, linestyle='--', c='r')
        # Set up mouth ratio tracking plot
        self.ax2 = self.axes[1]
        self.ax2.set_ylim([0,4]), self.ax2.set_xlabel('Time'), self.ax2.set_ylabel('Mouth Ratio')
        self.ax2.set_title('Mouth Ratio')
        self.mouth_line = self.ax2.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), animated=True)[0]
        # Set up eyebrow ration tracking plot
        self.ax3 = self.axes[2]
        self.ax3.set_ylim([0.05,0.3]), self.ax3.set_xlabel('Time'), self.ax3.set_ylabel('Eyebrow Ratio')
        self.ax3.set_title('Eyebrow Ratio')
        self.leyebrow_line = self.ax3.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), animated=True)[0]
        self.reyebrow_line = self.ax3.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), animated=True)[0]

        self.fig.show()
        self.fig.canvas.draw()
        self.background1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        self.background2 = self.fig.canvas.copy_from_bbox(self.ax2.bbox)
        self.background3 = self.fig.canvas.copy_from_bbox(self.ax3.bbox)

        # Points in mesh for horizontal and vertical extents of the eyes
        self.l_eye_vert = [159, 145]
        self.l_eye_hor = [39, 13]
        self.r_eye_vert = [384, 374]
        self.r_eye_hor = [263, 362]
        self.mouth_vert = [17,0]
        self.mouth_hor = [61, 291]

        # Points for eye outlines for calculating average eye position (position of eye centre)
        self.leye_idx = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        self.reye_idx = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

        # Points for eyebrow averaging
        self.leyebrow_idx = [55, 107, 66, 65, 105, 52, 63, 53, 70, 46]
        self.reyebrow_idx = [336, 285, 296, 295, 334, 282, 293, 283, 300, 276]
        self.face_width_idx = [454, 234]

    def get_ratio(self, total_landmarks):

        # Get the ratios of both eyes and combine
        r_vert = get_dist(total_landmarks, self.r_eye_vert)
        r_hor = get_dist(total_landmarks, self.r_eye_hor)
        r_ratio = r_hor/r_vert

        l_vert = get_dist(total_landmarks, self.l_eye_vert)
        l_hor = get_dist(total_landmarks, self.l_eye_hor)
        l_ratio = l_hor/l_vert

        ratio = (l_ratio + r_ratio)/2
        self.ratios_list.append(ratio)

        # Get ratio for mouth openness
        mouth_vert = get_dist(total_landmarks, self.mouth_vert)
        mouth_hor = get_dist(total_landmarks, self.mouth_hor)
        mouth_ratio = mouth_hor/mouth_vert
        self.mouth_ratios_list.append(mouth_ratio)

        # Get ratio for eyebrow raised-ness
        l_eye_avg = np.array(get_avg_position(total_landmarks, self.leye_idx))
        l_eyebrow_avg = np.array(get_avg_position(total_landmarks, self.leyebrow_idx))
        l_eyebrow_height = np.linalg.norm(l_eye_avg - l_eyebrow_avg)

        r_eye_avg = np.array(get_avg_position(total_landmarks, self.reye_idx))
        r_eyebrow_avg = np.array(get_avg_position(total_landmarks, self.reyebrow_idx))
        r_eyebrow_height = np.linalg.norm(r_eye_avg - r_eyebrow_avg)

        face_width = get_dist(total_landmarks, self.face_width_idx)
        l_eyebrow_ratio = l_eyebrow_height/face_width
        r_eyebrow_ratio = r_eyebrow_height/face_width
        self.l_eyebrow_ratio_list.append(l_eyebrow_ratio)
        self.r_eyebrow_ratio_list.append(r_eyebrow_ratio)

        # Update plots if enough data has been collected
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
        self.fig.canvas.restore_region(self.background2)
        self.fig.canvas.restore_region(self.background3)

        self.line.set_ydata(self.ratios_list[-(self.plotting_xdim+1):-1])  
        self.mouth_line.set_ydata(self.mouth_ratios_list[-(self.plotting_xdim+1):-1])
        self.reyebrow_line.set_ydata(self.r_eyebrow_ratio_list[-(self.plotting_xdim+1):-1])
        self.leyebrow_line.set_ydata(self.l_eyebrow_ratio_list[-(self.plotting_xdim+1):-1])

        self.ax1.draw_artist(self.line) 
        self.ax2.draw_artist(self.mouth_line)
        self.ax3.draw_artist(self.reyebrow_line) 
        self.ax3.draw_artist(self.leyebrow_line) 

        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)
        self.fig.canvas.blit(self.ax3.bbox)



    