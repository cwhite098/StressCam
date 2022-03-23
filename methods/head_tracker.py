import numpy as np
import matplotlib.pyplot as plt
import cv2


class Head_Tracker:

    def __init__(self, image_width, image_height, show_plots = True):
        
        self.show_plots = show_plots

        self.image_width = image_width
        self.image_height = image_height

        self.idx = [1, 33, 262, 61, 291, 199]
        
        self.x_list = []
        self.y_list = []
        self.z_list = []

        self.prev_nose_2d = False
        self.translation_list = []

        self.face_2d = []
        self.face_3d = []

        # Get the camera matrix - might change with calibration (?)
        self.focal_length = 1 * self.image_width
        self.cam_mat = np.array([[self.focal_length, 0, self.image_height/2],
                                 [0, self.focal_length,  self.image_width/2],
                                 [0,                 0,                   1]])

        # Get the distortion matrix - might change with calibration (?)
        self.dist_mat = np.zeros((4, 1), dtype=np.float64)

        self.plotting_xdim = 10

        if self.show_plots:
            # init plot
            self.fig, self.axes = plt.subplots(constrained_layout = True, nrows=2, ncols=1)
            self.fig.suptitle('HEAD TRACKER')
            # Set up eye ratio tracking plot
            self.ax1 = self.axes[0]
            self.ax1.set_ylim([-30,30]), self.ax1.set_xlabel('Time'), self.ax1.set_ylabel('Angle')
            self.ax1.set_title('Pitch/Yaw of Head')
            self.line1 = self.ax1.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), c='r', label='Pitch', animated=True)[0]
            self.line2 = self.ax1.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), c='g', label='Yaw', animated=True)[0]
            self.ax1.legend()

            self.ax2 = self.axes[1]
            self.ax2.set_ylim([0,50]), self.ax2.set_xlabel('Time'), self.ax2.set_ylabel('Angle')
            self.ax2.set_title('Inter-Frame-Translation')
            self.line3 = self.ax2.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim),animated=True)[0]

            self.fig.show()
            self.fig.canvas.draw()
            self.background1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
            self.background2 = self.fig.canvas.copy_from_bbox(self.ax2.bbox)

    def update_plots(self):

        self.fig.canvas.restore_region(self.background1)
        self.fig.canvas.restore_region(self.background2)
        self.line1.set_ydata(self.x_list[-(self.plotting_xdim+1):-1])
        self.line2.set_ydata(self.y_list[-(self.plotting_xdim+1):-1])
        self.line3.set_ydata(self.translation_list[-(self.plotting_xdim+1):-1])

        self.ax1.draw_artist(self.line1)
        self.ax1.draw_artist(self.line2)
        self.ax2.draw_artist(self.line3)

        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)




    def get_angular_position(self, landmarks, image):
        
        self.face_3d = []
        self.face_2d = []

        for idx, lm in enumerate(landmarks):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * self.image_width, lm.y * self.image_height)
                    nose_3d = (lm.x * self.image_width, lm.y * self.image_height, lm.z * 3000)

                    # Get inter-frame-translation of nose coords
                    if not self.prev_nose_2d:
                        inter_frame_translation = 0
                        self.prev_nose_2d = nose_2d
                    else:
                        inter_frame_translation = np.linalg.norm(np.array(nose_2d) - np.array(self.prev_nose_2d))
                        self.prev_nose_2d = nose_2d
                    self.translation_list.append(inter_frame_translation)

                x, y = int(lm.x * self.image_width), int(lm.y * self.image_height)

                # Get the 2D Coordinates
                self.face_2d.append([x, y])

                # Get the 3D Coordinates
                self.face_3d.append([x, y, lm.z])       
            
        # Convert it to the NumPy array
        self.face_2d = np.array(self.face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        self.face_3d = np.array(self.face_3d, dtype=np.float64)
        

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(self.face_3d, self.face_2d, self.cam_mat, self.dist_mat)

        # Get rotation matrix + Jacobian
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        x = angles[0]*360
        y = angles[1]*360
        z = angles[2]*360

        self.x_list.append(x)
        self.y_list.append(y)
        self.z_list.append(z)

        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, self.cam_mat, self.dist_mat)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
        
        cv2.line(image, p1, p2, (255, 0, 0), 3)

        if len(self.x_list) > self.plotting_xdim and self.show_plots:
            self.update_plots()

        return image




        

