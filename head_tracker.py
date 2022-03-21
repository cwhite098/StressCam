import numpy as np
import matplotlib.pyplot as plt
import cv2


class Head_Tracker:

    def __init__(self, image_width, image_height):

        self.image_width = image_width
        self.image_height = image_height

        self.idx = [1, 33, 262, 61, 291, 199]
        
        self.x_list = []
        self.y_list = []
        self.z_list = []

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

        # init plot
        self.fig, self.ax1 = plt.subplots(constrained_layout = True, nrows=1, ncols=1)
        self.fig.suptitle('HEAD TRACKER')
        # Set up eye ratio tracking plot
        self.ax1.set_ylim([-180,180]), self.ax1.set_xlabel('Time'), self.ax1.set_ylabel('Angle')
        self.ax1.set_title('XYZ of Head')
        self.line1 = self.ax1.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), c='r', label='x', animated=True)[0]
        self.line2 = self.ax1.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), c='g', label='y', animated=True)[0]
        self.line3 = self.ax1.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), c='b', label='z', animated=True)[0]
        self.ax1.legend()

        self.fig.show()
        self.fig.canvas.draw()
        self.background1 = self.fig.canvas.copy_from_bbox(self.ax1.bbox)

    def update_plots(self):

        self.fig.canvas.restore_region(self.background1)
        self.line1.set_ydata(self.x_list[-(self.plotting_xdim+1):-1])
        self.line2.set_ydata(self.y_list[-(self.plotting_xdim+1):-1])
        self.line3.set_ydata(self.z_list[-(self.plotting_xdim+1):-1])

        self.ax1.draw_artist(self.line1)
        self.ax1.draw_artist(self.line2)
        self.ax1.draw_artist(self.line3)

        self.fig.canvas.blit(self.ax1.bbox)




    def get_angular_position(self, total_landmarks_3d, image):
        
        points_of_interest = total_landmarks_3d[self.idx]

        # Get nose coords
        nose_2d = points_of_interest[0, :2]
        nose_3d = points_of_interest[0,:]

        # Get np arrays of 2d and 3d points
        self.face_2d = np.array(points_of_interest[:,:2], dtype=np.float64)
        # Remove that 3000 that multiplied the z coord in main
        points_of_interest[:,2] = points_of_interest[:,2]/3000
        self.face_3d = np.array(points_of_interest, dtype=np.float64)

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
        p2 = (int(nose_2d[0] - x * 10) , int(nose_2d[1] + y * 10))
        
        cv2.line(image, p1, p2, (255, 0, 0), 3)

        if len(self.x_list) > self.plotting_xdim:
            self.update_plots()

        return image




        

