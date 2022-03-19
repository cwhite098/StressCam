import numpy as np
import matplotlib.pyplot as plt


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

class Blink_Detector:
    def __init__(self):
        self.threshold = 10
        self.areas_list = []
        self.plotting_xdim = 100

        # init plot
        self.fig, self.axes = plt.subplots(constrained_layout = True, nrows=1, ncols=1)
        self.fig.suptitle('BLINK DETECTOR')
        self.axes.set_ylim([200,500]), self.axes.set_xlabel('Time'), self.axes.set_ylabel('Area')
        self.axes.set_title('Eye Area')
        self.line = self.axes.plot(np.linspace(0,self.plotting_xdim,self.plotting_xdim),np.zeros(self.plotting_xdim), animated=True)[0]
        self.fig.show()
        self.fig.canvas.draw()
        self.background1 = self.fig.canvas.copy_from_bbox(self.axes.bbox)


    def get_area(self, l_eye, r_eye):
        # Get the areas of both eyes and combine
        l_area = PolyArea(l_eye[:,0], l_eye[:,1])
        r_area = PolyArea(r_eye[:,0], r_eye[:,1])

        total_area = l_area + r_area

        self.areas_list.append(total_area)

        if len(self.areas_list) > self.plotting_xdim:
            self.update_plot()

    def update_plot(self):
        # Update the plot showing the eye area
        self.fig.canvas.restore_region(self.background1)
        self.line.set_ydata(self.areas_list[-(self.plotting_xdim+1):-1])  
        self.axes.draw_artist(self.line) 
        self.fig.canvas.blit(self.axes.bbox)



    