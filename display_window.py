from tkinter import *

class Window:

    def __init__(self):

        self.root=Tk()
        self.root.title('Measurements Window')


    def run(self):

        info=[
            'Heartrate:         71.0',
            'eyes_y:            0.232',
            'eyes_x:            0.913',
            'r_eyebrow:         0.154',
            'l_eyebrow:         0.157',
            'mouth:             2.14',
            'head pitch:        7.492',
            'head yaw:          -3.36',
            'head translation:  4.71',
            'resp signal:       1.38'
        ]
        len_info = len(info)
        for idx,label in enumerate(info):
            l = Label(self.root, text=label, width=100, anchor='w',font='TkFixedFont')
            l.pack(fill='both')
            l.place(relx=0, rely=(idx/len_info))
        self.root.mainloop()




if __name__ == '__main__':
    st = Window()
    st.run()
