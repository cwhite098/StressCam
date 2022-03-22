from tkinter import *
import random
from time import perf_counter, sleep
from playsound import playsound
import multiprocessing

class stroop_test:


    def __init__(self, timeout=2, sound=False):

        self.timeout = timeout
        self.score = 0
        self.attempts = 0
        self.colours = {'red':'r', 'blue':'b', 'green':'g', 'yellow':'y'}
        self.root = Tk()
        self.times_pressed = 0
        self.word, self.colour = self.stimulus()
        self.label = Label(self.root, text=self.word, fg=self.colour, font=("Comic Sans MS", 200), width =8, height=2)
        self.label.pack()
        self.img = PhotoImage(file='big-red-cross.png')
        self.image = Label(self.root, image=self.img, width =8000, height=2000)
        self.score_label = Label(self.root, text=f'Score: {self.score} / {self.attempts}',
                                 font=("Comic Sans MS", 50), width =12, height=1)
        self.score_label.place(relx = 0.6, rely = 0)
        self.start = True
        if sound:
            self.sound_thread = multiprocessing.Process(target=playsound, args=('rising.mp3',))
            self.sound_thread.start()

    def stimulus(self):
        self.times_pressed = 0
        self.word = random.choice(list(self.colours))
        self.same = random.choices([True, False], weights=(0.1, 0.9))[0]
        if self.same:
            return (self.word, self.word)
        list(self.colours).remove(self.word)
        self.colour = random.choice(list(self.colours))
        return self.word, self.colour

    def key_pressed(self, event):
        self.times_pressed += 1
        print(self.times_pressed)
        if self.times_pressed == 1:
            if self.colours[self.colour] == event.char:
                self.score += 1
                self.attempts += 1
            else:
                self.attempts += 1
                self.image = Label(self.root, image=self.img, width =800, height=500)
                self.image.place(relx=0.5, rely=0.5, anchor='center')
        self.score_label.config(text =f'Score: {self.score} / {self.attempts}')
        self.score_label.update()

        print(self.colours[self.colour], event.char)
        print(f'Score:{self.score}/{self.attempts}')

    def new_question(self):
        if self.image.winfo_exists():
            self.image.destroy()
        if self.times_pressed == 0:
            if not self.start:
                self.attempts += 1
            self.score_label.config(text =f'Score: {self.score} / {self.attempts}')
            self.score_label.update()
        self.start = False
        self.word, self.colour = self.stimulus()
        self.label.config(text=self.word, fg=self.colour)
        self.label.update()
        self.root.after((self.timeout)*1000, self.new_question)

    def quit_selected(self):
        if sound:
            self.sound_thread.terminate()
        self.root.destroy()

    def run(self):
        self.closebutton = Button(self.root, text='close', command=self.quit_selected)
        self.closebutton.pack(padx=50, pady=50)
        self.new_question()
        self.root.bind('<Key>', lambda event,: self.key_pressed(event))

        self.root.mainloop()

if __name__ == '__main__':
    st = stroop_test(timeout=2, sound=True)
    st.run()
