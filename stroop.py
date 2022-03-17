import tkinter as tk
import random

class stroop_test:


    def __init__(self, difficulty=1):

        self.colours = {'red':'r', 'blue':'b', 'green':'g', 'yellow':'y'}
        self.score = 0
        # self.same = False
        self.attempts = 0
        self.same = random.choices([True, False], weights=(0.3, 0.7))[0]
        self.word, self.colour = self.stimulus()
        self.root = tk.Tk()
        self.label = tk.Label(self.root, text=self.word, fg=self.colour, font=("Comic Sans MS", 100), width =10, height=4)
        self.label.pack()

        self.word, self.colour = self.stimulus()
        self.run()

    def stimulus(self):
        self.word = random.choice(list(self.colours))

        if self.same:
            return (self.word, self.word)

        list(self.colours).remove(self.word)
        self.colour = random.choice(list(self.colours))
        return self.word, self.colour

    def update_score(self, event):
        self.attempts += 1
        if self.colours[self.colour] == event.char:
            self.score += 1
        print(self.colours[self.colour], event.char)
        print(f'Score:{self.score}/{self.attempts}')
        self.next_selected(self.label)

    def next_selected(self, label):
        self.word, self.colour = self.stimulus()
        label.config(text=self.word, fg=self.colour)
        label.update()
        return

    def quit_selected(self):
        self.root.destroy()

    def run(self):
        self.closebutton = tk.Button(self.root, text='close', command=self.quit_selected)
        self.closebutton.pack(padx=50, pady=50)
        self.root.bind('<Key>', lambda event,: self.update_score(event))

        self.root.mainloop()



if __name__ == '__main__':
    st = stroop_test()