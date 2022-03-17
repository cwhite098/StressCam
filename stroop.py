import tkinter as tk
import random

def stroop_test():

    global colour, score, word, attempts
    colours = {'red':'r', 'blue':'b', 'green':'g', 'yellow':'y'}
    def stimulus(same):
        word = random.choice(list(colours))

        if same:
            return (word, word)

        list(colours).remove(word)
        colour = random.choice(list(colours))
        return word, colour

    def update_score(event):
        global attempts, score
        attempts += 1
        if colours[colour] == event.char:
            score += 1
        print(colours[colour], event.char)
        print(f'Score:{score}/{attempts}')
        next_selected()

    def next_selected():
        global word, colour
        word, colour = stimulus(random.choices([True, False], weights=(0.3, 0.7))[0])
        label.config(text=word, fg=colour)
        label.update()
        return

    def quit_selected():
        root.destroy()

    root = tk.Tk()
    score = 0
    attempts = 0

    # create label using stimulus
    word, colour = stimulus(False)
    label = tk.Label(root, text=word, fg=colour, font=("Comic Sans MS", 100), width =10, height=4)
    label.pack()

    closebutton = tk.Button(root, text='close', command=quit_selected)
    closebutton.pack(padx=50, pady=50)

    root.bind('<Key>', lambda event,: update_score(event))
    root.mainloop()

if __name__ == '__main__':
    stroop_test()