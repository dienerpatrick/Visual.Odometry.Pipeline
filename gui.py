import matplotlib.animation as animation
from matplotlib import style
import matplotlib.pyplot as plt
style.use('ggplot')
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import OptionMenu, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.image as mpimg
import os
import threading
from PIL import Image,ImageTk


def start_gui():

    root = tk.Tk()
    root.configure(background='white')
    root.geometry('1600x900')
    root.title('Visiual Odometry')

    style.use('ggplot')

    running = False

    try:
        os.remove("temp/trajectory.png")
        os.remove("temp/image.png")
    except:
        pass

    # configure the grid
    root.columnconfigure(0, weight=3)
    root.columnconfigure(1, weight=3)
    root.columnconfigure(2, weight=4)
    root.columnconfigure(3, weight=4)

    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=5)
    root.rowconfigure(2, weight=1)
    root.rowconfigure(3, weight=1)
    root.rowconfigure(4, weight=1)
    root.rowconfigure(5, weight=1)

    # HEADER

    canvas = tk.Canvas(root, background='white', width= 1920, height= 150, highlightthickness=0, relief='ridge')
    canvas.create_text(800, 50, anchor='center', text="VISION ALGORITHMS FOR MOBILE ROBOTICS SEMESTER PROJECT",
                       fill="black", font=('Helvetica 30 bold'))
    canvas.create_text(800, 80, anchor='center', text="Yannick Werner, David Filiberti, Felix Schnarrenberger, Patrick Diener",
                       fill="black", font=('Script 18 bold'))
    # logo = ImageTk.PhotoImage(Image.open("temp/eth_logo.png").resize((int(2560/15), int(427/15)), Image.ANTIALIAS))
    # canvas.create_image(40, 40, image=logo, anchor='nw')
    canvas.grid(column=0, row=0, columnspan=5, sticky=tk.EW)

    # POSES PLOT

    figure1 = plt.Figure()
    ax1 = figure1.add_subplot(111)

    r_poses = FigureCanvasTkAgg(figure1, root)
    r_poses.get_tk_widget().grid(row=1, column=2, columnspan=2, sticky=tk.NS)

    #IMAGES PLOT

    figure2 = plt.Figure()
    ax2 = figure2.add_subplot(111)

    r_images = FigureCanvasTkAgg(figure2, root)
    r_images.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky=tk.EW)

    def run():
        os.system(f'python ./main.py {var_dataset.get()} {var_algorithm.get()} {frames_entry.get()}')

    def start_run_thread():
        global submit_thread
        submit_thread = threading.Thread(target=run)
        submit_thread.daemon = True
        submit_thread.start()
        nonlocal running
        running = True

    # BUTTON
    run_button = tk.Button(root, text="Run Simulation", command=lambda:start_run_thread(), highlightbackground='white')
    run_button.grid(column=3, row=5, columnspan=1, padx= 50, sticky=tk.E)

    # CHOICES
    choices_dataset = ['kitti', 'parking', 'malaga']
    var_dataset = StringVar(root)
    option_dataset = OptionMenu(root, var_dataset, *choices_dataset)
    option_dataset["menu"].configure(activebackground="white")
    print(option_dataset["menu"].keys())
    var_dataset.set('kitti')
    option_dataset.grid(row=3, column=0, rowspan=1, sticky=tk.W, padx= 50)

    choices_algorithm = ['2D-2D', '3D-2D']
    var_algorithm = StringVar(root)
    option_algorithm = OptionMenu(root, var_algorithm, *choices_algorithm)
    option_algorithm["menu"].configure(activebackground="white")
    var_algorithm.set('2D-2D')
    option_algorithm.grid(row=4, column=0, rowspan=1, padx= 50, sticky=tk.W)

    def temp_text(e):
        frames_entry.delete(0, "end")

    var_frames = StringVar(root)
    frames_entry = tk.Entry(root, bg="white", fg="black", highlightbackground='white', textvariable=var_frames)
    frames_entry.insert(0, "   Number of Frames")
    frames_entry.grid(column=0, row=5, sticky=tk.W, padx= 50)
    frames_entry.bind("<FocusIn>", temp_text)

    def animate_poses(i):
        if running:
            try:
                img = mpimg.imread("temp/trajectory.png")
            except:
                return
            ax1.clear()
            ax1.grid(False)
            ax1.axis('off')
            ax1.imshow(img)
            os.remove("temp/trajectory.png")
            return

    def animate_images(i):
        if running:
            try:
                img = mpimg.imread("temp/image.png")
            except:
                return
            ax2.clear()
            # hide grid and axes
            ax2.grid(False)
            ax2.axis('off')
            ax2.imshow(img)
            os.remove("temp/image.png")
            return


    ani1 = animation.FuncAnimation(figure1, animate_images, interval=100)
    ani2 = animation.FuncAnimation(figure2, animate_poses, interval=100)

    root.mainloop()


if __name__ == "__main__":
    start_gui()