import tkinter as tk
from tkinter import ttk
import random
import make_random_fish as mrf
import numpy as np
import tensorflow as tf
from fish_pred import tensor_to_image
from PIL import ImageTk, Image

im_size = (480, 240)
class App:

    def make_slider(self, i):
        slide = tk.Scale(self.root, from_=0., to=1., 
                               orient='horizontal',
                                resolution=0.001,
                                command=lambda _ : self.run_model())
        slide.grid(column=i // 10, row=i % 10)
        return slide

    def randomize(self) :
        for i in range(self.total) :
            self.sliders[i].set(random.random()) 
        self.run_model()

    def tf_randomize(self) :
        vals = tf.random.uniform([20]).numpy()
        for i in range(self.total) :
            self.sliders[i].set(vals[i])
        self.run_model()

    def get_vals(self) :
        return [self.sliders[i].get() for i in range(self.total)]

    def run_model(self) :
        inps = tf.expand_dims(np.array(self.get_vals()), 0)
        data = self.model(inps, training=False)
        # print(inps)
        # print(data)
        self.img = tensor_to_image(data).resize(im_size)
        self.tk_img = ImageTk.PhotoImage(self.img)
        self.label.configure(image=self.tk_img)




    def __init__(self):
        self.root = tk.Tk()
        self.total = 20
        self.sliders = [self.make_slider(i) for i in range(self.total)]

        s1 = ttk.Separator(self.root, orient='horizontal')
        s1.grid(row=10,column=0,columnspan=2, ipadx=100)
        
        self.rand_b = tk.Button(self.root, text='Random', command=self.randomize)
        self.rand_b.grid(row=11,column=0)

        self.tf_rand_b = tk.Button(self.root, text='TF Random', command=self.tf_randomize)
        self.tf_rand_b.grid(row=11,column=1)

        s2 = ttk.Separator(self.root, orient='vertical')
        s2.grid(column=2,row=0,rowspan=13, ipady=220)

        self.img = Image.new('RGB', im_size)
        self.tk_img = ImageTk.PhotoImage(self.img)
        self.label = tk.Label(self.root, image=self.tk_img)
        self.label.grid(row=0,column=3,rowspan=12)

        self.model = mrf.rand_fish_model()
        path = 'best_so_far/random_fish/gen'

        cp = tf.train.Checkpoint(self.model)
        cp.restore(path).expect_partial()
        
        self.tf_randomize()

        self.root.mainloop()

app=App()