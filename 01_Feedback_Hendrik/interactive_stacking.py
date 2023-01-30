"""Press q to quit"""

import time
import numpy as np
from vedo import Plotter, Picture

rng = np.random.default_rng(seed=42)

#create random array of arrays with size 256x256
w, h = 256, 256
paddingWidth = 5
numberOfImages = 256  # must be devisible by rowbreakAfter
rowbreakAfter = 16


def createImage():
    data = rng.integers(0, 256, size=(numberOfImages, h, w), dtype=np.uint8)

    #do not pad allong fist axis, add "paddingWidth" padding to beginning and end of second and third axis  
    npad = ((0, 0), (paddingWidth, paddingWidth), (paddingWidth, paddingWidth))
    padded_data = np.pad(data, pad_width=npad, mode='constant', constant_values=255)

    rows = []
    for i in range(0, numberOfImages, rowbreakAfter):
        #concatenate arrays along "w" axis
        row = np.concatenate(padded_data[i:i+rowbreakAfter], axis=1)
        rows.append(row)
    #concatenate arrays along "h" axis
    return np.concatenate(rows, axis=0)

def bfunc():
    global timer_id
    plotter.timer_callback("destroy", timer_id)
    if "Play" in button.status():
        # instruct to call handle_timer() every 10 msec:
        timer_id = plotter.timer_callback("create", dt=100)
    button.switch()

def handle_timer(event):
    pic = Picture(createImage())
    plotter.pop().add(pic)


timer_id = None
plotter= Plotter(axes=0)
button = plotter.add_button(bfunc, states=[" Play ","Pause"], size=40)
evntId = plotter.add_callback("timer", handle_timer)

pic = Picture(createImage())


plotter.show(__doc__, pic, mode=8)