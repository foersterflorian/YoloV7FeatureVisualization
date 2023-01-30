from vedo import *
from mp_test import load_pickle
import time
#import numpy as np

N = None
timer_id = None
keys = None
feature_map_grid = None
plotter = None
button = None
mp_q2 = None
first_exec = True
reset_cam = True

def bfunc():
    global timer_id
    plotter.timer_callback("destroy", timer_id)
    if "Start" in button.status():
        # instruct to call handle_timer() every 10 msec:
        timer_id = plotter.timer_callback("create", dt=1000)
    button.switch()

def handle_timer(event):
    global reset_cam # first execution camera adaption
    if not mp_q2.empty():
        
        
        feature_map_grid = mp_q2.get() # retrieve image grid
        
        global first_exec
        if first_exec: # re-initialize key values on first update
            global keys
            keys = list(feature_map_grid.keys()) 
            first_exec = False
        
        print("Execution...")
        s = time.time()
        actors = plotter.actors ##############
        for i, key in enumerate(keys):
            s1 = time.time()
            pic = Picture(feature_map_grid[key], flip=True)
            e1 = time.time()
            s2 = time.time()
            #plotter.remove(actors)
            plotter.remove(actors, at=i)
            #plotter.at(i).add(pic, f"Layer {keys[i]+1}", resetcam=reset_cam)
            #plotter.at(i).add(pic, f"Layer {keys[i]+1}", resetcam=reset_cam)
            plotter.at(i).show(pic, f"Layer {keys[i]+1}", at=i, resetcam=reset_cam)
            #plotter.at(i).show(pic, resetcam=reset_cam)
            e2 = time.time()
            print(f"Time for converting image: {(e1 - s1) * 1000} ms")
            print(f"Time for display image: {(e2 - s2) * 1000} ms")
        if reset_cam:
            reset_cam = False
        e = time.time()
        print(f"Time for Timer Handling: {(e - s) * 1000} ms")
        #pic = Picture()
        #plotter.pop().add(pic)
    
def handle_timer_old(event):
    print("Execution...")
    s = time.time()
    for i, key in enumerate(keys):
        s1 = time.time()
        pic = Picture(feature_map_grid[key], flip=True)
        e1 = time.time()
        s2 = time.time()
        plotter.remove(at=i).at(i).add(pic, f"Layer {keys[i]+1}")
        e2 = time.time()
        print(f"Time for converting image: {(e1 - s1) * 1000} ms")
        print(f"Time for display image: {(e2 - s2) * 1000} ms")
    e = time.time()
    print(f"Time for Timer Handling: {(e - s) * 1000} ms")
    #pic = Picture()
    #plotter.pop().add(pic)
    
def init():
    for i in range(N):
        pic = Picture(feature_map_grid[keys[i]], flip=True)
        plotter.at(i).show(pic, f"Layer {keys[i]+1}")
        
def close_call():
    plotter.timer_callback("destroy", timer_id)
    plotter.break_interaction()
    plotter.close()
    
def show_mp(mp_input_queue):
    global N
    global timer_id
    global keys
    global feature_map_grid
    global plotter
    global button
    global mp_q2
    global first_exec
    first_exec = True # set first execution to True
    mp_q2 = mp_input_queue # assign global queue variable to input from main process
    feature_map_grid = load_pickle('test_img_grid.pkl')
    N = len(feature_map_grid)
    keys = list(feature_map_grid.keys())
    #plotter= Plotter(axes=0)
    plotter = Plotter(N=N, axes=0, size=(1760,990), sharecam=False, title='Feature Map Visualization')
    button = plotter.add_button(bfunc, states=["Start","Stop"], pos=(0.5, 0.05), size=10)
    button_init = plotter.add_button(init, states=["Init"], pos=(0.3, 0.05), size=10)
    button_close = plotter.add_button(close_call, states=["Close"], pos=(0.7, 0.05), size=10)
    evntId = plotter.add_callback("timer", handle_timer)
    
    plotter.interactive()
    
def show():
    global N
    global timer_id
    global keys
    global feature_map_grid
    global plotter
    global button
    feature_map_grid = load_pickle('test_img_grid.pkl')
    N = len(feature_map_grid)
    keys = list(feature_map_grid.keys())
    #plotter= Plotter(axes=0)
    plotter = Plotter(N=N, axes=0, size=(1760,990), sharecam=False, title='Feature Map Visualization')
    button = plotter.add_button(bfunc, states=["Start","Stop"], pos=(0.5, 0.05), size=10)
    button_init = plotter.add_button(init, states=["Init"], pos=(0.3, 0.05), size=10)
    button_close = plotter.add_button(close_call, states=["Close"], pos=(0.7, 0.05), size=10)
    evntId = plotter.add_callback("timer", handle_timer)
    
    plotter.interactive()

if __name__ == '__main__':
    show()