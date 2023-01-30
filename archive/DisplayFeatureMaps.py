import os
"""
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
"""
num_threads = "2"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import numpy as np
import cv2
import torch.multiprocessing as mp
#import multiprocessing as mp
#import multiprocess as mp
import time
import psutil


def display_results(queue_in, worker_id, event_terminating):
    """
    queue query to retrieve processed feature maps images from queue and display them
    """
    q_val = 0
    got_frame_time = None
    """
    p = psutil.Process()
    print(f"Child #{mp.current_process()}: {p}, affinity {p.cpu_affinity()}")
    p.cpu_affinity([worker_id])
    print(f"Child #{mp.current_process()}: {p}, affinity {p.cpu_affinity()}")
    """
    print(f"Worker Display with ID {mp.current_process()}")
    while True:
        if event_terminating.is_set():
            print(f"Stop process with Worker ID {mp.current_process()}")
            break
        if not queue_in.empty():
            frame_time = (time.time() - got_frame_time) if got_frame_time is not None else ""
            #print(f"Got frame in: {frame_time * 1000} ms")
            got_frame_time = time.time()
            # should use try because it's possible that non empty queue is already emptied by another process
            #print(f"Branch from Displaying-Process with ID {mp.current_process()}")
            #s = time.time()
            res_img = queue_in.get()
            cv2.imshow("Feature Maps", res_img)
            #print(f"Time for image displaying: {(time.time() - s) * 1000} ms")
            cv2.waitKey(1)
            
    cv2.destroyAllWindows()