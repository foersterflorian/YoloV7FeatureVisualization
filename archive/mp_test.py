import multiprocessing as mp
import numpy as np
from time import sleep
import os
import pickle

# save pickle
def save_pickle(filename, obj):
    with open(str(filename), 'wb') as f:
        pickle.dump(obj, f)
    print("Object saved")
        
# load pickle
def load_pickle(filename):
    with open(str(filename), 'rb') as f:
        obj = pickle.load(f)
    print("Object loaded")
    return obj
    
    
def worker_func(queue_in, queue_out):
    """
    queue query to retrieve items from queue
    """
    q_val = 0
    print(f"Worker ID {mp.current_process()}")
    while True:
        #print("TEST")
        #print(f"Worker ID {mp.current_process()}")
        if not queue_in.empty():
            # should use try because it's possible that non empty queue is already emptied by another process
            print(f"Branch from Process with ID {mp.current_process()}")
            q_val = queue_in.get()
            #print("Value retrieved.")
            #print(f"type {type(q_val)} from Process with ID {os.getpid()}")
            #print("Function call")
            ret = calc_grayscale_dataset(q_val)
            if queue_out.empty(): # only place if output is empty (prevent lagging with old frames)
                print("Function executed")
                queue_out.put(ret)
                print("Output placed.")


def calc_grayscale(layer_imgs, norm_val, npad, n_img_max=64, row_break_after=8):
    """
    Calculates normalized matrices in a list of 2D arrays and maps them to grayscale
    Input: NumPy array with 3 dimensions (num_pic, width, height)
    Output: modified NumPy array with same 3 dimensions
    """
    
    if layer_imgs.shape[0] <= 64:
        n = layer_imgs.shape[0]
    else:
        n = n_img_max
    
    # normalize images
    for i in range(n):
        #img = test_map[i,:,:]
        img = layer_imgs[i]
        #print("TEST")
        v_min = np.asarray([img.min()], dtype=np.float32)
        v_max = np.asarray([img.max()], dtype=np.float32)
        img -= v_min
        img /= (v_max - v_min)
        img *= norm_val
    
    # pad batch
    layer_imgs = np.pad(layer_imgs, pad_width=npad, mode='constant', constant_values=255)

    # pad and concatenate
    #padded_data = np.pad(data, pad_width=npad, mode='constant', constant_values=255)
    rows = []
    for i in range(0, n, row_break_after):
        #concatenate arrays along "w" axis
        row = np.concatenate(layer_imgs[i:i+row_break_after], axis=1)
        rows.append(row)
    
    #concatenate arrays along "h" axis
    return np.concatenate(rows, axis=0)


def calc_grayscale_dataset(data_dict, layer_nums=[]):
    """
    For whole data set as PyTorch Yolo tensor:
    Calculates normalized matrices in a list of 2D arrays and maps them to grayscale
    
    Input: User-defined tensor collection library added to the Yolo Model as attribute
    Output: Dictionary with layer number as key and corresponding normalized feature maps
    """
    
    ret_dict = {}
    norm_val = np.asarray([255], dtype=np.float32) # normalization value of 255 (grayscale --> white)
    padding_width = 5
    npad = ((0,0), (padding_width, padding_width), (padding_width, padding_width))

    if layer_nums:
        keys = layer_nums
    else:
        keys = data_dict.keys()

    for key in keys:
        # get PyTorch Yolo tensor with 3 dimensions (num_pics, width, height)
        # transform to NumPy array
        ret_dict[key] = calc_grayscale(data_dict[key][0].detach().numpy(), norm_val, npad)
        
    
    return ret_dict

			
if __name__ == '__main__':
    tensor_collection = load_pickle('tensor_collection_new.pkl')
    print("Dict loaded.")
    pool = mp.Pool(processes=8)
    mgr = mp.Manager()
    mp_q_in = mgr.Queue()
    mp_q_out = mgr.Queue()
    pool.apply_async(worker_func, (mp_q_in, mp_q_out))
    
    print("Put 20 Dicts in Queue...")
    for i in range(20):
        mp_q_in.put(tensor_collection, block=False)
    
    print("Waiting 7sec ...")

    sleep(7)

    print("Terminating processes...")
    pool.close()
    pool.terminate()
    pool.join()