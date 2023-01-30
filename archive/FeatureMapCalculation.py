import os
"""
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
"""
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import multiprocessing as mp
#import multiprocess as mp
import numpy as np
from time import sleep, time
#import os
import pickle
import cv2
import psutil

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
    
    
def worker_func(queue_in, queue_out, worker_id):
    """
    queue query to retrieve items from queue
    """
    q_val = 0
    """
    p = psutil.Process()
    #print(f"Child #{mp.current_process()}: {p}, affinity {p.cpu_affinity()}")
    p.cpu_affinity([worker_id])
    print(f"Child #{mp.current_process()}: {p}, affinity {p.cpu_affinity()}")
    """
    
    #sleep(4)
    print(f"Worker ID {mp.current_process()}")
    while True:
        #print("TEST")
        #print(f"Worker ID {mp.current_process()}")
        if not queue_in.empty():
            # should use try because it's possible that non empty queue is already emptied by another process
            #print(f"Branch from Process with ID {mp.current_process()}")
            q_val = queue_in.get()
            #receive_time = time()
            #print(f"Time between sending and receiving: {(receive_time - q_val['sendtime']) *1000} ms")
            #print("Value retrieved.")
            #print(f"type {type(q_val)} from Process with ID {os.getpid()}")
            #print("Function call")
            #s = time()
            ret = calc_feature_maps_dataset(q_val)
            #e = time()
            #print(f"Time for calculation: {(e - s) * 1000} ms")
            #s = time()
            if queue_out.empty(): # only place if output is empty (prevent lagging with old frames)
                #print("Function executed")
                queue_out.put(ret)
                #print(f"Time for putting queue: {(time() - s) * 1000} ms") 
                #print("Output placed.")


def calc_feature_maps_grid(layer_imgs, norm_val, npad_inner, layer_id=None, n_img_max=64, row_break_after=8):
    """
    - calculates normalized matrices to a maximum number of 64 elements and maps them to grayscale
    - puts elements in a grid
    Input: 
        - layer_imgs:      NumPy array with 3 dimensions (num_pic, width, height), derived from Yolo Model tensor collection
        - norm_val:        normalization value (max value from image processing 255) as NumPy array
        - npad_inner:      padding values for each feature map
        - layer_id:        number of layer which is processed
        - n_img_max:       maximum amount of feature maps which are considered, all maps with index >= n_img_max are negated
        - row_break_after: number of columns for image grid which is to be built
    Output: NumPy array - full normalized and resized image grid with all chosen feature maps and additional layer information
    """
    
    if layer_imgs.shape[0] <= n_img_max:
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
    
    # pad single feature maps for better appearance
    layer_imgs = np.pad(layer_imgs, pad_width=npad_inner, mode='constant', constant_values=255.0)
    
    # https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    nindex, height, width = layer_imgs[:n,:,:].shape
    nrows = n // row_break_after
    img_grid = (layer_imgs[:n,:,:].reshape(nrows, row_break_after, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*row_break_after))
    
    # convert data type
    img_grid = img_grid.astype(np.uint8)
    
    # resize image
    target_width = 400
    scale_percent = float(target_width / img_grid.shape[1])
    target_height = int(img_grid.shape[0] * scale_percent)
    dim = (target_width, target_height)
    if scale_percent > 1.0: # upsampling
        result = cv2.resize(img_grid, dim, interpolation=cv2.INTER_CUBIC)
    else: # downsampling
        result = cv2.resize(img_grid, dim, interpolation=cv2.INTER_AREA)
    
    # create new filled array of desired size
    size_x = 450
    size_y = 450
    use_fixed_bound_x = True
    img_canvas = np.full((size_x, size_y), 255, dtype=np.uint8)
    # place smaller image in the greater canvas
    # shifting values s_x, s_y
    if use_fixed_bound_x: # same distance from above
        s_x = 60
    else: # center vertically
        s_x = int((size_x - result.shape[0]) / 2)
    s_y = int((size_y - result.shape[1]) / 2) # center horizontally
    img_canvas[s_x:(s_x+result.shape[0]),s_y:(s_y+result.shape[1])] = result[:]
    
    # add text label with layer number
    layer_name = f"Layer {layer_id+1} ({n} out of {layer_imgs.shape[0]} outputs)"
    cv2.putText(img_canvas, layer_name, (15, 40), 0, 0.65, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
    
    
    return img_canvas


def calc_feature_maps_dataset(data_dict, layer_nums=[], ncol=4):
    """
    For whole data set as dictionary with PyTorch Yolo tensors:
        --> func call calc_feature_maps_grid:
            - calculates normalized matrices to a maximum number of 64 elements and maps them to grayscale
            - puts elements in a grid
    - order feature map grids as one single grid in a large picture, ready to be displayed by any viewer 
      which accepts NumPy arrays
    
    Input: User-defined tensor collection dictionary added to the Yolo Model as attribute
        - data_dict:    tensor collection from Yolo Model
        - layer_nums:   used to pre-filter layers for which calculation shall take place
        - ncol:         number of columns in final output image
    Output: Single image with provided feature maps visualized
    """
    #s = time()
    ret_dict = {}
    norm_val = np.asarray([255], dtype=np.float32) # normalization value of 255 (grayscale --> white)
    padding_width_inner = 5
    npad_inner = ((0,0), (padding_width_inner, padding_width_inner), (padding_width_inner, padding_width_inner))


    if layer_nums:
        keys = layer_nums
    else:
        keys = data_dict.keys()
    
    img_grid_list = []
    for key in keys:
        # get PyTorch Yolo tensor with 3 dimensions (num_pics, width, height)
        # transform to NumPy array
        img_grid_list.append(calc_feature_maps_grid(data_dict[key][0].detach().numpy(), norm_val, npad_inner, layer_id=key))
        #img_grid_list.append(calc_feature_maps_grid(data_dict[key][0], norm_val, npad_inner, layer_id=key))
       
    img_grid_array = np.stack(img_grid_list)
    nindex, height, width = img_grid_array.shape
    nrows = nindex // ncol
    img_grid = (img_grid_array.reshape(nrows, ncol, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncol))
    #e = time()
    #print(f"Time elapsed: {(e-s) * 1000} ms")
    return img_grid

			
if __name__ == '__main__':
    tensor_collection = load_pickle('tensor_collection_new.pkl')
    print("Dict loaded.")
    mgr = mp.Manager()
    mp_q_in = mgr.Queue()
    mp_q_out = mgr.Queue()
    #mp_q_in = mp.Queue()
    #mp_q_out = mp.Queue()
    num_proc = 4
    proc_list = []
    print("Starting processes...")
    print("... for feature map calculation ...")
    for i in range(num_proc):
        p = mp.Process(target=worker_func, args=(mp_q_in, mp_q_out, i))
        proc_list.append(p)
        p.start()
    
    print("Put 20 Dicts in Queue...")
    for i in range(30):
        mp_q_in.put(tensor_collection, block=False)
    
    print("Objects placed")
    print("Waiting 20sec ...")
    sleep(20)

    print("Terminating processes...")
    for p in proc_list:
        try:
            p.terminate()
            p.close()
        except:
            pass
    print("Program exit")