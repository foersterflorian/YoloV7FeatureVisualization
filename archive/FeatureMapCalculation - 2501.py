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

#import torch.multiprocessing as mp
import torch
import multiprocessing as mp
#import multiprocess as mp
import numpy as np
import time
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
    
    
def worker_func(queue_in, queue_out, worker_id, event):
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
    
    #time.sleep(4)
    print(f"Worker ID {mp.current_process()}")
    while True:
        #print("TEST")
        #print(f"Worker ID {mp.current_process()}")
        if not queue_in.empty():
            # should use try because it's possible that non empty queue is already emptied by another process
            #print(f"Branch from Process with ID {mp.current_process()}")
            q_val = queue_in.get()
            print("----------- Got Val in Worker Func")
            dict_clone = {}
            
            """
            for key, tensor in q_val.items():
                dict_clone[key] = tensor.clone().cpu()
                print(f"Max val {dict_clone[key].max()}")
                print(f"Min val {dict_clone[key].min()}")
                print(f"Size {dict_clone[key].size()}")
                print(f"Unique vals {torch.unique(dict_clone[key])}")
            """
            print("----------- Tensor still in GPU (local copy) ---------")
            dict_clone = q_val.clone()
            print(f"Max val {dict_clone.max()}")
            print(f"Min val {dict_clone.min()}")
            print(f"Size {dict_clone.size()}")
            print(f"Unique vals {torch.unique(dict_clone)}")
            print("----------------------------------------")
            
            print("----------- Tensor convert to CPU ---------")
            dict_clone2 = q_val.clone().cpu()
            print(f"Max val {dict_clone2.max()}")
            print(f"Min val {dict_clone2.min()}")
            print(f"Size {dict_clone2.size()}")
            print(f"Unique vals {torch.unique(dict_clone2)}")
            print("----------------------------------------")
            #queue_in.task_done()
            del q_val
            event.set()
            #print(f"Time between sending and receiving: {(receive_time - q_val['sendtime']) *1000} ms")
            #print("Value retrieved.")
            #print(f"type {type(q_val)} from Process with ID {os.getpid()}")
            #print("Function call")
            
            
            s = time.time()
            #ret = calc_feature_maps_dataset(dict_clone)
            #ret = q_val.clone().cpu().numpy()
            ret = dict_clone
            
            #queue_in.task_done()
            #ret = ret.astype(np.uint8)
            
            
            #queue_in.task_done()
            e = time.time()
            #del q_val
            #print(f"Time for calculation: {(e - s) * 1000} ms")
            if queue_out.empty(): # only place if output is empty (prevent lagging with old frames)
                #print("Function executed")
                queue_out.put(ret)
                #queue_out.put(q_ret)
                
                #print("Output placed.")


def calc_feature_maps_grid(img_grid, layer_inf):
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
    if use_fixed_bound_x: # same distance from top
        s_x = 60
    else: # center vertically
        s_x = int((size_x - result.shape[0]) / 2)
    s_y = int((size_y - result.shape[1]) / 2) # center horizontally
    img_canvas[s_x:(s_x+result.shape[0]),s_y:(s_y+result.shape[1])] = result[:]
    
    # add text label with layer number
    # layer_inf contains (layer_num, batch_size, output_num)
    layer_name = f"Layer {layer_inf[0]+1} ({layer_inf[1]} out of {layer_inf[2]} outputs)"
    cv2.putText(img_canvas, layer_name, (15, 40), 0, 0.65, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
    
    return img_canvas


def calc_feature_maps_dataset(data_dict, ncol=4):
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
    
    ret_dict = {}
    img_grid_list = []
    
    for key in data_dict.keys():
        # get PyTorch Yolo tensor with 3 dimensions (num_pics, width, height)
        # put on CPU and transform to NumPy array
        img_grid_list.append(calc_feature_maps_grid(data_dict[key].cpu().numpy(), key))
        #img_grid_list.append(calc_feature_maps_grid(data_dict[key][0], norm_val, npad_inner, layer_id=key))
       
    img_grid_array = np.stack(img_grid_list)
    nindex, height, width = img_grid_array.shape
    nrows = nindex // ncol
    img_grid = (img_grid_array.reshape(nrows, ncol, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncol))
    #print(f"Time elapsed: {(e-s) * 1000} ms")
    return img_grid

			
if __name__ == '__main__':
    tensor_collection = load_pickle('tensor_collection_test_0.pkl')
    # move tensors to GPU with gradient
    device = torch.device('cuda')
    print("#################### TENSOR PUTTING ###########################")
    for key, tensor in tensor_collection.items():
        tensor_collection[key] = tensor.to(device=device, dtype=torch.float32)
        print(f"Max val {tensor_collection[key].max()}")
        print(f"Min val {tensor_collection[key].min()}")
        print(f"Min val {tensor_collection[key].size()}")
    print("Put tensors on GPU")
    print("#############################################################")
    re_calc = False
    if re_calc:
        print("Put on CPU before passing")
        dict_clone = {}
        for key, tensor in tensor_collection.items():
            dict_clone[key] = tensor.clone().cpu()
            #print(f"Max val {dict_clone[key].max()}")
            #print(f"Min val {dict_clone[key].min()}")
            #print(f"Min val {dict_clone[key].size()}")
        tensor_collection = dict_clone
    
    
    mgr = mp.Manager()
    #mp_q_in = mgr.Queue()
    #mp_q_out = mgr.Queue()
    mp_q_in = mp.Queue()
    mp_q_out = mp.Queue()
    #mp_q_in = mp.JoinableQueue()
    #mp_q_out = mp.JoinableQueue()
    event = mp.Event()
    num_proc = 2
    proc_list = []
    print("Starting processes...")
    print("... for feature map calculation ...")
    for i in range(num_proc):
        p = mp.Process(target=worker_func, args=(mp_q_in, mp_q_out, i, event))
        proc_list.append(p)
        p.start()
    
    tensor = torch.unsqueeze(tensor_collection[(0,32,32)], 0)
    print("#################### TENSOR STRAIGHT BEFORE PUTTING ###########################")
    print(f"Max val {tensor.max()}")
    print(f"Min val {tensor.min()}")
    print(f"Min val {tensor.size()}")
    print("Put 20 Dicts in Queue...")
    for i in range(5):
        #mp_q_in.put(tensor_collection, block=False)
        mp_q_in.put(tensor)
        event.wait()
        event.clear()
    
    print("Objects placed")
    print("Waiting joining")
    #mp_q_in.join()

    print("Terminating processes...")
    for p in proc_list:
        try:
            p.terminate()
            p.close()
        except:
            pass
    print("Program exit")