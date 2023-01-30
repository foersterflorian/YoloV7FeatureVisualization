import os
"""
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6
"""
# define number of threads NumPy uses in the background
# compatibility with multiprocessing
num_threads = "4"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import time
import threading
import multiprocessing as mp
import queue
import argparse
import pickle
from pynput import keyboard

from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import LoadImages, letterbox
from utils.plots import plot_one_box
from utils.torch_utils import intersect_dicts, time_synchronized
from models.common import Conv
from models.yolo import Model

prog_exit = False # used to check if program should be exited


class LoadWebcamThreaded:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32, fps=30.0, frame_count=1):
        """
        Threaded webcam image retrieval
        ONLY 1 SOURCE SUPPORTED!
        """
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.fps = float(fps)
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set frame size 720p
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps) # set FPS
        
        self.img = None
        self.frame_count = frame_count # define to obtain every frame_count's frame, 1 = every frame
        self.waitTime = 1 / self.fps
        thread = threading.Thread(target=self.update, args=(self.cap,), daemon=True)
        print("Starting image retrieval thread...")
        thread.start()
        time.sleep(3) # load webcam stream

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Copy frame
        img0 = self.img.copy()
        img0 = cv2.flip(img0, 1)  # flip left-right
        #print(f"IMG SIZE = {img0.shape} \t IMG DTYPE = {img0.dtype}")

        # Print
        #assert ret_val, f'Camera Error {self.pipe}, Could not retrieve images'
        img_path = 'webcam.jpg'
        #print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None
        
    def update(self, cap):
        # Read next stream frame in a daemon thread
        n = 0
        print("Image retrieval thread started successfully.")
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == self.frame_count:  # read every frame_count's frame
                success, im = cap.retrieve()
                self.img = im if success else self.img * 0
                n = 0
            time.sleep(self.waitTime)  # wait time to not retrieve more images than necessary

    def __len__(self):
        return 0


def load_model(model_path, data_path='data/coco.yaml', device_type='cpu', save_init_tensor_collection=False):
    """
    loads specified model from provided path
    
    modelPath:      Path to PyTorch model, model weights as '.pt' file
    dataPath:       Path to YAML for model class names
    deviceType:     'cpu' (CPU) or 'cuda' (GPU)
                    GPU option has to be tested before deployment
    
    return:
        loaded PyTorch Yolo Model
        PyTorch device type
    
    """
    
    with open(data_path) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    if device_type != 'cpu':
        if torch.cuda.is_available():
            device_type = 'cuda'
            print("Set device type to CUDA and building for GPU")
        else:
            raise TypeError("Model should be built for GPU, but no CUDA GPU was found. Use device_type='cpu' instead.")
    
    device = torch.device(device_type)
    ckpt = torch.load(model_path, map_location=device) # load model from weights
    model = Model(ckpt['model'].yaml, ch=3, device=device).to(device) # build model with class Model from Yolo project with user-modified properties
    
    # normalization value in model
    model.norm_val = model.norm_val.to(device)
    
    # parse weights
    state_dict = ckpt['model'].float().state_dict() # FP32 model
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=[])
    model.load_state_dict(state_dict, strict=False)
    model.fuse().eval() # fuse layers
    
    # make model compatible with different PyTorch versions
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        
    model.names = data_dict['names'] # set class names of model to read YAML data
    
    if device_type != 'cpu': # half precision only supported on CUDA
        half = True
        model.half()  # to FP16
    else:
        half = False
    
    
    # initialize to retrieve one representation of a feature map collection
    imgsz = 640
    stride = int(model.stride.max())
    if device_type != 'cpu':
        cudnn.benchmark = True
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    # Padded resize
    img0 = np.full((720,1280,3), 255, dtype=np.uint8)
    img = letterbox(img0, imgsz, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # run model
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float() # uint8 to fp16/32 
    #print(f"IMG dtype = {img.dtype}")
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #img_0 = img
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    pred = model(img, augment=False)[0]
    
    if save_init_tensor_collection:
        save_pickle("data/tensor_collection_reference.pkl", model.tensor_collection)
    
    return model, device

  
def prediction(source, model, device, mp_q_in, threading_queue=None, event_handler_thread=None, event_terminating=None, img_size=640, n_th_frame=1, 
                save_img=False, output_img='output/detect.jpg', conf_thres=0.25):
    """
    TESTING, NOT FINISHED YET
    ONLY RETURNS ONE TENSOR COLLECTION FOR TESTING OF VEDO RENDERING
    
    loads images from given path and executes prediction
    only static, has to be enhanced for video feed predictions
    
    return:
        Prediction tensor
        Tensor collection with output of each layer (used for feature map visualization)
    """
    # control program execution / stop execution variable
    global prog_exit
    
    # set value for nms, later use flags in Python programm
    iou_thres = 0.45
    classes = None # filter classes
    agnostic_nms = False
    
    # model properties
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    np_gen = np.random.default_rng(seed=42)
    colors = [[np_gen.integers(0, 255, dtype=int) for _ in range(3)] for _ in names]
    
    # check for webcam
    if source.isnumeric():
        webcam = True
    else:
        webcam = False
        
    if webcam:
        #view_img = check_imshow()
        if device.type != 'cpu':
            cudnn.benchmark = True  # set True to speed up constant image size inference
        #dataset = LoadWebcam(pipe=source, img_size=imgsz, stride=stride)
        dataset = LoadWebcamThreaded(pipe=source, img_size=imgsz, stride=stride, frame_count=1)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    if device.type != 'cpu':
        half = True
        #model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # already run in 'load_model' function
    else:
        half = False
    
    #time.sleep(1)
    num_iter = 0
    # n_th_frame --> only broadcast feature maps of every n-th frame
    
    t0 = time.time()
    # for single images only length 1
    for path, img, im0s, vid_cap in dataset:
    
        ######################
        if prog_exit: # stop execution when global variable is set
            event_terminating.set() # start terminating child processes
            break
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() # uint8 to fp16/32 
        #print(f"IMG dtype = {img.dtype}")
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #img_0 = img
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        #t1 = time_synchronized()
        
        if not event_handler_thread.is_set():
            pred = model(img, augment=False)[0]
        else:
            print("[WARNING] Could not start prediction because tensor processing in handling thread not done yet. \
                    \nNecessary to prevent data corruption.")
            continue
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        
        #t2 = time_synchronized()
        #print(f"Interference Time: {(t2 - t1) * 1000} ms")
        torch.cuda.synchronize()
        # synchronize call necessary because feature map dictionary building is asynchronous in Yolo model
        
        # feature maps ready for postprocessing
        # only place if thread has successfully placed object and cleared event, and if frame should not be skipped
        if not event_handler_thread.is_set() and (num_iter % n_th_frame) == 0:
            threading_queue.put(model.tensor_collection, block=False)
            event_handler_thread.set() # signal to queue thread to handle queue IO
            #time.sleep(1/1000) # synchronization time to retrieve feature map data, not needed because of thread state check before model prediction
        else: # thread too slow
            print("Threading Queue not empty. Queue handler thread too slow.")
        
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                #p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p, s, im0, frame = path, '', im0s, dataset.count   
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            #p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            #s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #with open(txt_path + '.txt', 'a') as f:
                            #f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            
            #t3 = time.time()
            #print(f"Time for image post-processing: {(t3 - t2) * 1000} ms")
            #print(f"Time for interference & image post-processing: {(t3 - t1) * 1000} ms")
            
            # Stream results
            if webcam:
                cv2.imshow('Webcam', im0)
                cv2.waitKey(1)  # 1 millisecond
            else:
                if save_img:
                    cv2.imwrite(output_img, im0)
                    print(f"Detection image saved under: {output_img}")
                cv2.imshow('Image', im0)
                if cv2.waitKey(0) & 0xFF == ord('e'):
                    continue
        
        # frame iteration counter
        num_iter += 1
    
    # destroy all OpenCV windows
    cv2.destroyAllWindows()
    
       
    return True
  
    
def prediction_wo_mp(source, model, device, img_size=640, save_img=False, output_img='output/detect.jpg', use_colab=False, conf_thres=0.25):
    """
    TESTING AND DEBUGGING ONLY
    ONLY RETURNS ONE TENSOR COLLECTION FOR TESTING PURPOSES
    
    loads images from given path and executes prediction
    only static, has to be enhanced for video feed predictions
    
    return:
        Prediction tensor
        Tensor collection with output of each layer (used for feature map visualization)
    """
    
    # set value for nms, later use flags in Python programm
    iou_thres = 0.45
    classes = None # filter classes
    agnostic_nms = False
    
    if use_colab: # Google Colab has no support for OpenCV imshow
        print("Use Google Colab. Import OpenCV imshow patch.")
        from google.colab.patches import cv2_imshow
    
    
    # model properties
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]
    
    # check for webcam
    if source.isnumeric():
        #source = eval(source)
        webcam = True
    else:
        webcam = False
        
    if webcam:
        #view_img = check_imshow()
        if device.type != 'cpu':
            cudnn.benchmark = True  # set True to speed up constant image size inference
        #dataset = LoadWebcam(pipe=source, img_size=imgsz, stride=stride)
        dataset = LoadWebcamThreaded(pipe=source, img_size=imgsz, stride=stride, frame_count=1)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
    if device.type != 'cpu':
        half = True
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    else:
        half = False
    
    #time.sleep(1)
    num_iter = 0
    
    t0 = time.time()
    # for single images only length 1
    for path, img, im0s, vid_cap in dataset:
    
        ######################
        num_iter += 1
    
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() # uint8 to fp16/32 
        #print(f"IMG dtype = {img.dtype}")
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #img_0 = img
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        #print(f"IMG SIZE = {img.size()}")
        #print(f"AUGMENT..... {opt.augment}")
        pred = model(img, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()
        print(f"Interference Time: {(t2 - t1) * 1000} ms")
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                #p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p, s, im0, frame = path, '', im0s, dataset.count
                
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            #p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # img.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            #s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                        #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        #with open(txt_path + '.txt', 'a') as f:
                            #f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            print(f"Time for image post-processing: {(time.time() - t2) * 1000} ms")
            print(f"Time for interference & image post-processing: {(time.time() - t1) * 1000} ms")
            # Stream results
            if not use_colab:
                if webcam:
                    cv2.imshow('Webcam', im0)
                    cv2.waitKey(1)  # 1 millisecond
                else:
                    if save_img:
                        cv2.imwrite(output_img, im0)
                        print(f"Detection image saved under: {output_img}")
                    cv2.imshow('Image', im0)
                    if cv2.waitKey(0) & 0xFF == ord('e'):
                        continue
            else:
                if webcam: # no webcam support yet
                    #cv2_imshow(im0)
                    #cv2.waitKey(1)  # 1 millisecond
                    pass
                else:
                    if save_img:
                        cv2.imwrite(output_img, im0)
                        print(f"Detection image saved under: {output_img}")
                    cv2_imshow(im0)

    if not use_colab:
        cv2.destroyAllWindows()
    
       
    return True    


def queue_handler(threading_queue, mp_queue_in, event_handler_thread, 
                metadata_worker, event_worker_sync, lock_worker_sync):
    """
    Queue handler thread with shared memory (SHM) approach for faster procesing
    SHM approach
    - write in arrays of "shm_array_dict" with lock from main handler thread
    - make local copy of dictionary with for-loop
    - does not rely on deepcopy anymore, all done via SHM and proper synchronization
    """
    # buffer arrays for feature maps
    shm_array_dict = {}
    buff_list_thread = []
    for key, metadata in metadata_worker.items():
        buff_display_name, shape, dtype = metadata
        existing_shm = mp.shared_memory.SharedMemory(name=buff_display_name)
        buff_list_thread.append(existing_shm)
        shm_array_dict[key] = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    # SHM approach
    # write in arrays of "shm_array_dict" with lock from main handler thread
    
    # main loop
    while True:
        event_handler_thread.wait()
        #s = time.time()
        q_val = threading_queue.get()
        
        lock_worker_sync.acquire() # blocking because it should be the next process to aquire the lock
        for key, array in q_val.items():
            shm_array_dict[key][:] = array[:]
        lock_worker_sync.release()
        event_worker_sync.set() # now retrieve values in child processes
        
        #e = time.time()
        #print(f"Thread got and placed value in: {(e - s) * 1000} ms")
        event_handler_thread.clear()
    


def worker_func(queue_in, queue_out, worker_id, event_terminating, 
                metadata_worker, event_worker_sync, lock_worker_sync,
                metadata_display, event_display_sync, lock_display_sync):
    """
    feature map calculation worker process with shared memory (SHM) approach for faster procesing
    - write in arrays of "shm_array_dict" with lock from main handler thread
    - make a local copy of the dictionary with for-loop
    """
    # buffer arrays for feature maps
    shm_array_dict = {}
    local_array_dict = {}
    buff_list = []
    for key, metadata in metadata_worker.items():
        buff_display_name, shape, dtype = metadata
        existing_shm = mp.shared_memory.SharedMemory(name=buff_display_name)
        buff_list.append(existing_shm)
        shm_array_dict[key] = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        local_array_dict[key] = np.ndarray(shape, dtype=dtype)
    
    # buffer display
    buff_display_name, shape, dtype = metadata_display
    display_shm = mp.shared_memory.SharedMemory(name=buff_display_name)
    buff_list.append(display_shm)
    display_img = np.ndarray(shape, dtype=dtype, buffer=display_shm.buf)
    
    print(f"Calculation Worker with ID {mp.current_process()}")
    
    while True:
        # wait for call from main thread
        is_in_time = event_worker_sync.wait(timeout=2.0)
        
        # check for termination event
        if not is_in_time:
            if event_terminating.is_set():
                print(f"Stop process Calculation Worker with ID {mp.current_process()}")
                break
            else:
                continue
        
        # only one worker should acquire lock, others not be blocked
        ret = lock_worker_sync.acquire(block=False) 
        
        if ret is not False: # acquired lock
            event_worker_sync.clear() # other child processes should wait till event is set
            # make local copy
            for key, array in shm_array_dict.items():
                local_array_dict[key][:] = array[:]
            
            # copy done
            lock_worker_sync.release()
            
            # perform calculation
            #s = time.time()
            ret = calc_feature_maps_dataset(local_array_dict)
            #e = time.time()
            #print(f"Time for feature map calculation: {(e - s) * 1000} ms")
            
            # write display image
            lock_display_sync.acquire()
            display_img[:] = ret[:]
            event_display_sync.set()
            lock_display_sync.release()
    
    # clean up
    print(f"Closing SHM buffers in Calculation Worker with ID {mp.current_process()}")
    for buff in buff_list: 
        buff.close()


def calc_feature_maps_grid(img_grid, layer_inf):
    """
    - resizes and pads provided image grid for one layer
    Input: 
        - img_grid:      feature maps grid as NumPy array with 2 dimensions (width, height), derived from Yolo Model tensor collection
                         --> normalization and grid building done inside the customized Yolo model
        - layer_inf:     layer information for feature map (layer number, number of displayed outputs [max. 64], number of total outputs)
    Output: NumPy array - full resized and padded image grid for one provided layer with additional layer information
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
            - resizes and pads provided image grid for one layer
    - order feature map grids as one single grid in a large picture, ready to be displayed by any viewer 
      which accepts NumPy arrays
    
    Input: User-defined tensor collection dictionary added to the Yolo Model as attribute
        - data_dict:    tensor collection from Yolo Model
        - ncol:         number of columns in final output image
    Output: Single image as NumPy array with provided feature maps visualized
    """
    
    img_grid_list = []
    
    for layer_inf, tensor in data_dict.items():
        # get normalized grid PyTorch Yolo CPU tensor with 2 dimensions (width, height)
        # ONLY USE CPU TENSORS IN MULTIPROCESSING ENVIRONMENT - SHARING OF CUDA TENSORS NOT RELIABLE
        # transform to NumPy array
        img_grid_list.append(calc_feature_maps_grid(tensor, layer_inf))
    
    # re-arrange single feature map grids into one single image
    img_grid_array = np.stack(img_grid_list)
    nindex, height, width = img_grid_array.shape
    nrows = nindex // ncol
    img_grid = (img_grid_array.reshape(nrows, ncol, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncol))
    
    return img_grid


def display_results(queue_in, worker_id, event_terminating, metadata_display, event_display_sync, lock_display_sync):
    """
    feature map display process with shared memory (SHM) approach for faster procesing
    retrieve processed feature maps images from other worker processes via SHM
    """
    buff_display_name, shape, dtype = metadata_display
    existing_shm = mp.shared_memory.SharedMemory(name=buff_display_name)
    display_img = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    print(f"Display Worker with ID {mp.current_process()}")
    while True:
        # wait for call from worker processes
        is_in_time = event_display_sync.wait(timeout=2.0)
        
        # check for termination event
        if not is_in_time:
            if event_terminating.is_set():
                print(f"Stop process Display Worker with ID {mp.current_process()}")
                break
            else:
                continue
        
        # display image
        lock_display_sync.acquire()
        cv2.imshow("Feature Maps", display_img)
        cv2.waitKey(1)
        lock_display_sync.release()
        event_display_sync.clear()
    
    # clean up
    print(f"Closing SHM buffer in Display Worker with ID {mp.current_process()}")
    existing_shm.close()
    cv2.destroyAllWindows()


def program_intercept(key):
    """
    Keyboard listener action to control program execution, used to terminate program
    """
    global prog_exit
    if key == keyboard.Key.esc:
        # Stop listener
        print("ESC pressed. Stopping program execution...")
        prog_exit = True
        return False


def save_pickle(filename, obj):
    """save pickle object to path"""
    with open(str(filename), 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved under: '{filename}'")
        


def load_pickle(filename):
    """load pickle object from path"""
    with open(str(filename), 'rb') as f:
        obj = pickle.load(f)
    print("Object loaded")
    return obj


if __name__ == '__main__':
    """
    main program
    """
    
    # argument parser
    parser = argparse.ArgumentParser(
                    prog = 'YoloV7 Realtime Object Detection with Feature Map Visualization',
                    description = 'Displays Live Webcam Video feed with Feature Map Visualization')
    parser.add_argument('-d', '--device', choices=['cpu','gpu'], dest='device_type', default='cpu',
                    help='select device type for Yolo Model, performing on CPU or GPU')
    parser.add_argument('-c', '--conf', dest='conf_thres',
                    type=float, default=0.65,
                    help='confidence threshold for detection accuracy')
    parser.add_argument('-n', '--numproc', dest='num_proc',
                    type=int, default=3,
                    help='number of feature map calculation processes')
    parser.add_argument('-s', '--source', dest='source', default='0',
                    help='path as path or source to webcam as integer')
    parser.add_argument('-f', '--numFMStream', dest='n_th_frame', default=1, type=int,
                    help='every n_th frame is displayed as feature map visualization, default=1 --> realtime')
    parser.add_argument('-w', '--weights', dest='weights', default='data/yolov7.pt',
                    help='model weights')
                    
    args = parser.parse_args()
    
    
    # load model on selected device
    print("Loading and building model...")
    model, device = load_model(args.weights, device_type=args.device_type, save_init_tensor_collection=False)
    print("Model loaded.")
    
    # multiprocessing preparations
    print("Preparing multiprocessing environment")
    threading_queue = queue.Queue()
    mgr = mp.Manager()
    #mp_q_in = mgr.Queue()
    #mp_q_out = mgr.Queue()
    # main process synchronisation
    event_handler_thread = threading.Event() # synchronization of queue handler thread
    #event_terminating = mgr.Event() # used to securely stop child processes
    event_terminating = mp.Event() # used to securely stop child processes
    
    # worker process synchronization
    event_worker_sync = mp.Event()
    lock_worker_sync = mp.Lock()
    #event_worker_sync = mgr.Event()
    #lock_worker_sync = mgr.Lock()
    # shared memory for arrays in worker processes
    # load example data set with all necessary information
    print("Prepare shared memory space...")
    #tensor_collection_reference = load_pickle("data/tensor_collection_reference.pkl")
    tensor_collection_reference = model.tensor_collection
    buff_list = []
    metadata_worker = {}
    print("Collecting metadata...")
    for i, [key, array] in enumerate(tensor_collection_reference.items()):
        print(f"Entry {i+1} \t Shape: {array.shape} \t Byte size: {array.nbytes} \t dtype: {array.dtype}")
        buff_worker_name = f"worker_{i}"
        buff_size = int(array.nbytes * 2.2) # set byte size with security factor, not too large
        buff = mp.shared_memory.SharedMemory(create=True, size=buff_size, name=buff_worker_name)
        buff_list.append(buff) # save buffer to release it later
        print(f"Entry {i+1} \t Buffer created with size {buff_size} Bytes")
        metadata_worker[key] = (buff_worker_name, array.shape, array.dtype)
        print(f"Entry {i+1} \t Saved metadata")
    
    
    # display process
    buff_display_name = "display"
    buff_display = mp.shared_memory.SharedMemory(create=True, size=5000000, name=buff_display_name)
    buff_list.append(buff_display)
    metadata_display = (buff_display_name, (900,1800), np.uint8) # (buffer_name, shape, dtype)
    event_display_sync = mp.Event() # synchronize display process
    lock_display_sync = mp.Lock() # synchronize display process
    #event_display_sync = mgr.Event() # synchronize display process
    #lock_display_sync = mgr.Lock() # synchronize display process
    
    num_proc = args.num_proc # define number of worker processes for feature map calculation
    proc_list = []
    
    # queue handler QH
    thread_QH = threading.Thread(target=queue_handler, args=(threading_queue, mp_q_in, event_handler_thread, 
                    metadata_worker, event_worker_sync, lock_worker_sync), daemon=True)
    # keyboard listener to stop program, [ESC] configured
    listener = keyboard.Listener(on_press=program_intercept, daemon=True)
    
    print(f"Starting {num_proc} processes...")
    print("... for feature map calculation ...")
    for i in range(num_proc):
        p = mp.Process(target=worker_func, args=(mp_q_in, mp_q_out, i, event_terminating, 
                        metadata_worker, event_worker_sync, lock_display_sync,
                        metadata_display, event_display_sync, lock_display_sync))
        proc_list.append(p)
        p.start()
    print("... for Display ...")
    p = mp.Process(target=display_results, args=(mp_q_out, num_proc, event_terminating, metadata_display,
                    event_display_sync, lock_display_sync))
    proc_list.append(p)
    p.start()
    print("Processes started")
    
    print("Starting Queue Handler Thread...")
    thread_QH.start()
    print("Starting Keyboard Listener...")
    listener.start()
    print("To stop program use ESC key")
    print("Starting prediction...")
    with torch.no_grad():
        ret = prediction(args.source, model, device, mp_q_in, threading_queue=threading_queue, 
                        event_handler_thread=event_handler_thread, event_terminating=event_terminating,
                        n_th_frame=args.n_th_frame, save_img=False, conf_thres=args.conf_thres)
    
    print("Terminating processes...")
    for p in proc_list:
        p.join(5)
        if p.exitcode == None:
            p.terminate()
        p.close()
    
    # release shared memory allocations
    print("Releasing SHM Buffers...")
    for buff in buff_list:
        buff.close()
        buff.unlink()
    
    if args.device_type == 'gpu':
        print("Empty GPU Memory...")
        torch.cuda.empty_cache() # empty GPU memory used by PyTorch
    
    print("Program exit")