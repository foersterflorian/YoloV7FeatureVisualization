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


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models, transforms
import yaml
import time
import threading
import multiprocessing as mp
#import multiprocess as mp

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import LoadImages, letterbox
from utils.plots import plot_one_box
from utils.torch_utils import intersect_dicts, time_synchronized
from models.common import Conv
from models.yolo import Model
#from models.yolo_old import Model

import sys
import FeatureMapCalculation

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
        time.sleep(5) # load webcam stream

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


class LoadWebcam:  # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set frame size 720p
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30.0) # set FPS

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if isinstance(self.pipe, int):  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, f'Camera Error {self.pipe}, Could not retrieve images'
        img_path = 'webcam.jpg'
        #print(f'webcam {self.count}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


def load_model(model_path, data_path='data/coco.yaml', device_type='cpu'):
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
    model = Model(ckpt['model'].yaml, ch=3).to(device) # build model with class Model from Yolo project with user-modified properties
    
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
        model.half()  # to FP16
     
    
    return model, device


def prediction(source, model, device, mp_pool, img_size=640, save_img=False, output_img='output/detect.jpg', use_colab=False, conf_thres=0.25):
    """
    TESTING, NOT FINISHED YET
    ONLY RETURNS ONE TENSOR COLLECTION FOR TESTING OF VEDO RENDERING
    
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
    n_th_frame = 2 # only broadcast feature maps of every n-th frame
    
    t0 = time.time()
    # for single images only length 1
    for path, img, im0s, vid_cap in dataset:
    
        ######################
    
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
        ########################### CALCULATION OF FEATURE MAPS ################################
        if (num_iter % n_th_frame) == 0:
            ret_pool = mp_pool.apply_async(func=FeatureMapCalculation.calc_feature_maps_dataset, args=(model.tensor_collection,))
        
        """
        test = model.tensor_collection
        s = time.time()
        ret = FeatureMapCalculation.calc_feature_maps_dataset(test)
        e = time.time()
        print(f"Main Process execution time: {(e-s) * 1000} ms")
        """
        
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()
        ##################################################################print(f"Interference Time: {(t2 - t1) * 1000} ms")
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                #p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                p, s, im0, frame = path, '', im0s, dataset.count
                
                ######################################
                #cv2.imshow('TEST', im0)
                #cv2.waitKey(1)
                #num_iter += 1
                #if num_iter < 300:
                #    time.sleep(1)
                #    continue
                
                
                
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
            ##################################################################print(f"Time for image post-processing: {(time.time() - t2) * 1000} ms")
            ##################################################################print(f"Time for interference & image post-processing: {(time.time() - t1) * 1000} ms")
            
            # mp pool result
            feature_map = ret_pool.get()
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
                if webcam: # webcam support in dedicated class
                    pass
                else:
                    if save_img:
                        cv2.imwrite(output_img, im0)
                        print(f"Detection image saved under: {output_img}")
                    cv2_imshow(im0)
        
        num_iter += 1
        
    if not use_colab:
        cv2.destroyAllWindows()
    
       
    return True
    
    
    """
    TESTING, NOT FINISHED YET
    ONLY RETURNS ONE TENSOR COLLECTION FOR TESTING OF VEDO RENDERING
    
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
                
                ######################################
                #cv2.imshow('TEST', im0)
                #cv2.waitKey(1)
                #num_iter += 1
                #if num_iter < 300:
                #    time.sleep(1)
                #    continue 
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

    
if __name__ == '__main__':
    """
    ******* TODO ********
    - use flags for device type
    - use flag for confidence threshold
    - empty cache for GPU mode
    """
    #from mp_test import worker_func
    #import DisplayVedo
    import FeatureMapCalculation
    import DisplayFeatureMaps
    """
    import psutil
    p = psutil.Process()
    print(f"Child #{mp.current_process()}: {p}, affinity {p.cpu_affinity()}")
    p.cpu_affinity([0])
    """
    
    print("Load and build model...")
    #model, device = load_model('yolov7.pt', device_type='cpu')
    model, device = load_model('yolov7.pt', device_type='gpu')
    print("Model loaded.")
    
    # multiprocessing preparations
    print("Prepare multiprocessing environment")
    num_proc = 6
    mp_pool = mp.Pool(processes=num_proc)
    print("Pool created")

    
    print("Start prediction")
    with torch.no_grad():
        ret = prediction('0', model, device, mp_pool, save_img=False, conf_thres=0.6)
    
    print("Terminating processes...")
    mp_pool.terminate()
    mp_pool.close()
    
    torch.cuda.empty_cache()
    print("Program exit")