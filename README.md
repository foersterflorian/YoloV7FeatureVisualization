# **Yolo V7 Feature Map Visualization**
Visualization of pre-chosen Yolo V7 layer feature maps
## Functions
- displaying a live webcam feed with object prediction based on the [YoloV7 model](https://github.com/WongKinYiu/yolov7) with pre-trained weights on the COCO dataset
- additional displaying of the model's feature maps for 8 pre-chosen convolutional layers ``(0,1,2,24,59,84,99,104)``
- all information is processed in real-time with 30 FPS (tested on an Nvidia RTX 4070 Ti)
- perform single image prediction if source is provided as path to an image file

## Installation of Conda Environment
Install environment with Conda environment file

``$ conda env create -f env.yaml``

The name of the environment is set to ``YoloFMV``. This can be changed in the YAML configuration file.

## Overview of program options (flags)
- ``-d, --device``: select device type where model should be executed (CPU or GPU) (default: ``gpu``)
- ``-c, --conf``: confidence threshold for predictions, only predictions above this threshold are visualized (default: ``0.65``)
- ``-n, --numproc``: number of worker processes for feature map post-processing (default: ``3``)
- ``-s, --source``: path to source file (for single image) or video pipe (integer in OpenCV manner) (default: ``0`` =  primary webcam's feed)
- ``-f, --numFMStream``: every Nth frame is displayed as feature map visualization (default: ``1`` = process every frame --> realtime)
- ``-w, --weights``: pre-trained model weights (default: ``data/yolov7.pt``)
- ``-i, --datasetinfo``: path to the information about trained model data like class names as YAML file (default: ``data/coco.yaml``)
- ``--ncol``: total number of columns in final display image (choice from ``{1,2,4,8}``) (default: ``8``)
- ``--ncolFM``: total number of columns in the feature maps grid (choice from ``{1,2,4,8}``) (default: ``4``)
- ``--height``: total height of final display image (default: ``1060``)
- ``--width``: total width of final display image (default: ``3840``)