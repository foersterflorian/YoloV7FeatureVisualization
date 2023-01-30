# **YoloV7FeatureVisualization**
Visualization of pre-chosen YoloV7 layer feature maps
## Functions
- displaying a live webcam feed with object prediction based on the [YoloV7 model](https://github.com/WongKinYiu/yolov7) with pre-trained weights on the COCO dataset
- additional displaying of the model's feature maps for 8 pre-chosen convolutional layers ``(0,1,2,24,59,84,99,104)``
- all information in processed in real-time with 30 FPS (tested on a NVidia RTX 4070Ti)
- perform single image prediction if source is provided as path

## Installation of Conda Environment
Install environment with Conda environment file
``$ conda env create -f env.yaml``
The name of the environment is set to ``YoloFMV``

## Overview of program options (flags)
- ``-d, --device``: select device type where model should be executed (CPU or GPU) (default: ``cpu``)
- ``-c, --conf``: confidence threshold for predictions, only predictions above this threshold are visualized (default: ``0.65``)
- ``-n, --numproc``: number of worker processes for feature map post-processing (default: ``3``)
- ``-s, --source``: path to source file (for single image) or video pipe (integer in OpenCV manner) (default: ``0``)
- ``-f, --numFMStream``: every Nth frame is displayed as feature map visualization (default: ``1`` = realtime)
- ``-w, --weights``: pre-trained model weights (default: ``data/yolov7.pt``)
- ``-i, --datasetinfo``: information about trained model data like class names as YAML file (default: ``data/coco.yaml``)