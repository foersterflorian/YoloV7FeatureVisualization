#!/usr/bin/env python
# coding: utf-8


import YoloDemonstration as yld



if __name__ == '__main__':
    print("Load and build model...")
    model, device = yld.load_model('yolov7.pt', device_type='cpu')
    print("Model loaded.")
    ret = yld.prediction('1', model, device, save_img=False)