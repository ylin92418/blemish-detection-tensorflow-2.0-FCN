# -*- coding: utf-8 -*-
"""
Created on Oct 19 2020

@author: Y. Lin
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from config import cfg
import numpy as np
import matplotlib.pyplot as plt


if cfg.EVAL.USE_CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        print('=> Using GPU')
    else:
        print("=> Using CPU")
else:
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.compat.v1.ConfigProto(log_device_placement=False,gpu_options=gpu_options)
    sess = tf.compat.v1.Session(config=config)
    
from model import FCN

INPUT_SIZE          = cfg.EVAL.INPUT_SIZE
FIRST_LAYER_SIZE    = cfg.EVAL.FIRST_LAYER_SIZE
MAX_BATCH_SIZE      = cfg.EVAL.MAX_BATCH_SIZE
FCN_TYPE            = cfg.EVAL.FCN_TYPE
n_class             = cfg.EVAL.N_CLASSES 
THR                 = cfg.EVAL.THRESHOLD  

def visualize_prob_map(image, label_images, thr):
    '''
    Input:
        image = input image, type: uint8, size = [im_w, im_h, c]
        label_images = output image for single class, size = [num_of_classes, im_h, im_w]
                     pixel value = 0~1, float
        thr = prob. threshold for truncating unwanted pixel in prob. map
    '''
    canvas = np.zeros(shape = (image.shape[0],3*image.shape[1],3))
    left_image = image.copy()
    right_image = image.copy()
    middle_image = np.zeros_like(right_image).astype(float)
    for i in range(0,label_images.shape[-1]):
        if i == 0: continue
        right_image[label_images[:,:,i] > thr, i] = 255
        middle_image[:,:,i] = label_images[:,:,i]
    
    # for i, label_image in enumerate(label_images):
    #     right_image[label_image > thr, i] = 255

    middle_image = (middle_image*255).astype(np.uint8)
    
    canvas[:,0:image.shape[1],:] = left_image
    canvas[:,image.shape[1]:image.shape[1]*2,:] = middle_image
    canvas[:,image.shape[1]*2:image.shape[1]*3,:] = right_image
    
    return canvas
    
def ini_model():
    image_size = (FIRST_LAYER_SIZE,FIRST_LAYER_SIZE)
    model = FCN(n_class, image_size[0], image_size[1], FCN_stage = FCN_TYPE)
    model.load_weights("FCN8s.h5")
    
    return model

if __name__ == "__main__":
    
    model = ini_model()

    ## test a image with larger image size, slicing images into small pieces
    ## and feeding into the model in batches to save time.
    ## Note: if doing so, use GPU to run the program
    
    p = "./validation_data"
    for eval_data in (os.listdir(p)):
        eval_data_path = os.path.join(p,eval_data)
        image = cv2.imread(eval_data_path) #Read the test image
        h, w, _ = image.shape
        
        num_fra = int(np.ceil(h/INPUT_SIZE))*int(np.ceil(w/INPUT_SIZE))
        stack_image_arr = np.zeros( (int(np.ceil(num_fra / MAX_BATCH_SIZE)),
                                    MAX_BATCH_SIZE,
                                    FIRST_LAYER_SIZE,
                                    FIRST_LAYER_SIZE,
                                    3))
        pred_bbox = np.zeros( (num_fra,
                                FIRST_LAYER_SIZE * FIRST_LAYER_SIZE,
                                n_class))
        # segmantize all images into a stack
        
    
        arr_idx = 0
        fra = 0
        for i in range(0,h,INPUT_SIZE):
            if i+INPUT_SIZE > h:
                idy = h-INPUT_SIZE
            else:
                idy = i
            for j in range(0,w,INPUT_SIZE):
                if j+INPUT_SIZE > w:
                    idx = w-INPUT_SIZE
                else:
                    idx = j
                roi = image[idy:idy+INPUT_SIZE, idx:idx+INPUT_SIZE, :]
                stack_image_arr[fra, arr_idx] = cv2.resize(roi, 
                                                    (FIRST_LAYER_SIZE
                                                    , FIRST_LAYER_SIZE))
                arr_idx += 1
                if arr_idx >= MAX_BATCH_SIZE:
                    arr_idx = 0
                    fra += 1
                    
        # batch all images
        # Note: stack_image_arr = [batch, images_num_per_batch, input.shape]
        idx = 0
        for i in range(0, stack_image_arr.shape[0]):
            tmp =  model.predict(stack_image_arr[i])
            if idx + tmp.shape[0] > num_fra:
                pred_bbox[idx : idx + tmp.shape[0]] = tmp[0 : num_fra - idx - tmp.shape[0]]
            else:
                pred_bbox[idx : idx + tmp.shape[0]] = tmp
            idx += tmp.shape[0]
            
        # unpack images
        label_image = np.zeros(shape = [h, w, n_class])
        arr_idx = 0
        for i in range(0,h,INPUT_SIZE):
            if i+INPUT_SIZE > h:
                idy = h-INPUT_SIZE
            else:
                idy = i
            for j in range(0,w,INPUT_SIZE):
                if j+INPUT_SIZE > w:
                    idx = w-INPUT_SIZE
                else:
                    idx = j
                label_1d = pred_bbox[arr_idx]
                for c in range(n_class):
                    label_reshape = np.reshape(label_1d[:,c], 
                                                (FIRST_LAYER_SIZE, FIRST_LAYER_SIZE))
                label_image[idy:idy+INPUT_SIZE, idx:idx+INPUT_SIZE, c] = cv2.resize(
                    label_reshape, (INPUT_SIZE, INPUT_SIZE))
                
                arr_idx += 1
                
        canvas = visualize_prob_map(image, label_image, THR)
        cv2.imwrite("./validation_result/0206_ep11_tst_result_thr_%d_%d_%f_%s"%
                                (INPUT_SIZE,FIRST_LAYER_SIZE,THR,p.split('/')[-1]), canvas)
        