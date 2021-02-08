# -*- coding: utf-8 -*-
"""
Created on Oct 20 2020

@author: Y. Lin
"""

from easydict import EasyDict as edict

_C = edict()
cfg = _C


# Config. for training
# Retrieve value by cfg.TRAIN.XXXX

_C.TRAIN                    = edict() 

_C.TRAIN.IMG_SIZE           = (224,224)
_C.TRAIN.BATCH_SIZE         = 5
_C.TRAIN.EPOCHS             = 60  
_C.TRAIN.FCN_TYPE           = 2    
_C.TRAIN.N_CLASSES          = 2 
_C.TRAIN.OPTIMIZER          = 0 # 0: Adelta, 1: Adam 
_C.TRAIN.DATASET            = ['./dataset/FCN_500_imageset', './dataset/FCN_imageset'] 
_C.TRAIN.LABELSET           = ['./dataset/FCN_500_refine_txt', './dataset/FCN_refine_txt'] 

# Config. for evaluation
# Retrieve value by cfg.EVAL.XXXX

_C.EVAL                     = edict() 

_C.EVAL.USE_CPU             = True
_C.EVAL.INPUT_SIZE          = 320
_C.EVAL.FIRST_LAYER_SIZE    = 320
_C.EVAL.FCN_TYPE            = 2  
_C.EVAL.N_CLASSES           = 2 
_C.EVAL.THRESHOLD           = 0.1
_C.EVAL.MAX_BATCH_SIZE      = 2
