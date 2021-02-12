# -*- coding: utf-8 -*-
"""
Created on Oct 20 2020

@author: Y. Lin
"""

from easydict import EasyDict

_CFG = EasyDict()
cfg = _CFG


# Config. for training
# Retrieve value by cfg.TRAIN.XXXX

_CFG.TRAIN                    = EasyDict() 

_CFG.TRAIN.IMG_SIZE           = (224,224)
_CFG.TRAIN.BATCH_SIZE         = 5
_CFG.TRAIN.EPOCHS             = 60  
_CFG.TRAIN.FCN_TYPE           = 2    
_CFG.TRAIN.N_CLASSES          = 2 
_CFG.TRAIN.OPTIMIZER          = 0 # 0: Adelta, 1: Adam 
_CFG.TRAIN.DATASET            = ['./dataset/FCN_500_imageset', './dataset/FCN_imageset'] 
_CFG.TRAIN.LABELSET           = ['./dataset/FCN_500_refine_txt', './dataset/FCN_refine_txt'] 

# Config. for evaluation
# Retrieve value by cfg.EVAL.XXXX

_CFG.EVAL                     = EasyDict() 

_CFG.EVAL.USE_CPU             = True
_CFG.EVAL.INPUT_SIZE          = 320
_CFG.EVAL.FIRST_LAYER_SIZE    = 320
_CFG.EVAL.FCN_TYPE            = 2  
_CFG.EVAL.N_CLASSES           = 2 
_CFG.EVAL.THRESHOLD           = 0.1
_CFG.EVAL.MAX_BATCH_SIZE      = 2
