# -*- coding: utf-8 -*-
"""
Created on Oct 20 2020

@author: Y. Lin
"""

import os
# if you want to show tensorflow's log, mark this line
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.compat.v1.enable_eager_execution() 

# Remember to limit GPU's memory by not over 0.7
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
config = tf.compat.v1.ConfigProto(log_device_placement=False,gpu_options=gpu_options)
sess = tf.compat.v1.Session(config=config)

from config import cfg
from model import FCN
from data_generator_multi_path_crop import Generator
import datetime

OPTIMIZER = [tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
             tf.keras.optimizers.Adam(lr=0.0001)]

def train(model, train_generator, epochs = 60):
    
    model.compile(optimizer=OPTIMIZER[cfg.TRAIN.OPTIMIZER],
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                          histogram_freq=1, 
                                                          update_freq = 30)
    
    callback = tf.keras.callbacks.ModelCheckpoint("FCN%ds.h5"%cfg.TRAIN.FCN_TYPE, 
                                                  verbose=2, 
                                                  save_weights_only=True)
    step_per_epoch = len(train_generator)
    model.fit_generator(generator=train_generator,
                                    steps_per_epoch=step_per_epoch,
                                    epochs=epochs,
                                    callbacks=[callback,tensorboard_callback])


if __name__ == "__main__":
    image_size = cfg.TRAIN.IMG_SIZE
    nClasses = cfg.TRAIN.N_CLASSES

    model = FCN(nClasses, image_size[0], image_size[1], cfg.TRAIN.FCN_TYPE)
    print ("==> Model initialized, Model info:")
    print(model.summary())
    print("Layers of model: %d"%len(model.layers))

    BATCH_SIZE= cfg.TRAIN.BATCH_SIZE
    train_generator = Generator(cfg.TRAIN.DATASET,
                                cfg.TRAIN.LABELSET, 
                                BATCH_SIZE=BATCH_SIZE, 
                                n_class=nClasses, 
                                image_size = image_size)

    train(model, train_generator, epochs=cfg.TRAIN.EPOCHS)
