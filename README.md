# Tensorflow-2.0-FCN
This repository contains an example of a FCN implementation based on Tensorflow 2.X


Below is an example of a dust-screening result based on FCN2s
![image](https://github.com/ylin1992/Tensorflow-2.0-FCN/blob/main/validation_result/tst_result.png)

## Installation
Install packages from requirements.txt
```pip3
$ pip3 install -r ./packages/requirements.txt
```

## How to use
### Folder structure
```bash
.
├── dataset
│   ├── FCN_500_imageset                # raw images of dataset, it's ok to rename the folder as you wish, remember to modify the data path in config.py
│   └── FCN_500_refine_txx              # 2-D txt labels, it's ok to rename the folder as you wish, remember to modify the data path in config.py
├── validation_data                     # input images for evaluation
├── validation_result                   # evaluation result
├── logs
│   └── fit                             # tensorbaord logs directory
├── packages
│   └── requirements.txt
├── config.py
├── data_generator_multi_path_crop.py
├── model.py
├── train.py
└── validation.py
```
## Training
### Setup configuration
Specify your own data set and training paramenter in ```config.py```

```python3
_CFG.TRAIN.IMG_SIZE           = (224,224)
_CFG.TRAIN.BATCH_SIZE         = 5
_CFG.TRAIN.EPOCHS             = 60  
_CFG.TRAIN.FCN_TYPE           = 2    
_CFG.TRAIN.N_CLASSES          = 2 
_CFG.TRAIN.OPTIMIZER          = 0 # 0: Adelta, 1: Adam 
_CFG.TRAIN.DATASET            = ['./dataset/FCN_500_imageset', './dataset/FCN_imageset'] 
_CFG.TRAIN.LABELSET           = ['./dataset/FCN_500_refine_txt', './dataset/FCN_refine_txt'] 
```
Notice: 
  * ```DATASET``` and ```LABELSET``` support list of input images
  * labels are 2D array recorded in text files, each class is denoted as corresponding value

### Train your model
#### Dataset
Dataset can be stored in wherever you want, the py supports multiple paths input, which means you are allowed to input multiple dataset stored in different paths.  Just to be careful that all data and label have to be set as the same order.
In this example, I put a demo image in `./dataset/FCN_500_imageset`, and the corresponding label is stored in `./dataset/FCN_500_refine_txt`
 
 So in this case, config file should be set as:
```python3
_CFG.TRAIN.DATASET            = ['./dataset/FCN_500_imageset']
_CFG.TRAIN.LABELSET           = ['./dataset/FCN_500_refine_txt']
```
Please refer to `./dataset` for details on how to setup your own dataset

#### Start your own training set
simply input below comment in the directory
```bash
$ python3 train.py

6725/6725 [==============================] - ETA: 0s - loss: 0.0131 - accuracy: 0.9964     
Epoch 00001: saving model to FCN2s.h5
6725/6725 [==============================] - 16032s 2s/step - loss: 0.0131 - accuracy: 0.9964
Epoch 2/60
6725/6725 [==============================] - ETA: 0s - loss: 0.0062 - accuracy: 0.9980     
Epoch 00002: saving model to FCN2s.h5
...
Epoch 60/60
6725/6725 [==============================] - ETA: 0s - loss: 0.0023 - accuracy: 0.9992     
Epoch 00060: saving model to FCN2s.h5

```
The model is saved as `FCN[n]s.h5` in the root folder, where `n = FCN_TYPE` 

### Trace your training by Tensorboard
`train.py` sets updating frequency of Tensorboard's callback as 30 steps, you can adjust it as you want by modifying below code.  Note that the more frequent the callback is triggered, the slower training rate you get.
```python3
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1, 
                                                      update_freq = 30)
```
In the root directory, type below comments in terminal to activate tensorboard
```bash
$ tensorboard --logdir="./logs/fit"
```
Lauch a browser and type `http://localhost:6006/` in URL, then check the visulaized trainging process
![image](https://github.com/ylin1992/Tensorflow-2.0-FCN/blob/main/packages/tensorboard.png)
## Evaluate your own model
### Setup configuration
Specify your own data set and training paramenter in ```config.py```
```python3
_CFG.EVAL.USE_CPU             = True          # if you want to evaluate result in the middle of training process, set this parameter as True.
                                              # otherwise, set this value as False to use GPU instead
_CFG.EVAL.INPUT_SIZE          = 320
_CFG.EVAL.FIRST_LAYER_SIZE    = 320
_CFG.EVAL.FCN_TYPE            = 2             # FCN_TYPE has to be set as 2, 4, or 8 and should be the same as configuration for training
_CFG.EVAL.N_CLASSES           = 2 
_CFG.EVAL.THRESHOLD           = 0.5
_CFG.EVAL.MAX_BATCH_SIZE      = 2             # in validation.py, an input image is sliced into small fragments and zipped into batch
                                              # adjust this parameter based on your GPU RAM
                                              # i.e., I used GTX1660 Ti, 6GB GRAM, FCN2s with first layer = (224,224,3), this parameter
                                              # can be up to 20
```
### Evaluate the model
Put your data in `./validation_data` and evaluate the result by typing the comment as follows
```bash
$ python3 validation.py
```
evaluation result will be saved in ```./validation_result```
