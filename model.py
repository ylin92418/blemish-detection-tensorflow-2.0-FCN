# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:02:10 2021

@author: A85K
"""

from keras.applications import vgg16
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, add, Dropout, Reshape, Activation


def FCN(nClasses, input_height, input_width, FCN_stage = 2):
    '''
    An augmentation of fine-features learning model,
    generating fcn2s, fcn4s and fcn8s by declaring FCN_stage

    Parameters
    ----------
    nClasses : number of classes

    input_height : int
        image height of first layer of VGG-16.
    input_width : int
        image width of first layer of VGG-16.
    FCN_stage : FCN stage number, optional
        declare the number of fcn net. The default is 2.

    Returns
    -------
    fcn_model : Keras.model
        output model.

    '''
    
    assert FCN_stage == 2 or FCN_stage == 4 or FCN_stage == 8
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    model = vgg16.VGG16(include_top=False,
                        weights='imagenet', 
                        input_tensor=img_input,
                        pooling=None,
                        classes=1000)
    
    assert isinstance(model, Model)

    vgg_out = Conv2D(filters=4096, kernel_size=(7,7), padding="same",
                     activation="relu",name="fc6")(model.output)
    
    vgg_out = Dropout(rate=0.5)(vgg_out)
    
    vgg_out = Conv2D(filters=4096,kernel_size=(1,1),padding="same",
                     activation="relu",name="fc7")(vgg_out)
    
    vgg_out = Dropout(rate=0.5)(vgg_out)
    
    vgg_out = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", 
                     activation="relu", kernel_initializer="he_normal",name="score_fr")(vgg_out)
    
    vgg_out = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", 
                              activation=None,name="score2")(vgg_out)

    fcn8 = Model(inputs=img_input, outputs=vgg_out)
    

    skip_con1 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool4")(fcn8.get_layer("block4_pool").output)
    fuse_1 = add(inputs=[skip_con1, fcn8.output])

    # 8x upsampling

    x_upsmaple8 = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2),
                        padding="valid", activation=None,name="score4")(fuse_1)

    skip_con2 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool3")(fcn8.get_layer("block3_pool").output)
    fuse_2 = add(inputs=[skip_con2, x_upsmaple8])
    
    if FCN_stage == 8:
        output = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8),
                         padding="valid", activation=None, name="upsample")(fuse_2)
        output = Reshape((-1, nClasses))(output)
        output = Activation("softmax")(output)
        fcn_model = Model(inputs=fcn8.input, outputs=output)
        
        return fcn_model
    
    # 4x unpsampling layer
    
    x_upsample4 = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2),
                                 padding="valid", activation=None, name="score5")(fuse_2)   
    skip_con3 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool2")(fcn8.get_layer("block2_pool").output)   
    fuse_3 = add(inputs=[skip_con3, x_upsample4])
    
    if FCN_stage == 4:
        output = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4),
                              padding="valid", activation=None, name="upsample")(fuse_3)
        output = Reshape((-1, nClasses))(output)
        output = Activation("softmax")(output)
        fcn_model = Model(inputs=fcn8.input, outputs=output)
        
        return fcn_model
    
    # 2x upsampling layer
    
    x_upsample2 = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score6")(fuse_3)
    skip_con4 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                        name="score_pool1")(fcn8.get_layer("block1_pool").output)
    fuse_4 = add(inputs=[skip_con4, x_upsample2])

    output = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2),
                          padding="valid", activation=None, name="upsample_2x")(fuse_4)
    output = Reshape((-1, nClasses))(output)
    output = Activation("softmax")(output)

    fcn_model = Model(inputs=fcn8.input, outputs=output)

    return fcn_model


if __name__ == '__main__':
    import os
    os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'

    m = FCN(15, 224, 224, FCN_stage=2)
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='model_fcn8.png')
    print(m.summary())
    print(len(m.layers))