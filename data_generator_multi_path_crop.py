# -*- coding: utf-8 -*-
"""
Created on Oct 15 2020

@author: Y. Lin
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import random
import matplotlib.pyplot as plt

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])


class Generator(tf.keras.utils.Sequence):

    def __init__(self, trainset_path=[], labels_path=[], BATCH_SIZE=10, 
                 shuffle_images=True, n_class=2, image_size=(224,224), crop=True):
        
        assert trainset_path != []
        assert labels_path != []
        
        self.batch_size = BATCH_SIZE
        self.shuffle_images = shuffle_images
        self.load_image_paths_labels(trainset_path, labels_path)
        self.create_image_groups()
        self.n_class = n_class
        self.image_size = image_size
        self.min_crop_ratio = 0.1
        self.crop = crop
        
    def process_image(self,image ):
        '''
        Data augmentation

        Parameters
        ----------
        image : an input image

        Returns
        -------
        image : an output image being processed

        '''
        gamma = random.uniform(0.9, 1.1)
        bri = random.uniform(0.5, 2.0)
        image = image.copy() ** gamma
        image = np.clip(image, 0, 255)
        image = image.copy() * bri
        image = np.clip(image, 0, 255)
        image = (image / 255. - rgb_mean) / rgb_std    
        return image
    
    
    def load_image_paths_labels(self, trainset_path,  labels_path):
        '''
        Load images' and label's path into 
        self.image_path and self.image_labels 
        
        This function is called only once when instance
        
        Note:
        if len(DATASET_PATH) > 1, make sure that the labels path are 
        fed as the order for that of data set
        
        example:
        DATASET_PATH = ["./dataset/train_set1", "./dataset/train_set2"]
        LABELS_PATH = ["./dataset/label_set1", "./dataset/label_set2"]
        
        '''
        
        self.image_paths = []
        self.image_labels = []
        for i, dataset in enumerate(trainset_path):
            for image_file_name in os.listdir(dataset):
                self.image_paths.append(os.path.join(dataset, image_file_name))
                self.image_labels.append(os.path.join(labels_path[i], image_file_name.split(".")[0]+".txt"))
        # assert len(self.image_paths) == len(self.image_labels)

    def create_image_groups(self):
        if self.shuffle_images:
            # Randomly shuffle dataset
            seed = 10
            np.random.seed(seed)
            np.random.shuffle(self.image_paths)
            np.random.seed(seed)
            np.random.shuffle(self.image_labels)

        # Divide image_paths and image_labels into groups of BATCH_SIZE
        self.image_groups = [[self.image_paths[x % len(self.image_paths)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.image_paths), self.batch_size)]
        self.label_groups = [[self.image_labels[x % len(self.image_labels)] for x in range(i, i + self.batch_size)]
                              for i in range(0, len(self.image_labels), self.batch_size)]

    def random_crop(self, image, label): 
        '''
        Randomly crop image to augment training set
        
        Parameters
        ----------
        image : np.array, shape = (w,h,3)
            input image.
        label : np.array, shape = (w,h)
            input label.

        Returns
        -------
        image : np.array, shape = (w_new,h_new,3)
            cropped iamge.
        label : np.array, shape = (w_new,h_new)
            cropped label.

        '''
        if np.sum(label) == 0:
            return image, label
        shape = label.shape
        [max_idy, max_idx] = [np.max(e) for e in np.where(label > 0)]
        [min_idy, min_idx] = [np.min(e) for e in np.where(label > 0)]
        w = random.uniform(self.min_crop_ratio,1.0) * shape[1]
        h = random.uniform(self.min_crop_ratio,1.0) * shape[0]
        
        bbox_l = int(max(0, min_idx - w//2))
        bbox_r = int(min(shape[1], max_idx + w // 2))
        bbox_u = int(max(0, min_idy - h//2))
        bbox_b = int(min(shape[0], max_idy + h // 2))
        
        # print(min_idx, max_idx, min_idy, max_idy)
        # print(bbox_l, bbox_r, bbox_u, bbox_b)
        
        image = image[bbox_u:bbox_b, bbox_l:bbox_r].copy()
        label = label[bbox_u:bbox_b, bbox_l:bbox_r].copy()
        return image, label
        
    def load_images(self, image_group, label_group):
        '''
        Read images and labels from paths

        Parameters
        ----------
        image_group : type = [str], size = (batch_size)
            an array containing images' path in a batch
        label_group : type = [str], size = (batch_size)
            an array containing labels' path in a batch.

        Returns
        -------
        images : type = np.array , size = (batch_size, w, h, 3)
            an array containing images in a batch.
        labels : type = np.array , size = (batch_size, w, h)
            an array containing labels in a batch.

        '''
        images = []
        labels = []
        for i, image_path in enumerate(image_group):
            image = cv2.imread(image_path, 1)
            image = self.process_image(image)            
            label_path = label_group[i]
            
            with open(label_path) as file:
                label = np.array([[float(digit) 
                                    for digit in line.split()] 
                                    for line in file])
                
            if self.crop == True:
                image, label = self.random_crop(image, label)
            image = cv2.resize(image, self.image_size,interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, self.image_size,interpolation=cv2.INTER_NEAREST)
            label = np.ceil(label)
            label = cv2.dilate(label, kernel = np.ones((5,5)), iterations  = 1)
            labels.append(self.expand_label_info(label))
            images.append(image)
            
        return images, labels

    def expand_label_info(self,label):
        '''
        Reshape a 2D label array to 1D
        
        Parameters
        ----------
        label : input label, size = (w, h)

        Returns
        -------
        label_out : output label, size = (1, w * h)

        '''
        label_out = np.zeros(shape = [self.image_size[0], self.image_size[1], self.n_class])
        for c in range(self.n_class):
            label_out[:, :, c] = (label == c).astype(int)
        label_out = np.reshape(label_out, (-1, self.n_class))
        return label_out

    def batch_image(self, image_group, label_group):
        # get the max image shape
        max_shape = [max(image.shape[x] for image in image_group) for x in range(3)]
        max_shape_label = [max(label.shape for label in label_group)]
        
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype='float32') 
        label_batch = np.zeros((self.batch_size,) + max_shape_label, dtype='float32')

        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image
            label_batch[image_index, :label_group[image_index].shape[0], :label_group[image_index].shape[1]] = label_group[image_index]
        return image_batch, label_batch
    
    def __len__(self):
        """
            Used for define steps_per_epoch
        """

        return len(self.image_groups)

    def __getitem__(self, index):
        """
            where generator feeds data to model.fit / fit_generator
        """
        image_group = self.image_groups[index]
        label_group = self.label_groups[index]
        images, labels = self.load_images(image_group, label_group)
        image_batch, label_batch = self.batch_image(images, labels)

        return np.array(image_batch), np.array(label_batch)

if __name__ == "__main__":

    image_size = (320,320)
    BASE_PATH = 'dataset'
    batch = 5
    train_generator = Generator(['./dataset/FCN_500_imageset', './dataset/FCN_imageset'],
                                ['./dataset/FCN_500_refine_txt', './dataset/FCN_refine_txt'],
                                BATCH_SIZE=batch, 
                                n_class=2, 
                                image_size=image_size,
                                crop=False)
    print(len(train_generator))
    image_batch, label_group = train_generator.__getitem__(20)
    print(image_batch.shape)
    print(label_group.shape)
    
    for i in range(batch):
        plt.figure()
        plt.imshow(image_batch[i,:,:,0])
        plt.figure()
        plt.imshow(np.reshape(label_group[i,:,1],image_size))
        
