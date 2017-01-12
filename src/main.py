#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:56:59 2016

@author: aitor / marks
"""

import os
import nnmodel
import utils
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.visualize_util import plot
from datetime import datetime
import matplotlib.pyplot as plt
import cv2



# Some utils
# TODO - Move to utils
def graph_training_loss_history(filename, losses):
    #plt.figure(figsize=(6, 3))
    plt.figure()
    plt.plot(losses)
    plt.ylabel('error')
    plt.xlabel('batch')
    plt.title('training error')
    plt.savefig(filename)
    #plt.show()

################################################################################
# Generator for Keras over jpeg dataset (@ai-tor/@marks)
################################################################################
def data_generator(batchsize, image_list, get_speed = True, img_transpose = True, resize = True,
                   crop = False, min_speed = 4, min_angle = 0.1, straight_road_prob = 0.2, label = 'Train'):


    print "data_generator(): label {}, batchsize {}, image_list.shape {}".format(label, batchsize, image_list.shape)
    if resize == False:
        width, height = 640, 480
    # Else go with the values set in utils
    else:
        width, height = 200, 66

    # while 1:
    if img_transpose == True:
        x = np.zeros((batchsize, ch, width, height), dtype=np.uint8)
    else:
        x = np.zeros((batchsize, height, width, ch), dtype=np.uint8)

    y = np.zeros(batchsize)     # Steering
    z = np.zeros(batchsize)     # Speed

    i = 0
    while True:
        for idx in range(len(image_list)):

            steering = image_list.at[idx, 'angle']
            speed = image_list.at[idx, 'speed']
            imagepath = os.path.join(image_list.at[idx, 'imagepath'], image_list.at[idx, 'filename'])
            image = cv2.imread(imagepath)
            #Crop before resize!!
            if crop == True:
                # im[y1:y2, x1:x2]
                image = image[200:480, 0:640]
            img = cv2.resize(image, (width, height))

            r = np.random.uniform(0, 1)
            if ((abs(steering) > min_angle) and speed >= min_speed) or (r < straight_road_prob):
                if img_transpose == True:
                    x[i, :, :, :] = img.transpose(2,1,0)   # Transpose the image to fit into the CNN later...
                else:
                    x[i, :, :, :] = img

                #x[i, :, :, :] = img.transpose(2,1,0)   # Transpose the image to fit into the CNN later...
                y[i] = float(steering)
                z[i] = float(speed)
                #print ("{} {}: Steering {} / File {}".format(label, idx, y[i], imagepath))
                i = i + 1

            if (i == batchsize):
                i = 0
                # print "x: {} / y: {}".format(x, y)
                if get_speed == True:
                    yield (x, y, z)
                else:
                    yield (x, y)

                    ################################################################################
# Generator for Keras over jpeg dataset (@ai-tor/@marks)
################################################################################
def val_data_generator(batchsize, image_list, get_speed = True, img_transpose = True, resize = True,
                   crop = False, min_speed = 4, min_angle = 0.1, straight_road_prob = 0.2, label = 'Train'):


    print "data_generator(): label {}, batchsize {}, image_list.shape {}".format(label, batchsize, image_list.shape)
    if resize == False:
        width, height = 640, 480
    # Else go with the values set in utils
    else:
        width, height = 200, 66

    # while 1:
    if img_transpose == True:
        x = np.zeros((batchsize, ch, width, height), dtype=np.uint8)
    else:
        x = np.zeros((batchsize, height, width, ch), dtype=np.uint8)

    y = np.zeros(batchsize)     # Steering
    z = np.zeros(batchsize)     # Speed

    i = 0
    while True:
        for idx in range(len(image_list)):

            steering = image_list.at[idx, 'angle']
            speed = image_list.at[idx, 'speed']
            imagepath = os.path.join(image_list.at[idx, 'imagepath'], image_list.at[idx, 'filename'])
            image = cv2.imread(imagepath)
            #Crop before resize!!
            if crop == True:
                # im[y1:y2, x1:x2]
                image = image[200:480, 0:640]
            img = cv2.resize(image, (width, height))

            r = np.random.uniform(0, 1)
            if ((abs(steering) > min_angle) and speed >= min_speed) or (r < straight_road_prob):
                if img_transpose == True:
                    x[i, :, :, :] = img.transpose(2,1,0)   # Transpose the image to fit into the CNN later...
                else:
                    x[i, :, :, :] = img

                #x[i, :, :, :] = img.transpose(2,1,0)   # Transpose the image to fit into the CNN later...
                y[i] = float(steering)
                z[i] = float(speed)
                #print ("{} {}: Steering {} / File {}".format(label, idx, y[i], imagepath))
                i = i + 1

            if (i == batchsize):
                i = 0
                # print "x: {} / y: {}".format(x, y)
                if get_speed == True:
                    yield (x, y, z)
                else:
                    yield (x, y)

# Train model--------------------
model, LossHistory = nnmodel.getNNModel()
history = LossHistory()
optimizer = Adam(lr=1e-4)
model.compile(optimizer, loss="mse")
plot(model, to_file='model.png')
stopping_callback = EarlyStopping(patience=5)

train_data_path = utils.train_data_path  #'../data/Challenge 2/train'
test_data_path = utils.test_data_path   #'../data/Challenge 2/test'
ch, width, height = utils.ch, utils.width, utils.height

print "Preparing training and validation data..."
train_paths = utils.get_data_paths(train_data_path)
train_path_list, valid_path_list = utils.split_train_and_validate(train_paths, 0.8)    # Use 80% for training, 20% for validation

# Get list of training images
train_images_df = utils.get_image_df(train_path_list).head(1024)
num_train_images = train_images_df.shape[0]
print "Found {} training images.".format(num_train_images)

# Get list of validation images
valid_images_df = utils.get_image_df(valid_path_list).head(1024)
num_valid_images = valid_images_df.shape[0]
print "Found {} validation images.".format(num_valid_images)

# Save validation images for use by the viewer later
valid_images_df.to_csv('../data/Challenge 2/validate_list.csv')     # TODO: Move filename to somewhere else and parameterise

# Now set up generators for training
#train_generator = utils.data_generator(128, train_images_df, get_speed=False) #, crop=True)
#val_data = utils.data_generator(32, valid_images_df, get_speed=False)  #, crop=True)
train_generator = data_generator(128, train_images_df, get_speed=False, crop=True, straight_road_prob = 1)
val_data = val_data_generator(128, valid_images_df, get_speed=False, crop=True, label='Val', straight_road_prob = 1)

hist = model.fit_generator(
    train_generator,
    samples_per_epoch=num_train_images,
    nb_epoch=5,
    validation_data=val_data,
    nb_val_samples=num_valid_images,
    callbacks=[history])

# Save time/date stamped trained model
model_date = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
model.save("../model/trained_model_{}.h5".format(model_date))

# TODO - Rework this graphing...
graph_training_loss_history('../model/loss_history_{}.png'.format(model_date), history.losses)

