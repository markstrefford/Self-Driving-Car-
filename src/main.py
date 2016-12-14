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
train_images_df = utils.get_image_df(train_path_list)
num_train_images = train_images_df.shape[0]
print "Found {} training images.".format(num_train_images)

# Get list of validation images
valid_images_df = utils.get_image_df(valid_path_list)
num_valid_images = valid_images_df.shape[0]
print "Found {} validation images.".format(num_valid_images)

# Save validation images for use by the viewer later
valid_images_df.to_csv('../data/Challenge 2/validate_list.csv')     # TODO: Move filename to somewhere else and parameterise

# Now set up generators for training
train_generator = utils.data_generator(32, train_images_df, get_speed=False) #, crop=True)
val_data = utils.data_generator(32, valid_images_df, get_speed=False)  #, crop=True)

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

