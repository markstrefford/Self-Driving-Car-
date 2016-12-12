#!/usr/bin/env python
import argparse
#import sys
import numpy as np
#import h5py
import pygame
#import json
#from keras.models import model_from_json
import utils as u
import cv2
import os

train_data_path = u.train_data_path  #'../data/Challenge 2/Train'
ch, width, height = u.ch, u.width, u.height

# Setup viewer
frame_size = (480,360)
x_scaling, y_scaling = 1.5, 1.5     # For the steering trace


# Based on code from comma.ai viewer
# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))   # *2 required due to viewer size (640x480)

def perspective_tform(x, y):
  p1, p2 = tform3_img((x,y))[0]
  return p2 * y_scaling, p1 * x_scaling

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
  row, col = perspective_tform(x, y)
  if row >= 0 and row < img.shape[0] and\
     col >= 0 and col < img.shape[1]:
    img[row-sz:row+sz, col-sz:col+sz] = color

def draw_path(img, path_x, path_y, color):
  for x, y in zip(path_x, path_y):
    draw_pt(img, x, y, color)

def calc_curvature(v_ego, angle_steers, angle_offset=0):
  deg_to_rad = np.pi/180.
  slip_fator = 0.0014 # slip factor obtained from real data
  steer_ratio = 14.8  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
  wheel_base = 2.85   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

  angle_steers_rad = (angle_steers - angle_offset) #* deg_to_rad (Udacity data already in rads)
  curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
  return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
  #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
  curvature = calc_curvature(v_ego, angle_steers, angle_offset)

  # clip is to avoid arcsin NaNs due to too sharp turns
  y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
  return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255)):
    path_x = np.arange(0., 20.1, 0.25)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)

##########################################################
#
# Merge left, right and centre images into a single image
#
# Based on http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
# and http://stackoverflow.com/questions/35588570/cv2-featuredetector-createsift-causes-segmentation-fault
#
# SIFT and SURF are patented, so using open source BRISK instead (not FREAK was preferred but not available in OpenCV2.14 build!?)
#
##########################################################



def detect_and_describe(image):
    img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    briskDetector = cv2.BRISK()
    (kps, features) = briskDetector.detectAndCompute(img1.png, None)
    kps=np.float32([kp.pt for kp in kps])
    return (kps, features)


def merge_camera_images(images, ratio=0.75, reprojThresh=4.0, show_matches = True):
    kps1, features1=detect_and_describe(images[0])
    kps2, features2=detect_and_describe(images[1])
    kps3, features3=detect_and_describe(images[2])



    # bf = cv2.BFMatcher()
    # matches = bf.knnmatch(des1, des2, k=2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    print "merge_camera_images: img3.shape = {}".format(img3.shape)
    return img3

##########################################################


# TODO - Predict based on the image, the speed, or just get something from an array
# TODO - May have multiple versions of this function in the end...
def predict_steering_angle(i, img, speed):
    return 0    # Default to straight ahead for now


# Main Loop
i=0
#path = utils.get_datafile()
#data = test_load_udacity_dataset(path)

print "Preparing image data for viewer..."
train_paths = u.get_data_paths(train_data_path)
#print train_paths
img_df = u.get_image_list(train_paths)
#images_df = img_df.loc[img_df['frame_id']=='center_camera'].reset_index(drop=True) # , inplace=True)   # Get centre camera images only
images_df = img_df      # TODO: Remove later
num_images = images_df.shape[0]/3       # Assume 3 images per time interval
print "Found {} images.".format(num_images)

#for img, steering, speed in u.udacity_data_generator(1, images_df, range(len(images_df))):   # (128, images_df, train_image_idx, 't')
# for img, steering, speed in u.data_generator(1, images_df, get_speed = True, img_transpose=False,
#                                                      resize = False, min_speed = 0, min_angle = 0):   # (128, images_df, train_image_idx, 't')

# Need 3 images (left, centre, right)
# So iterate through images_df, take 3 images
frame=np.zeros((frame_size[1],frame_size[0]*3,3), dtype=np.uint8)
left_cam = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
right_cam = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
center_cam = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

for index, image_data in images_df.iterrows():
    time_index = image_data['index']
    steering = image_data['angle']
    speed = image_data['speed']
    imagepath = os.path.join(image_data.at['imagepath'], image_data['filename'])
    img = cv2.imread(imagepath)
    #img -= int(np.mean(img))   # From http://cs231n.github.io/neural-networks-2/#init
    image = cv2.resize(img, frame_size)  # Resize as we have 3 images next to each other!

    frame_id = image_data['frame_id']
    if frame_id == 'left_camera':
        left_cam = image;
        offset = 0
    elif frame_id == 'center_camera':
        center_cam = image
        offset = frame_size[0]
        # Only predict for a center camera image
        predicted_steering = predict_steering_angle(i, img, speed)
        draw_path_on(image, speed, steering)
        draw_path_on(image, speed, predicted_steering, (0, 255, 0))
    elif frame_id == 'right_camera':
        right_cam = image;
        offset = frame_size[0] * 2
    else:
        print "ERROR: Frame_id {} is unsupported!".format(frame_id)
        offset = -1  # Should never get here!
        break

    # Place image in the larger frame now!
    # TODO - Use np.hstack!!
    for row in range(frame_size[1]):
        #print "Row: {}, Offset: {}, Offset+frame_size[0]: {}, image[row, :, :].shape: {}".format(row, offset, offset+frame_size[0], image[row, :, :].shape)
        frame[row, offset:offset+frame_size[0], :] = image[row, :, :]

    # Merge images
    merged_image = merge_camera_images([left_cam, center_cam, right_cam])

    # Display image
    #cv2.imshow('Udacity challenge 2 - viewer', frame)
    cv2.imshow('Udacity challenge 2 - viewer', merged_image)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

    print "{}: Steering angle: {} / Speed: {}".format(i,steering, speed)
    i += 1

cv2.destroyAllWindows()




