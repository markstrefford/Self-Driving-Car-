#!/usr/bin/env python

# This is a viewer for an existing CSV file
# Note that it does not do any predictions, but if the CSV file contains a column
# called 'predicted_steering' then this will also be rendered to show the difference
# between the actual and predicted steering angles
#
# (c) 2017 Timelaps AI Limited
#
# Written by Mark Strefford 12/1/2017

import argparse
import cv2
import scipy.misc
import csv

# Setup viewer
frame_size = (640,480)
deg_to_rad = scipy.pi / 180

def run_viewer(input_file, image_directory, delim, steering_units, show_predicted):
    with open(image_directory + '/' + input_file) as f:
            reader = csv.DictReader(f, delimiter=delim)
            for row in reader:
                print row

                filename = row['filename']  # + '.jpg'
                steering_angle = float(row['steering_angle'])
                print 'steering angle {} {}'.format(steering_angle, steering_units)
                if steering_units == 'deg':
                    steering_angle = -steering_angle * deg_to_rad

                image = scipy.misc.imread(args.data_dir + "/" + filename, mode="RGB")
                full_image = cv2.resize(image, frame_size)
                img = cv2.imread('wheel.png', -1)
                # img = scipy.misc.imresize(img, 0.2)
                height, width, _ = img.shape
                # TODO - Radians vs degrees??
                M = cv2.getRotationMatrix2D((width / 2, height / 2), steering_angle * 360.0 / scipy.pi, 1)
                dst = cv2.warpAffine(img, M, (width, height))

                x_offset = (full_image.shape[1] - width) / 2
                y_offset = 300
                new_height = min(height, full_image.shape[0] - y_offset)
                for c in range(0, 3):
                    alpha = dst[0:new_height, :, 3] / 255.0
                    color = dst[0:new_height, :, c] * (alpha)
                    beta = full_image[y_offset:y_offset + new_height, x_offset:x_offset + width, c] * (1.0 - alpha)
                    full_image[y_offset:y_offset + new_height, x_offset:x_offset + width, c] = color + beta

                cv2.imshow("Output", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run LSTM on test data and write to file')
    parser.add_argument('--input', '-i', action='store', dest='input_file',
                        default='data.csv', help='Input model csv file name')
    parser.add_argument('--data-dir', '--data', action='store', dest='data_dir',
                        default='/vol/test/')
    parser.add_argument('--show_predictions', action='store', dest='show_predicted',
                        default=False, help='Show predicted steering angles')
    parser.add_argument('--delimiter', action='store', dest='delimiter',
                        default=',', help='Delimeter')
    parser.add_argument('--units', action='store', dest='steering_units',
                        default='rad', help='Units: rad or deg')
    args = parser.parse_args()

    run_viewer(args.input_file, args.data_dir, args.delimiter, args.steering_units, args.show_predicted)


