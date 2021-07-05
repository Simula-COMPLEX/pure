#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:56:47 2021

@author: ozgur
"""
import os
import sys
from google_drive_downloader import GoogleDriveDownloader as gdd

from keras.optimizers import Adam
from keras import backend as K
import itertools
import pandas as pd
from imageio import imread
from keras.preprocessing import image
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300 # pylint: disable=import-error
from keras_loss_function.keras_ssd_loss import SSDLoss # pylint: disable=import-error

ROOT_DIR = os.path.abspath("ssd_keras")
sys.path.append(ROOT_DIR)  # To find local version of the library

WEIGHT_PATH = 'tmp_model/VGG_VOC0712_SSD_300x300_iter_120000.h5'

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
# Set the image size.
IMG_HEIGHT = 300
IMG_WIDTH = 300
COLOR_CODES = itertools.cycle(('r','g','b','c','m','y','k'))

def __download_model():
    gdd.download_file_from_google_drive(file_id='19FH-pBzYPCbPi7jxRJSB-Y9YIwAYGOE1',
                                    dest_path='tmp_model/VGG_VOC0712_SSD_300x300_iter_120000.h5',
                                    showsize=True,
                                    overwrite=False)


def get_model(p_size,mc_dropout=True):
    """ MC dropouts comaptible SSD300 model load method """
    __download_model()
    K.clear_session() # Clear previous models from memory.

    model = ssd_300(image_size=(IMG_HEIGHT, IMG_WIDTH, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400,
                    mc_dropout=mc_dropout,
                    dropout_size=p_size)

    # 2: Load the trained weights into the model.
    model.load_weights(WEIGHT_PATH, by_name=True)
    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    return model

def get_pred_uncertainty(fname, model, T=20, 
                         plot_ground_truth=False, mc_dropout=True):
    ground_truth_file = fname + '.txt'
    ground_truth = pd.read_csv(ground_truth_file,
                               names=['class','x1','y1','x2','y2'],
                               sep=' ')
    
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    orig_images.append(imread(fname))
    img = image.load_img(fname, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = image.img_to_array(img) 
    input_images.append(img)
    input_images = np.array(input_images)
    
    mc_locations = []

    for _ in range(T):
        y_pred = model.predict(input_images)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        
        for box in y_pred_thresh[0]:
            x1 = box[2] * orig_images[0].shape[1] / IMG_WIDTH
            y1 = box[3] * orig_images[0].shape[0] / IMG_HEIGHT
            x2 = box[4] * orig_images[0].shape[1] / IMG_WIDTH
            y2 = box[5] * orig_images[0].shape[0] / IMG_HEIGHT
            width, height = x2 - x1, y2 - y1
            mc_locations.append(np.array([x1,y1,x2,y2,x1+width/2,y1+height/2]))
            
    mc_locations = np.array(mc_locations)
    avg_surface = -1.0
    if mc_locations.shape[0]:
        clustering = DBSCAN(eps=100, min_samples=2).fit(mc_locations)
        mc_locations = np.c_[mc_locations,clustering.labels_.ravel()]
        
        mc_locations_df = pd.DataFrame(data=mc_locations, columns=['x1','y1','x2','y2','center_x','center_y','label'])
        cluster_labels = np.unique(mc_locations[:,6])
        total_cluster_surface = 0.0
        if mc_dropout == True:
            for cluster_label in cluster_labels:
                cluster_df = mc_locations_df.query('label == ' + str(cluster_label))
                if cluster_df.shape[0] > 2:
                    center_data = cluster_df[['x1','y1']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x2','y1']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x1','y2']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x2','y2']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                avg_surface = total_cluster_surface/mc_locations.shape[0]
    
    if plot_ground_truth:
        data = plt.imread(fname)
        fig, ax = plt.subplots(1,1,sharey=True, figsize=(12,5))
        ax.imshow(data)
        for i in range(mc_locations.shape[0]):
            x1, y1, x2, y2 = mc_locations[i,0:4]
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
            ax.scatter(mc_locations[i,4],mc_locations[i,5], marker='x',
                       c='g',s=150)
            
        for index, row in ground_truth.iterrows():
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'],row['y2']
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, 
                             color='white',lw=3,ls='--')
            ax.add_patch(rect)
        plt.show()
    
    return mc_locations, avg_surface

if __name__ == "__main__":
    tmp_model = get_model(0.05)
    get_pred_uncertainty('images/Stanford/00002.jpg',tmp_model, T=10, 
                         plot_ground_truth=True)
