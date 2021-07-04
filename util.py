#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:56:47 2021

@author: ozgur
"""
import os,sys,glob,ntpath
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
ROOT_DIR = os.path.abspath("ssd_keras")
sys.path.append(ROOT_DIR)  # To find local version of the library

WEIGHT_PATH = 'tmp_model/VGG_VOC0712_SSD_300x300_iter_120000.h5'


def __download_model():
    gdd.download_file_from_google_drive(file_id='19FH-pBzYPCbPi7jxRJSB-Y9YIwAYGOE1',
                                    dest_path='tmp_model/VGG_VOC0712_SSD_300x300_iter_120000.h5',
                                    showsize=True,
                                    overwrite=False)

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
           'sheep', 'sofa', 'train', 'tvmonitor']
# Set the image size.
img_height = 300
img_width = 300

from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

import os,sys,glob,ntpath
from keras import backend as K
import pandas as pd
from random import shuffle

from tqdm import tqdm
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

def get_model(p_size,mc_dropout=True):
    K.clear_session() # Clear previous models from memory.

    model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
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

if __name__ == "__main__":
    fname = __download_model()
    print(fname)
    tmp_model = get_model(0.2)
