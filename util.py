#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 19:56:47 2021

@author: ozgur
"""
import os
import sys

from google_drive_downloader import GoogleDriveDownloader as gdd
from scipy import stats
from glob import glob
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

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
import ml_metrics
from prettytable import PrettyTable

ROOT_DIR = os.path.abspath("ssd_keras")
sys.path.append(ROOT_DIR)  # To find local version of the library
from models.keras_ssd300 import ssd_300 # pylint: disable=import-error
from keras_loss_function.keras_ssd_loss import SSDLoss # pylint: disable=import-error

WEIGHT_PATH = 'tmp_model/VGG_VOC0712_SSD_300x300_iter_120000.h5'

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
# Set the image size.
IMG_HEIGHT = 300
IMG_WIDTH = 300
COLOR_CODES = itertools.cycle(('r','g','b','c','m','y','k'))
c = (255/255., 127/255., 14/255.)

def __download_model():
    gdd.download_file_from_google_drive(file_id='19FH-pBzYPCbPi7jxRJSB-Y9YIwAYGOE1',
                                    dest_path='tmp_model/VGG_VOC0712_SSD_300x300_iter_120000.h5',
                                    showsize=True,
                                    overwrite=False)

def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_evaluation_result(model, image_path, image_type, T=20,
                          plot_ground_truth=False,
                          mc_dropout=True):
    iou_threshold = 0.5
    total_hull = 0.0

    search_path = image_path +  '*.' + image_type
    files = glob(search_path)

    f_result_path = 'model_pred_res_no_mc.csv'
    f_out = open(f_result_path,'w')

    for i in tqdm(range(len(files))):
        fname = files[i]
        ground_truth_file = fname + '.txt'
        ground_truth = pd.read_csv(ground_truth_file,
                                       names=['class','x1','y1','x2','y2'],
                                       sep=' ')

        mc_locations, uncertainties = get_pred_uncertainty(fname=fname, model=model,
                                                           T=T, plot_ground_truth=plot_ground_truth,
                                                           mc_dropout=mc_dropout)

        max_iou_val_list = []
        max_mAP_val_list = []
        avg_hull = 0.0
        if mc_locations.shape[0]:
            num_of_detected = 0.0
            mc_locations_df = pd.DataFrame(data=mc_locations,
                                           columns=['x1','y1','x2','y2','center_x',
                                                    'center_y','label'])
            mc_locations_df['label'] = pd.to_numeric(mc_locations_df['label'], downcast='integer')
            cluster_labels = np.unique(mc_locations_df.label.values)
            total_hull = 0.0
            for c_label in cluster_labels:
                if c_label == -1.0:
                    continue
                tmp_df = mc_locations_df.query('label == ' + str(c_label))
                tmp_df.drop(['label'], axis=1, inplace=True)
                avg_locations = tmp_df.values.mean(axis=0)
                x1_avg,y1_avg,x2_avg,y2_avg = avg_locations[0:4]
                x1_avg = np.min([avg_locations[0],avg_locations[2]])
                x2_avg = np.max([avg_locations[0],avg_locations[2]])
                y1_avg = np.min([avg_locations[1],avg_locations[3]])
                y2_avg = np.max([avg_locations[1],avg_locations[3]])

                max_mAP = 0.0
                max_iou_val = 0.0
                for index, row in ground_truth.iterrows():
                    x1 = np.min([row['x1'],row['x2']])
                    x2 = np.max([row['x1'],row['x2']])
                    y1 = np.min([row['y1'],row['y2']])
                    y2 = np.max([row['y1'],row['y2']])

                    iou_val = get_iou([x1_avg,y1_avg,x2_avg,y2_avg],[x1,y1,x2,y2])

                    map_list_pred = []
                    map_list_gt = []
                    map_list_pred.append([x1_avg,y1_avg,x2_avg,y2_avg])
                    map_list_gt.append([x1,y1,x2,y2])
                    obj_map = ml_metrics.mapk(map_list_gt, map_list_pred)
                    max_mAP = np.max([max_mAP,obj_map])

                    max_iou_val = np.max([max_iou_val, iou_val])

                max_iou_val_list.append(max_iou_val)
                max_mAP_val_list.append(max_mAP)

                mc_locations_obj = tmp_df.values
                if mc_locations_obj.shape[0] > 3:
                    tmp_hull_area = 0.0
                    points = mc_locations_obj[:,0:2]
                    hull1 = ConvexHull(points)
                    tmp_hull_area += hull1.area

                    points = mc_locations_obj[:,2:4]
                    hull1 = ConvexHull(points)
                    tmp_hull_area += hull1.area

                    points = np.c_[mc_locations_obj[:,0],mc_locations_obj[:,3]]
                    hull1 = ConvexHull(points)
                    tmp_hull_area += hull1.area

                    points = np.c_[mc_locations_obj[:,2],mc_locations_obj[:,1]]
                    hull1 = ConvexHull(points)
                    tmp_hull_area += hull1.area

                    if tmp_hull_area <= 150000:
                        total_hull += tmp_hull_area
                        num_of_detected += 1.0

            avg_hull = total_hull / len(cluster_labels)
            avg_hull = total_hull / (num_of_detected + 1e-20)

            max_iou_val_list = np.array(max_iou_val_list)
            detected = (max_iou_val_list >= iou_threshold)
            num_of_detected = max_iou_val_list[detected]

            f_out.write(str(ground_truth.shape[0])
                        + '\t' + str(num_of_detected.shape[0]) + '\t'
                        + str(max_iou_val_list.shape[0]) + '\t'
                        + str(max_iou_val_list).replace('\n',' ')
                        + '\t' + str(avg_hull) + '\n')

    f_out.close()

    df_result = pd.read_csv(f_result_path,names=['gt_obj','true_detected',
                                                 'detected','iou_preds',
                                                 'uncertainty'], sep='\t' )
    total_objects = np.sum(df_result.gt_obj.values)
    total_true_detected = np.sum(df_result.true_detected.values)
    total_detected = np.sum(df_result.detected.values)

    uio_vals = [np.fromstring(v.replace('[','').replace(']',''), sep=' ').mean() for v in df_result.iou_preds.values]
    uio_vals = np.array(uio_vals)
    uio_vals[np.isnan(uio_vals)] = 0

    avg_iou = uio_vals.mean()
    avg_iou = uio_vals[(uio_vals >= iou_threshold)].mean()

    TP = total_true_detected
    precision = total_true_detected/total_detected
    recall = total_true_detected/total_objects
    f1 = stats.hmean([precision, recall] )
    uncertainty = np.sum(df_result.uncertainty.values) / total_detected

    return avg_iou, TP,precision,recall,f1, uncertainty,total_detected,total_objects

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
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(4.5, 2.5)
        plt.show()

    return mc_locations, avg_surface

def plot_area_unc_vs_area(experiment_file):
    
    col_names = ['uncertainty','iou']
    df = pd.read_csv(experiment_file, delimiter='\t',names=col_names)
    h = sns.jointplot(x='uncertainty',y='iou', data=df, kind='reg',
               scatter_kws={'alpha':0.75,'color':c},
               marginal_kws=dict(bins=30),)
    h.ax_joint.set_xlabel('Uncertainty')
    h.ax_joint.set_ylabel('IoU')
    plt.show()

if __name__ == "__main__":
    fname = 'images/Stanford/00002.jpg'
    image_path = 'images/Stanford/'
    image_type = 'jpg'
    tmp_model = get_model(0.3)
    '''
    get_pred_uncertainty(fname,tmp_model, T=10,
                         plot_ground_truth=True)
    avg_iou, TP,precision,recall,f1, uncertainty,total_detected,total_objects = get_evaluation_result(tmp_model, image_path,image_type)
    t = PrettyTable(['Metric', 'Value'])
    t.add_row(['Avg. IoU', np.round(avg_iou,4)])
    t.add_row(['TP', TP])
    t.add_row(['Precision',precision])
    t.add_row(['Recall',recall])
    t.add_row(['F1',f1])
    t.add_row(['Uncertainty',np.round(uncertainty,4)])
    t.add_row(['Total detected',total_detected])
    t.add_row(['Total objects',total_objects])
    print(t)
    '''
    
    experiment_file = 'experiments_results_for_rq2.csv'
    
    plot_area_unc_vs_area(experiment_file)
    
    f_out = open(experiment_file,'a')
    for _ in tqdm(range(10000), 'Overall experiments'):
        avg_iou, TP,precision,recall,f1, \
            uncertainty,total_detected,\
                total_objects = get_evaluation_result(tmp_model,
                                                      image_path,
                                                      image_type,
                                                      T=30)
        
        f_out.write(str(uncertainty) + '\t' + str(avg_iou) + '\n')
    f_out.close()
    plot_area_unc_vs_area(experiment_file)
    
    