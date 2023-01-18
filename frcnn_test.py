# https://github.com/eleow/tfKerasFRCNN

from FRCNN import plotAccAndLoss
import os

# path = os.path.dirname(os.path.abspath(__file__))

path = "D:/Users/kirar/Documents/[Lux]_ISM/[ISM]_3rd_Semester/[3rd_Semester]_Computer Vision and Image Analysis/stream_1/202212282340"

plotAccAndLoss(path + '/FRCNN_vgg.csv')

import math
parseAnnotation = False

# Parsing of data especially through Google Colab is slow, so we should save the results so that we do it once only
# ----------------------
baseModelName = "FRCNN"
base_net_type = 'vgg'   # either 'vgg' or 'resnet50'
modelName = baseModelName + "_" + base_net_type
model_path_name = modelName + ".hdf5"

model_path = path + "/" + model_path_name

im_size = 300                       # shorter-side length. Original is 600, half it to save training time
anchor_box_scales = [64,128,256]    # also half box_scales accordingly. Original is [128,256,512]
anchor_box_ratios = [[1,1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]] # anchor box ratios area == 1
num_rois = 256
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)


import pickle
if parseAnnotation:
  # Load image information
  from FRCNN import parseAnnotationFile
  annotation_train_path = './annotation_train_cvia_demo_alldata.txt'


  classes_of_interest = [ 'cheops',
                            'debris',
                            'double_star',
                            'earth_observation_sat_1',
                            'lisa_pathfinder',
                            'proba_2',
                            'proba_3_csc',
                            'proba_3_ocs',
                            'smart_1',
                            'soho',
                            'xmm_newton',
                            'bg']

  train_data, classes_count, class_mapping = parseAnnotationFile(annotation_train_path, mode='simple', filteredList=classes_of_interest)

  annotation_test_path = './annotation_train_cvia_demo_alldata.txt'
  test_data, _ , _ = parseAnnotationFile(annotation_test_path, mode='simple', filteredList=classes_of_interest)

  with open('./stream1_cvia.pickle', 'wb') as f2:
      pickle.dump((train_data, classes_count, class_mapping), f2)
  with open('./stream1_cvia.pickle', 'wb') as f2:
      pickle.dump(test_data, f2)

else:
  # Load from pickle
  with open('./stream1_cvia.pickle', 'rb') as f_in:
      train_data, classes_count, class_mapping = pickle.load(f_in)
  
  for i in range(len(train_data)):
    train_data[i]['filepath'] = train_data[i]['filepath'].replace('\\', '/')

  with open('./stream1_cvia.pickle', 'rb') as f_in:
      test_data = pickle.load(f_in)


# Create model and load trained weights (Note: class mapping and num_classes should be based on training set)
# ----------------------
from FRCNN import FRCNN
frcnn_test = FRCNN(input_shape=(None,None,3), num_anchors=num_anchors, num_rois=num_rois, base_net_type=base_net_type, num_classes = len(classes_count))
frcnn_test.load_config(anchor_box_scales=anchor_box_scales, anchor_box_ratios=anchor_box_ratios, num_rois=num_rois, target_size=im_size)
frcnn_test.load_weights(model_path)
frcnn_test.compile()

# Load array of images
# ----------------------
from FRCNN import convertDataToImg
# test_imgs = convertDataToImg(test_data)
# predicts = frcnn_test.predict(test_imgs, class_mapping=class_mapping, verbose=2, bbox_threshold=0.5, overlap_thres=0.2)

# If images are in a folder
# ----------------------
import cv2

# path_imgs = "D:/Users/kirar/Documents/[Lux]_ISM/[ISM]_3rd_Semester/[3rd_Semester]_Computer Vision and Image Analysis/stream_1/first_results/train"
# imgPaths = [os.path.join(path_imgs, s) for s in os.listdir(path_imgs)]
# test_imgs2 = []
# for path in imgPaths:
#     test_imgs2.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))
# predicts = frcnn_test.predict(test_imgs2, class_mapping=class_mapping, verbose=2, bbox_threshold=0.5, overlap_thres=0.2)

# Modified
# ----------------------
# import glob
# path_imgs = glob.glob("D:/Users/kirar/Documents/[Lux]_ISM/[ISM]_3rd_Semester/[3rd_Semester]_Computer Vision and Image Analysis/stream_1/first_results/train/*.png")

# images = []
# for img in path_imgs:
#     images.append(cv2.imread(img))
# predicts = frcnn_test.predict(images, class_mapping=class_mapping, verbose=2, bbox_threshold=0.5, overlap_thres=0.2)


# One image
# ----------------------
# img_path = "D:/Users/kirar/Documents/[Lux]_ISM/[ISM]_3rd_Semester/[3rd_Semester]_Computer Vision and Image Analysis/stream_1/train/img000013.png"

# img = cv2.imread(img_path)
# # print(img.shape)
# # cv2_imshow(img)
# predicts = frcnn_test.predict([img], class_mapping=class_mapping, verbose=2, bbox_threshold=0.5, overlap_thres=0.2)

# Evaluate Model
# ----------------------
import numpy as np
evaluate = frcnn_test.evaluate(test_data, class_mapping=class_mapping, verbose=2)


