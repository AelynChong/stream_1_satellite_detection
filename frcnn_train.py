
# Configuration
# ----------------------
import math
import json
import os

baseModelName = "FRCNN"
base_net_type = 'vgg'   # either 'vgg' or 'resnet50'
modelName = baseModelName + "_" + base_net_type
model_path = modelName + ".hdf5"
csv_path = modelName + ".csv"

num_epochs = 66
steps = 1000

im_size = 300                       # shorter-side length. Original is 600, half it to save training time
anchor_box_scales = [64,128,256]    # also half box_scales accordingly. Original is [128,256,512]
anchor_box_ratios = [[1,1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]] # anchor box ratios area == 1
num_rois = 256


# Load Data
# ----------------------
parseAnnotation = False

# Parsing of data especially through Google Colab is slow, so we should save the results so that we do it once only
import pickle
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))
annotation_train_path = path + './annotation_train_cvia_demo_alldata.txt'


if parseAnnotation:
    from FRCNN import parseAnnotationFile
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

    with open(path + '/stream1_cvia.pickle', 'wb') as f2:
        pickle.dump((train_data, classes_count, class_mapping), f2)
  
    for i in range(len(train_data)):
        train_data[i]['filepath'] = path + "/" + train_data[i]['filepath']

else:
    # Load from pickle
    with open(path + '/stream1_cvia.pickle', 'rb') as f_in:
        train_data, classes_count, class_mapping = pickle.load(f_in)
        # train_data, classes_count, class_mapping = json.dump(obj, f_in, indent=2)
  
    for i in range(len(train_data)):
        train_data[i]['filepath'] = train_data[i]['filepath'].replace('\\', '/')
        train_data[i]['filepath'] = path + "/" + train_data[i]['filepath']


# Inspect annotation file with a sample image
# ----------------------
from FRCNN import viewAnnotatedImage
# viewAnnotatedImage('./annotation_train_cvia_demo.txt', 'train/img110000.png')
# viewAnnotatedImage('./annotation_train_cvia_demo.txt', 'train/img000016.png')


# Create and Train FRCNN model
# ----------------------
from FRCNN import FRCNN
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
frcnn = FRCNN(input_shape=(None,None,3), num_anchors=num_anchors, num_rois=num_rois, base_net_type=base_net_type, num_classes = len(classes_count))
frcnn.compile()


# Visualise
# ----------------------
#frcnn.model_rpn.summary()
#frcnn.summary()
# Plot structure of FRCNN
from tensorflow.keras.utils import plot_model
plot_model(frcnn.model_all, to_file=modelName+'.png', show_shapes=True, show_layer_names=False, rankdir='TB')


# Train
# ----------------------
## create iterator
from FRCNN import FRCNNGenerator, inspect, preprocess_input
train_it = FRCNNGenerator(train_data,
    target_size=im_size,
    horizontal_flip=True, vertical_flip=False, rotation_range=5, 
    width_shift_range=0.2,
    shuffle=True, base_net_type=base_net_type,
    preprocessing_function=preprocess_input
)

# inspect(train_it, im_size)

frcnn.fit_generator(train_it, target_size = im_size, class_mapping = class_mapping, epochs=num_epochs, steps_per_epoch=steps,
    model_path=model_path, csv_path=csv_path, initial_epoch=-1)
