# -*- coding: utf-8 -*-
"""
Mask R-CNN - Inspect Weights of a Trained Model

This code includes code and visualizations to test,
debug, and evaluate the Mask R-CNN model.

@author: Jacky Gao
@date: Thu Dec 10 02:15:47 2020
"""

import os

import tensorflow as tf
import matplotlib.pyplot as plt

# Import Mask RCNN
import mrcnn.model as modellib
from mrcnn import visualize


# Import sample module
from mrcnn.samples import CocoConfig

# Directory to save logs and trained model
MODEL_DIR = 'log_coco'

# Local path to trained weights file
COCO_MODEL_PATH =\
    os.path.join('pretrained_model','mask_rcnn_coco.h5')

# Configurations
config = CocoConfig()


#%% Notebook Preferences
# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0


#%% 
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


#%% Load Model
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set weights file path
weights_path = COCO_MODEL_PATH

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


#%% Review Weight Stats
# Show stats of all trainable weights
# visualize.display_weight_stats(model, html=True)
visualize.display_weight_stats(model, html=False)


#%% Histograms of Weights
# Pick layer types to display
LAYER_TYPES = ['Conv2D', 'Dense', 'Conv2DTranspose']

# Get layers
layers = model.get_trainable_layers()
layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES, 
                layers))

# Display Histograms
fig, ax = plt.subplots(len(layers), 2, figsize=(10, 3*len(layers)),
                       gridspec_kw={"hspace":1})
for l, layer in enumerate(layers):
    weights = layer.get_weights()
    for w, weight in enumerate(weights):
        tensor = layer.weights[w]
        ax[l, w].set_title(tensor.name)
        _ = ax[l, w].hist(weight[w].flatten(), 50)
