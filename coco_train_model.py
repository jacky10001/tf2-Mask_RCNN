# -*- coding: utf-8 -*-
""" Usage: Set UserArg parameter:

# Train a new model starting from pre-trained COCO weights
    command  = 'train'
    dataset  = '/path/to/coco/'
    model    = 'coco'

# Train a new model starting from ImageNet weights. Also auto download COCO dataset
    command  = train
    dataset  = '/path/to/coco/'
    model    = 'imagenet'
    download = True

# Continue training a model that you had trained earlier
    command  = train
    dataset  = '/path/to/coco/'
    model    = '/path/to/weights.h5'

# Continue training the last model you trained
    command  = train
    dataset  = '/path/to/coco/'
    model    = 'last'

# Run COCO evaluatoin on the last model you trained
    command  = evaluate
    dataset  = '/path/to/coco/'
    model    = 'last'

@author: Jacky Gao
@date: Wed Dec  9 16:40:06 2020
"""

import os

import mrcnn.model as modellib
import imgaug  # pip install imgaug==0.2.6

# Import sample module
from mrcnn.samples import CocoConfig
from mrcnn.samples import CocoDataset
from mrcnn.samples import evaluate_coco


#%%
############################################################
#  Configurations
############################################################

class UserArg:
    # 'train' or 'evaluate' on MS COCO
    command = 'evaluate'
    
    # Path to weights
    model = 'coco'  # '*.h5' file or 'coco' and 'last'
    
    # Directory of the MS-COCO dataset
    dataset = r'D:\YJ\MyDatasets\COCO\coco2014'
    
    # '2014' or '2017' | Year of the MS-COCO dataset (default=2014)
    year = '2014'
    
    # Logs and checkpoints directory (default=logs/)
    logs = 'log_coco'
    
    # Images to use for evaluation (default=500)
    limit = 500
    
    # True or False | Automatically download and unzip MS-COCO files (default=False)
    download = True
    
    # pre-trained COCO weight
    COCO_MODEL_PATH =\
        os.path.join('pretrained_model','mask_rcnn_coco.h5')


#%% Parse arguments
args = UserArg()
print('Train Mask R-CNN on MS COCO.')
print("Command: ", args.command)
print("Model: ", args.model)
print("Dataset: ", args.dataset)
print("Year: ", args.year)
print("Logs: ", args.logs)
print("Auto Download: ", args.download)

os.makedirs(args.logs, exist_ok=True)


#%% Configurations
if args.command == "train":
    config = CocoConfig()
else:
    class InferenceConfig(CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0
    config = InferenceConfig()
config.display()


#%% Create model
if args.command == "train":
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=args.logs)
else:
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=args.logs)
print('Create Mask RCNN model for %s'%args.command)


#%% Load weights
# Select weights file to load
if args.model.lower() == "coco":
    model_path = args.COCO_MODEL_PATH
elif args.model.lower() == "last":
    # Find last trained weights
    model_path = model.find_last()
elif args.model.lower() == "imagenet":
    # Start from ImageNet trained weights
    model_path = model.get_imagenet_weights()
else:
    model_path = args.model
print("Loading weights ", model_path)
model.load_weights(model_path, by_name=True)


#%% Train or evaluate
if args.command == "train":
    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = CocoDataset()
    dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
    if args.year in '2014':
        dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
    dataset_train.prepare()
    
    # Validation dataset
    dataset_val = CocoDataset()
    val_type = "val" if args.year in '2017' else "minival"
    dataset_val.load_coco(args.dataset, val_type, year=args.year, auto_download=args.download)
    dataset_val.prepare()
    
    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)
    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all',
                augmentation=augmentation)

elif args.command == "evaluate":
    # Validation dataset
    dataset_val = CocoDataset()
    val_type = "val" if args.year in '2017' else "minival"
    coco = dataset_val.load_coco(args.dataset, val_type, year=args.year, return_coco=True, auto_download=args.download)
    dataset_val.prepare()
    print("Running COCO evaluation on {} images.".format(args.limit))
    evaluate_coco(model, dataset_val, coco, "bbox", limit=args.limit)

else:
    print("'{}' is not recognized. "
          "Use 'train' or 'evaluate'".format(args.command))