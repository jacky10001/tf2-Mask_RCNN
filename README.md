# Mask RCNN on TensorFlow2 and Keras

I refer [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN), and modify code to TensorFlow 2.1 version.<br>
Also I implementation of model on MS COCO dataset.

### Requir package
I use Anaconda to create environment and code edit on Spyder
* tensorflow 2.1.0 (2.3.0 also can run, but will very many warning)
* Keras 2.3.1
* others in [requirements.txt](https://github.com/jacky10001/Mask_RCNN-tf2/blob/main/requirements.txt)

### Computer setup
* OS: Windows 10 64bit
* GPU: NVIDIA GTX 1060
* CUDA: 10.1
* CuDNN: 7.6.5 for CUDA 10.1

### Work
Update: 2020 Dec. 12  
Only input and evaluate COCO data by pre-trained model.  
For COCO training work not yet.  
- [X] run 'mrcnn' on TF2 (use Shapes data)
- [X] input COCO dataset for training
- [X] run inspect code from [matterport/Mask_RCNN/samples/coco](https://github.com/matterport/Mask_RCNN/tree/master/samples/coco)
- [X] check training happen on COCO dataset
- [X] remove mask part for implementation of Faster RCNN model ([Here](https://github.com/jacky10001/Faster_RCNN-tf2))

