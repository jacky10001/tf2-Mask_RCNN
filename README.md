# Mask RCNN on TensorFlow2 and Keras

I modify and test code on TensorFlow 2.1 <br>
Also I implementation of model on MS COCO dataset.

I modify code from [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).

### Requir package
I use Anaconda to create environment and code edit on Spyder
* tensorflow 2.1.0
* Keras 2.3.1
* others in [requirements.txt](https://github.com/jacky10001/Mask_RCNN-tf2/blob/main/requirements.txt)

### Work setup
* OS: Windows 10 64bit
* GPU: NVIDIA GTX 1060
* CUDA: 10.1
* CuDNN: 7.6.5 for CUDA 10.1

### Work
Update: 2020 Dec. 10
- [X] run 'mrcnn'  on TF2 (use Shapes data)
- [X] input COCO dataset for training
- [X] run inspect code from [matterport/Mask_RCNN/samples/coco](https://github.com/matterport/Mask_RCNN/tree/master/samples/coco).
- [ ] input own dataset
- [ ] change backbone
- [ ] remove mask part for implementation of Faster RCNN model
