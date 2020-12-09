# Mask RCNN on TensorFlow2 and Keras

I modify and test code on TensorFlow 2.1 <br>
Also I implementation of model on MS COCO dataset.

I modify code from [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).

### Requir package
I use Anaconda to create environment and code edit on Spyder
* tensorflow 2.1.0
* Keras 2.3.1
* others in [requirements.txt](https://github.com/jacky10001/Mask_RCNN-tf2/blob/main/requirements.txt)

### Computer setup
* OS: Windows 10 64bit
* GPU: NVIDIA GTX 1060
* CUDA: 10.1
* CuDNN: 7.6.5 for CUDA 10.1

### Changed log:
    
    1.  import keras
        import keras.backend as K
        import keras.layers as KL
        import keras.engine as KE
        import keras.models as KM
    ==> from tensorflow import keras
        from tensorflow.keras import backend as K
        from tensorflow.keras import layers as KL
        from tensorflow.keras import models as KM
            ** All KE replace to KL
    
    2.  anchors = KL.Lambda(lambda x: tf.Variable
    ==> anchors = KL.Lambda(lambda x: tf.constant
                            
    3.  s = K.int_shape(x)
        mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
    ==> mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    
    4.  tf.random_shuffle
    ==> tf.random.shuffle
    
    5.  tf.log 
    ==> tf.math.log
    
    6.  try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving
    ==> from tensorflow.python.keras.saving import hdf5_format
    
    7.  self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
    ==> # self.keras_model._losses = []
        # self.keras_model._per_input_losses = {}
        
    8.  for name in loss_names:
            layer = self.keras_model.get_layer(name)
            loss = tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.)
            self.keras_model.add_loss(loss)
        
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(lambda: tf.add_n(reg_losses))

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            loss = tf.reduce_mean(layer.output, keepdims=True) * self.config.LOSS_WEIGHTS.get(name, 1.)
            self.keras_model.add_metric(loss, name=name)
            
    this is not all

### Work
Update: 2020 Dec. 10
- [X] run 'mrcnn'  on TF2 (use Shapes data)
- [X] input COCO dataset for training
- [X] run inspect code from [matterport/Mask_RCNN/samples/coco](https://github.com/matterport/Mask_RCNN/tree/master/samples/coco)
- [ ] input own dataset
- [ ] change backbone
- [ ] remove mask part for implementation of Faster RCNN model
