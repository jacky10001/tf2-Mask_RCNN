
### Changed log in mrcnn/model.py


* change import function
```
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
```
```
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL    # All KE replace to KL
from tensorflow.keras import models as KM
```


* Can't use None
```
s = K.int_shape(x)
mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
```
```
mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
```


* modify to TF2 function
```
tf.random_shuffle
```
```
tf.random.shuffle
```


* modify to TF2 function
```
tf.log 
```
```
tf.math.log
```


* If use 'tf.keras' package need change function, because 'tf.keras' not 'engine' and 'topology'.  
And 'tf.keras' saving module have 'hdf5_format' some function move to 'hdf5_format'  
* if use 'keras', don't change everything I use this, and 'keras' version is 2.3.1
```
try:
    from keras.engine import saving
        except ImportError:
    # Keras before 2.2 used the 'topology' namespace.
    from keras.engine import topology as saving
```
```
from tensorflow.python.keras.saving import hdf5_format
```


* change losses and metricsto TF2 format in  
MaskRCNN class and build method  
```
self.keras_model._losses = []
self.keras_model._per_input_losses = {}
  ** from add loss to complie **
```
```
# self.keras_model._losses = []
# self.keras_model._per_input_losses = {}

for name in loss_names:
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

    self.keras_model.compile(
        optimizer=optimizer,
        loss=[None] * len(self.keras_model.outputs))
```


* Define number in tf.range(), not use  None
```
indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
```
```
indices = tf.stack([tf.range(1000), class_ids], axis=1)
```


* Define AnchorsLayer to replace KL.Lambda
```
anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
```
```
class AnchorsLayer(KL.Layer):
    def __init__(self, **kwargs):
        super(AnchorsLayer, self).__init__(**kwargs)
    
    def call(self, anchor):
        return anchor[0]
    
    def get_config(self):
        config = super(AnchorsLayer, self).get_config()
        return config
        
anchors = AnchorsLayer(name="anchors")([anchors,input_image])
```


* use tf.cast to change data type
```
tf.to_float(tf.gather(class_ids, keep))
```
```
tf.cast(tf.gather(class_ids, keep), 'float32')
```


* s[1] value is None, direct use -1
```
s = K.int_shape(x)
mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
```
```
# s = K.int_shape(x)
mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
```


* Keras Model haven't uses_learning_phase  
Because I use package of Keras is higher than original repository 
```
inputs = model.inputs
if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    inputs += [K.learning_phase()]
```
```
# inputs = model.inputs
# if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
#   inputs += [K.learning_phase()]
```
```
if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    model_in.append(0.)
```
```
# if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
    #   model_in.append(0.)
```



### Changed log in mrcnn/utils.py

* modify to TF2 function
```
dh = tf.log(gt_height / height)
dw = tf.log(gt_width / width)
```
```
dh = tf.math.log(gt_height / height)
dw = tf.math.log(gt_width / width)
```
