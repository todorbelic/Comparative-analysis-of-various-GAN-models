from tensorflow.keras import layers
import tensorflow as tf

class Resize(layers.Layer):
    def __init__(self, img_dim=(64, 64), **kwargs):
        super(Resize, self).__init__(kwargs)
        self.img_dim = img_dim

    def call(self, inputs, *args, **kwargs):
        return tf.image.resize(inputs, self.img_dim, method='nearest')

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.img_dim[0], self.img_dim[1], input_shape[-1]