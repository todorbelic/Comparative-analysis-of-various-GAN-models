from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from models.BEGAN.resize import Resize


class Block(layers.Layer):
    def __init__(self, n_filters1, n_filters2, required_dimension, kernel=(3, 3), stride=(1, 1)):
        super(Block, self).__init__()
        self.cnv1_layer = Conv2D(n_filters1, kernel, stride, padding='same', activation=tf.nn.elu)
        self.cnv2_layer = Conv2D(n_filters2, kernel, stride, padding='same', activation=tf.nn.elu)
        self.upsample_layer1 = Resize([required_dimension, required_dimension])

    @tf.function
    def call(self, inputs, training=False):
        x = self.cnv1_layer(inputs, training=training)
        x = self.cnv2_layer(x, training=training)
        return self.upsample_layer1(x)

