import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Dropout


class Discriminator(tf.keras.Model):
    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape
        self.leaky_relu1 = LeakyReLU(alpha=0.2)
        self.leaky_relu2 = LeakyReLU(alpha=0.2)
        self.leaky_relu3 = LeakyReLU(alpha=0.2)
        self.cnv1_layer = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape)
        self.cnv2_layer = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape)
        self.cnv3_layer = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=in_shape)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, input_tensor, training=False):
        x = self.cnv1_layer(input_tensor, training=training)
        x = self.leaky_relu1(x)
        x = self.cnv2_layer(x, training=training)
        x = self.leaky_relu2(x)
        x = self.cnv3_layer(x, training=training)
        x = self.leaky_relu3(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        return self.output_layer(x)

    def summary(self):
        x = tf.keras.Input(shape=self.in_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
