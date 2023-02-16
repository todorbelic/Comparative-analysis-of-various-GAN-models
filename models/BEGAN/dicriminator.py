import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Convolution2D, Input, Flatten

from models.BEGAN.block import Block


class Discriminator(tf.keras.Model):
    def __init__(self, latent_dim, required_dimension):
        super(Discriminator, self).__init__()
        input_shape = [required_dimension, required_dimension, 3]
        self.inputs = Input(shape=input_shape)
        self.encoder_block1 = Block(16, 32, required_dimension)
        self.encoder_block2 = Block(32, 48, required_dimension // 2)
        self.encoder_block3 = Block(48, 48, required_dimension // 4)
        self.encoder_block4 = Block(48, 48, required_dimension // 8)
        self.flatten = Flatten()
        self.dense1 = Dense(8 * 8 * 3 * 16)
        self.dense2 = Dense(latent_dim)
        self.n_nodes = 8 * 8 * 16
        self.fc1_layer = Dense(self.n_nodes, input_dim=(latent_dim,))
        self.reshape_layer = Reshape((8, 8, 16))
        self.decoder_block1 = Block(16, 16, required_dimension // 8)
        self.decoder_block2 = Block(32, 16, required_dimension // 4)
        self.decoder_block3 = Block(32, 16, required_dimension // 2)
        self.decoder_block4 = Block(32, 16, required_dimension)
        self.cnv1_layer = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.elu)
        self.cnv2_layer = Convolution2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.elu)
        self.cnv3_layer = Convolution2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=tf.nn.elu)
        self.cnv4_layer = Convolution2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')

    def call(self, input_tensor, training=False):
        x = self.encoder_block1(input_tensor, training=training)
        x = self.encoder_block2(x, training=training)
        x = self.encoder_block3(x, training=training)
        x = self.encoder_block4(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.fc1_layer(x, training=training)
        x = self.reshape_layer(x)
        x = self.decoder_block1(x, training=training)
        x = self.decoder_block2(x, training=training)
        x = self.decoder_block3(x, training=training)
        x = self.decoder_block4(x, training=training)
        x = self.cnv1_layer(x, training=training)
        x = self.cnv2_layer(x, training=training)
        x = self.cnv3_layer(x, training=training)
        return self.cnv4_layer(x, training=training)