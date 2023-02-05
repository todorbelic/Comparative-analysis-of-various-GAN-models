import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization


class Generator(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.n_nodes = 4 * 4 * 1024
        self.latent_dim = latent_dim
        self.fc1_layer = Dense(self.n_nodes, input_dim=(latent_dim, ), activation='relu')
        self.reshape_layer = Reshape((4, 4, 1024))
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()
        self.batch_norm4 = BatchNormalization()
        self.cnv1_layer = Conv2DTranspose(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')
        self.cnv2_layer = Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')
        self.cnv3_layer = Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')
        self.cnv4_layer = Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), activation='tanh', padding='same')

    def call(self, input_tensor, training=False):
        x = self.fc1_layer(input_tensor, training=training)
        x = self.batch_norm1(x)
        x = self.reshape_layer(x)
        x = self.cnv1_layer(x, training=training)
        x = self.batch_norm2(x)
        x = self.cnv2_layer(x, training=training)
        x = self.batch_norm3(x)
        x = self.cnv3_layer(x, training=training)
        x = self.batch_norm4(x)
        x = self.cnv4_layer(x, training=training)
        return x

    @staticmethod
    def loss(fake_img):
        return -tf.reduce_mean(fake_img)

    def summary(self):
        x = tf.keras.Input(shape=(self.latent_dim, ))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
