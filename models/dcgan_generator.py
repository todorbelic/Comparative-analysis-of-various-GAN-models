import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, ReLU, LeakyReLU, Dropout


class Generator(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.n_nodes = 8 * 8 * 128
        self.latent_dim = latent_dim
        self.fc1_layer = Dense(self.n_nodes, input_dim=latent_dim)
        self.reshape_layer = Reshape((8, 8, 128))
        self.cnv1_layer = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                          activation='relu')
        self.cnv2_layer = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                          activation='relu')
        self.cnv3_layer = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same',
                                          activation='relu')
        self.cnv4_layer = Conv2D(filters=3, kernel_size=(5, 5), activation='sigmoid', padding='same')

    def call(self, input_tensor, training=False):
        x = self.fc1_layer(input_tensor, training=training)
        x = self.reshape_layer(x)
        x = self.cnv1_layer(x, training=training)
        x = self.cnv2_layer(x, training=training)
        x = self.cnv3_layer(x, training=training)
        x = self.cnv4_layer(x, training=training)

        return x

    def summary(self):
        x = tf.keras.Input(shape=(self.n_nodes, self.latent_dim))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
