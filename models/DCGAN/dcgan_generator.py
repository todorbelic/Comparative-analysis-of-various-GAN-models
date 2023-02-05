import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization
cross_entropy = tf.keras.losses.BinaryCrossentropy()


class Generator(tf.keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.n_nodes = 8 * 8 * 128
        self.latent_dim = latent_dim
        self.fc1_layer = Dense(self.n_nodes, input_dim=latent_dim)
        self.reshape_layer = Reshape((8, 8, 128))
        self.leaky_relu1 = LeakyReLU(alpha=0.2)
        self.leaky_relu2 = LeakyReLU(alpha=0.2)
        self.leaky_relu3 = LeakyReLU(alpha=0.2)
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()
        self.cnv1_layer = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.cnv2_layer = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.cnv3_layer = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding='same')
        self.cnv4_layer = Conv2D(filters=3, kernel_size=(5, 5), activation='sigmoid', padding='same')

    def call(self, input_tensor, training=False):
        x = self.fc1_layer(input_tensor, training=training)
        x = self.reshape_layer(x)
        x = self.cnv1_layer(x, training=training)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)
        x = self.cnv2_layer(x, training=training)
        x = self.batch_norm2(x)
        x = self.leaky_relu2(x)
        x = self.cnv3_layer(x, training=training)
        x = self.batch_norm3(x)
        x = self.leaky_relu3(x)
        x = self.cnv4_layer(x, training=training)
        return x

    @staticmethod
    def loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def summary(self):
        x = tf.keras.Input(shape=(self.n_nodes, self.latent_dim))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
