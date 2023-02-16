import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Dropout


class Critic(tf.keras.Model):
    def __init__(self, in_shape, n_critic):
        super().__init__()
        self.n_critic = n_critic
        self.in_shape = in_shape
        self.leaky_relu1 = LeakyReLU(alpha=0.2)
        self.leaky_relu2 = LeakyReLU(alpha=0.2)
        self.leaky_relu3 = LeakyReLU(alpha=0.2)
        self.leaky_relu4 = LeakyReLU(alpha=0.2)
        self.drop_out_layer1 = Dropout(0.3)
        self.drop_out_layer2 = Dropout(0.3)
        self.batch_norm1 = BatchNormalization()
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()
        self.cnv1_layer = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=in_shape)
        self.cnv2_layer = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=in_shape)
        self.cnv3_layer = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=in_shape)
        self.cnv4_layer = Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=in_shape)
        self.output_layer = Dense(1)

    def call(self, input_tensor, training=False):
        x = self.cnv1_layer(input_tensor, training=training)
        x = self.leaky_relu1(x)
        x = self.cnv2_layer(x, training=training)
        x = self.batch_norm1(x)
        x = self.leaky_relu2(x)
        x = self.drop_out_layer1(x)
        x = self.cnv3_layer(x, training=training)
        x = self.batch_norm2(x)
        x = self.leaky_relu3(x)
        x = self.drop_out_layer2(x)
        x = self.cnv4_layer(x, training=training)
        x = self.batch_norm3(x)
        x = self.leaky_relu4(x)
        x = Flatten()(x)
        x = Dropout(0.3)(x)
        return self.output_layer(x)

    @staticmethod
    def loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    def summary(self):
        x = tf.keras.Input(shape=self.in_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()
