import tensorflow as tf
from models.utils import generate_and_save_images
import numpy as np

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

class BEGAN:
    def __init__(self, generator, critic, gamma):
        self.generator = generator
        self.discriminator = critic
        self.gamma = gamma

    @staticmethod
    def began_autoencoder_loss(out, inp):
        diff = tf.abs(out-inp)
        return tf.reduce_mean(diff)

    def get_loss_values(self, k_t, gamma, d_real_in, d_real_out, d_gen_in, d_gen_out):
        ae_real = self.began_autoencoder_loss(d_real_out, d_real_in)
        ae_gen = self.began_autoencoder_loss(d_gen_out, d_gen_in)

        d_loss = ae_real - k_t * ae_gen
        g_loss = ae_gen

        lambda_v = 0.001
        k_tp = k_t + lambda_v * (gamma * ae_real - ae_gen)

        convergence_measure = ae_real + np.abs(gamma * ae_real - ae_gen)

        return g_loss, d_loss, k_tp, convergence_measure

    def train_step(self, k_t, images, batch_size):

        d_gen_in = tf.random.normal([batch_size, self.generator.latent_dim])
        d_real_in = images

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(d_gen_in, training=True)

            d_generated_images = self.discriminator(generated_images, training=True)
            discriminated_images = self.discriminator(d_real_in, training=True)

            gen_loss, disc_loss, k_t, convergence_measure = self.get_loss_values(k_t, self.gamma, d_real_in, discriminated_images, generated_images, d_generated_images)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss, k_t, convergence_measure

    def train(self, dataset, n_epochs=100, n_batch=16, n_images=1000):
        step_size = n_images // n_batch
        k_t = 0.0
        step = 0
        convergence_measure = 0.0
        seed = tf.random.normal([16, self.generator.latent_dim])
        for epoch in range(n_epochs):
            g_loss = 0
            d_loss = 0
            for _ in range(step_size):
                g_loss, d_loss, k_t, convergence_measure = self.train_step(min(max(k_t, 0.0), 1.0),
                                                                      next(iter(dataset)), n_batch)
                step += 1

                if (step % 300 == 0):
                    print('Generator loss:{} Discrimantor loss:{} Convergence:{} K_t:{} step: {}'.format(g_loss, d_loss,
                                                                                                         convergence_measure,
                                                                                                        k_t, step))
                    self.generator.save_weights('/content/gdrive/MyDrive/GAN Project/weights/BEGAN/generator/' + str(step))
                    self.discriminator.save_weights('/content/gdrive/MyDrive/GAN Project/weights/BEGAN/generator/' + str(step))
                    generate_and_save_images(self.generator,
                                             epoch + 1,
                                             seed)

            print('Generator loss:{} Discrimantor loss:{} Convergence:{} K_t:{} step: {}'.format(g_loss, d_loss,
                                                                                                 convergence_measure,
                                                                                                 k_t, step))
            generate_and_save_images(self.generator,
                                     epoch + 1,
                                     seed)

        generate_and_save_images(self.generator,
                                 n_epochs,
                                 seed)
