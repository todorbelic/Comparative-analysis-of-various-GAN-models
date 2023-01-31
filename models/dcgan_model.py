import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from dcgan_generator import Generator
from dcgan_discriminator import Discriminator
from utils import real_samples, fake_samples, latent_vector, performance_summary, get_data

generator_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

class DCGANModel:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

    @tf.function
    def train_step(self, images, latent_dim, batch_size):
        noise = tf.random.normal([batch_size, latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator.loss(fake_output)
            disc_loss = self.discriminator.loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return disc_loss, gen_loss

    def train(self, dataset, n_epochs=100, n_batch=32, n_eval=10):
        latent_dim = self.generator.latent_dim

        # Our batch to train the discriminator will consist of half real images and half fake (generated) images
        for epoch in range(n_epochs):

            for image_batch in tqdm(dataset):
                disc_loss, gen_loss = self.train_step(image_batch, latent_dim, n_batch)

            # Evaluate the model every 100 epochs
            if epoch % n_eval == 0:
                print("Epoch number: ", epoch)
                print("*** Training ***")
                print("Discriminator Loss ", disc_loss)
                print("Generator Loss: ", gen_loss)
                performance_summary(self.generator, self.discriminator, dataset, latent_dim)

if __name__ == "__main__":
    generator = Generator(128)
    discriminator = Discriminator((64, 64, 3))

    gan_model = DCGANModel(generator, discriminator)
    data_lowres = get_data()
    gan_model.train(data_lowres, n_epochs=100, n_batch=32, n_eval=5)