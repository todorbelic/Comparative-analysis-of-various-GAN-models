import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from dcgan_generator import Generator
from dcgan_discriminator import Discriminator
from utils import real_samples, fake_samples, latent_vector, performance_summary, get_data


class DCGANModel:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics='accuracy')

        self.model = self.__get_model(generator, discriminator)

    def __get_model(self, generator, discriminator):
        discriminator.trainable = False

        model = Sequential(name="DCGAN")
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
        return model

    def train(self, dataset, n_epochs=1001, n_batch=32, n_eval=100):
        latent_dim = self.generator.latent_dim

        # Our batch to train the discriminator will consist of half real images and half fake (generated) images
        half_batch = int(n_batch / 2)

        # We will manually enumare epochs
        for i in range(n_epochs):

            # Discriminator training
            # Prep real samples
            x_real, y_real = real_samples(half_batch, dataset)
            # Prep fake (generated) samples
            x_fake, y_fake = fake_samples(self.generator, latent_dim, half_batch)

            # Train the discriminator using real and fake samples
            X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            discriminator_loss, _ = self.discriminator.train_on_batch(X, y)

            # Generator training
            # Get values from the latent space to be used as inputs for the generator
            x_gan = latent_vector(latent_dim, n_batch)
            # While we are generating fake samples,
            # we want GAN generator model to create examples that resemble the real ones,
            # hence we want to pass labels corresponding to real samples, i.e. y=1, not 0.
            y_gan = np.ones((n_batch, 1))

            # Train the generator via a composite GAN model
            generator_loss = self.model.train_on_batch(x_gan, y_gan)

            # Evaluate the model at every n_eval epochs
            if i % n_eval == 0:
                print("Epoch number: ", i)
                print("*** Training ***")
                print("Discriminator Loss ", discriminator_loss)
                print("Generator Loss: ", generator_loss)
                performance_summary(self.generator, self.discriminator, dataset, latent_dim)


if __name__ == "__main__":
    generator = Generator(100)
    discriminator = Discriminator((64, 64, 3))

    gan_model = DCGANModel(generator, discriminator)
    data_lowres = get_data()
    gan_model.train(data_lowres, n_epochs=1001, n_batch=64, n_eval=100)