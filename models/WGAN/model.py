import tensorflow as tf
from tqdm import tqdm
from generator import Generator
from critic import Critic
from models.utils import performance_summary, get_data

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.0, beta_2=0.9)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.0, beta_2=0.9)


class WGANModel:
    def __init__(self, generator, critic, gp_weight=10):
        self.generator = generator
        self.critic = critic
        self.gp_weight = gp_weight

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, images, latent_dim, batch_size):

        for i in range(self.critic.n_critic):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, latent_dim)
            )
            with tf.GradientTape() as disc_tape:
                generated_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.critic(generated_images, training=True)
                real_logits = self.critic(images, training=True)
                d_cost = self.critic.loss(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, images, generated_images)
                critic_loss = d_cost + gp * self.gp_weight

            gradients_of_critic = disc_tape.gradient(critic_loss, self.critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.critic(generated_images, training=True)
            generator_loss = self.generator.loss(gen_img_logits)

        gen_gradient = tape.gradient(generator_loss, self.generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return d_cost, generator_loss

    def train(self, dataset, n_epochs=100, n_batch=32, n_eval=10):
        latent_dim = self.generator.latent_dim

        # Our batch to train the discriminator will consist of half real images and half fake (generated) images
        for epoch in range(n_epochs):

            for image_batch in tqdm(dataset):
                batch_size = tf.shape(image_batch)[0]
                disc_loss, gen_loss = self.train_step(image_batch, latent_dim, batch_size)

            # Evaluate the model every 100 epochs
            if epoch % n_eval == 0:
                print("Epoch number: ", epoch)
                print("*** Training ***")
                print("Discriminator Loss ", disc_loss)
                print("Generator Loss: ", gen_loss)
                performance_summary(self.generator, self.critic, dataset, latent_dim)


if __name__ == "__main__":
    generator = Generator(128)
    critic = Critic((64, 64, 3), 3) ## 5??
    gan_model = WGANModel(generator, critic)
    data_lowres = get_data()
    gan_model.train(data_lowres, n_epochs=100, n_batch=32, n_eval=5)