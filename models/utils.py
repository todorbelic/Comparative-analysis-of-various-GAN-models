# utils.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def real_samples(n, dataset):
    # Samples of real data
    X = dataset[np.random.choice(dataset.shape[0], n, replace=True), :]

    # Class labels
    y = np.ones((n, 1))
    return X, y


def latent_vector(latent_dim, n):
    # Generate points in the latent space
    latent_input = np.random.randn(latent_dim * n)

    # Reshape into a batch of inputs for the network
    latent_input = latent_input.reshape(n, latent_dim)
    return latent_input


def fake_samples(generator, latent_dim, n):
    # Generate points in latent space
    latent_output = latent_vector(latent_dim, n)
    # Predict outputs (i.e., generate fake samples)
    X = generator(latent_output)
    # Create class labels
    y = np.zeros((n, 1))
    return X, y


def generate_and_save_images(generator, epoch, test_input):
    predictions = generator(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :])

        plt.axis('off')

    plt.show()

def performance_summary(generator, discriminator, dataset, latent_dim, n=50):
    # Get samples of the real data
    # x_real, y_real = real_samples(n, dataset)
    # Evaluate the descriminator on real data
    #_, real_accuracy = discriminator.evaluate(x_real, y_real, verbose=0)

    # Get fake (generated) samples
    x_fake, y_fake = fake_samples(generator, latent_dim, n)
    # Evaluate the descriminator on fake (generated) data
    #_, fake_accuracy = discriminator.evaluate(x_fake, y_fake, verbose=0)

    # summarize discriminator performance
    print("*** Evaluation ***")
    #print("Discriminator Accuracy on REAL images: ", real_accuracy)
    #print("Discriminator Accuracy on FAKE (generated) images: ", fake_accuracy)

    # Display 6 fake images
    x_fake_inv_trans = x_fake.numpy().reshape(-1, 1)
    x_fake_inv_trans = x_fake_inv_trans.reshape(n, 64, 64, 3)

    fig, axs = plt.subplots(2, 3, sharey=False, tight_layout=True, figsize=(12, 6), facecolor='white')
    k = 0
    for i in range(0, 2):
        for j in range(0, 3):
            axs[i, j].matshow(x_fake_inv_trans[k])
            k = k + 1
    plt.show()


def get_data():
    img_location = '../data'
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=img_location, label_mode=None, image_size=(64, 64), batch_size=32, shuffle=True
    ).map(lambda x: x / 255.0)
    return dataset
