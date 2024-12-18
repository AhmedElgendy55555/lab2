import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5  
x_train = np.expand_dims(x_train, axis=-1)  

BUFFER_SIZE = 60000
BATCH_SIZE = 256

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(100,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1024, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(28 * 28 * 1, activation="tanh"),
        layers.Reshape((28, 28, 1))
    ])
    return model


def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = layers.Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy")
    return gan

gan = build_gan(generator, discriminator)


def train(dataset, epochs):
    for epoch in range(epochs):
        for real_images in dataset:
            batch_size = tf.shape(real_images)[0]

            noise = tf.random.normal([batch_size, 100])
            generated_images = generator(noise)

            real_labels = tf.ones((batch_size, 1))
            fake_labels = tf.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = tf.random.normal([batch_size, 100])
            misleading_labels = tf.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, misleading_labels)

        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss[0]}, D Accuracy: {d_loss[1]}, G Loss: {g_loss}")

        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1)

def generate_and_save_images(model, epoch):
    noise = tf.random.normal([16, 100])
    generated_images = model(noise, training=False)
    generated_images = (generated_images + 1) / 2.0  

    fig = plt.figure(figsize=(4, 4))

    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap="gray")
        plt.axis("off")

    plt.savefig(f"image_at_epoch_{epoch}.png")
    plt.show()

EPOCHS = 50
train(dataset, EPOCHS)
