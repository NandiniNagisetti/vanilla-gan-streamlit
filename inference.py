import tensorflow as tf
import numpy as np

NOISE_DIM = 100

def load_generator_model():
    return tf.keras.models.load_model("G_final.keras")

def generate_images(generator, num_images=5):
    noise = tf.random.normal([num_images, NOISE_DIM])
    images = generator(noise, training=False)
    images = (images + 1) / 2.0
    return images.numpy()
