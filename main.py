import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, ReLU, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
from sklearn.utils import shuffle

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Custom InstanceNormalization Layer
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=input_shape[-1:], initializer="ones", trainable=True)
        self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer="zeros", trainable=True)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        return self.scale * (inputs - mean) / tf.sqrt(variance + self.epsilon) + self.offset

IMG_HEIGHT = 96
IMG_WIDTH = 96
CHANNELS = 1
BATCH_SIZE = 16
EPOCHS = 300

# Dataset loader
def load_dataset(photo_dir, sketch_dir):
    photo_images = []
    sketch_images = []
    photo_files = sorted(os.listdir(photo_dir))
    sketch_files = sorted(os.listdir(sketch_dir))

    for photo_file, sketch_file in zip(photo_files, sketch_files):
        try:
            photo_path = os.path.join(photo_dir, photo_file)
            sketch_path = os.path.join(sketch_dir, sketch_file)

            photo = Image.open(photo_path).convert('L').resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
            sketch = Image.open(sketch_path).convert('L').resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)

            photo = np.expand_dims(np.asarray(photo, dtype=np.float32) / 127.5 - 1, axis=-1)
            sketch = np.expand_dims(np.asarray(sketch, dtype=np.float32) / 127.5 - 1, axis=-1)

            if photo.shape == (IMG_HEIGHT, IMG_WIDTH, 1) and sketch.shape == (IMG_HEIGHT, IMG_WIDTH, 1):
                photo_images.append(photo)
                sketch_images.append(sketch)

        except Exception as e:
            print(f"Skipped: {photo_file}, {sketch_file} due to error: {e}")
            continue

    return np.array(photo_images, dtype=np.float32), np.array(sketch_images, dtype=np.float32)

# Generator
def conv_block(x, filters, batch_norm=True):
    x = Conv2D(filters, 4, strides=2, padding='same')(x)
    if batch_norm:
        x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

def deconv_block(x, skip, filters, dropout=False):
    x = Conv2DTranspose(filters, 4, strides=2, padding='same')(x)
    x = InstanceNormalization()(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = ReLU()(x)
    x = Concatenate()([x, skip])
    return x

def build_unet_generator():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    e1 = conv_block(inputs, 64, batch_norm=False)
    e2 = conv_block(e1, 128)
    e3 = conv_block(e2, 256)
    e4 = conv_block(e3, 512)
    e5 = conv_block(e4, 512)
    d1 = deconv_block(e5, e4, 512, dropout=True)
    d2 = deconv_block(d1, e3, 256, dropout=True)
    d3 = deconv_block(d2, e2, 128)
    d4 = deconv_block(d3, e1, 64)
    outputs = Conv2DTranspose(CHANNELS, 4, strides=2, padding='same', activation='tanh')(d4)
    return Model(inputs, outputs)

# Discriminator
def build_discriminator():
    inp = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    tar = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    x = Concatenate()([inp, tar])
    x = conv_block(x, 64, batch_norm=False)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = conv_block(x, 512)
    x = Conv2D(1, 4, strides=1, padding='same')(x)
    return Model([inp, tar], x)

# Loss functions
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + 100 * l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

# Training loop
def train(photo_data, sketch_data):
    generator = build_unet_generator()
    discriminator = build_discriminator()
    gen_optimizer = Adam(2e-4, beta_1=0.5)
    disc_optimizer = Adam(2e-4, beta_1=0.5)

    g_losses, d_losses = [], []

    for epoch in range(EPOCHS):
        photo_data, sketch_data = shuffle(photo_data, sketch_data)
        for i in range(0, len(photo_data), BATCH_SIZE):
            input_images = photo_data[i:i+BATCH_SIZE]
            target_images = sketch_data[i:i+BATCH_SIZE]

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(input_images, training=True)

                disc_real_output = discriminator([input_images, target_images], training=True)
                input_images = tf.convert_to_tensor(input_images, dtype=tf.float32)
                gen_output = tf.convert_to_tensor(gen_output, dtype=tf.float32)
                disc_generated_output = discriminator([input_images, gen_output], training=True)

                gen_loss = generator_loss(disc_generated_output, gen_output, target_images)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        g_losses.append(gen_loss.numpy())
        d_losses.append(disc_loss.numpy())

        print(f"Epoch {epoch+1}/{EPOCHS}, Gen Loss: {gen_loss.numpy():.4f}, Disc Loss: {disc_loss.numpy():.4f}")

        if (epoch+1) % 50 == 0:
            sample = generator(tf.convert_to_tensor([photo_data[0]]), training=False)[0].numpy()
            sample = (sample + 1) * 127.5
            Image.fromarray(sample.squeeze().astype(np.uint8)).save(f"epoch_{epoch+1}.png")

    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.savefig("loss_plot.png")
    plt.show()
    generator.save("pix2pix_generator.h5")
    print("✅ Generator model saved as 'pix2pix_generator.h5'")
    return generator

if __name__ == "__main__":
    photo_dir = r"C:/Users/user/Desktop/face sketch/archive/photos"
    sketch_dir = r"C:/Users/user/Desktop/face sketch/archive/sketches"
    photos, sketches = load_dataset(photo_dir, sketch_dir)
    print(f"Dataset loaded: {len(photos)} samples")
    generator = train(photos, sketches)

