import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, ReLU, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
from sklearn.utils import shuffle

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create output directory
output_dir = "retraining_outputs"
os.makedirs(output_dir, exist_ok=True)

# Custom normalization layer
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

# Constants
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 96, 96, 1
BATCH_SIZE = 16
START_EPOCH = 300
EPOCHS = 500

def load_dataset(photo_dir, sketch_dir):
    photo_images, sketch_images = [], []
    photo_files = sorted(os.listdir(photo_dir))
    sketch_files = sorted(os.listdir(sketch_dir))

    for photo_file, sketch_file in zip(photo_files, sketch_files):
        try:
            photo = Image.open(os.path.join(photo_dir, photo_file)).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
            sketch = Image.open(os.path.join(sketch_dir, sketch_file)).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))

            photo = np.expand_dims(np.asarray(photo, dtype=np.float32) / 127.5 - 1, axis=-1)
            sketch = np.expand_dims(np.asarray(sketch, dtype=np.float32) / 127.5 - 1, axis=-1)

            if photo.shape == (IMG_HEIGHT, IMG_WIDTH, 1) and sketch.shape == (IMG_HEIGHT, IMG_WIDTH, 1):
                photo_images.append(photo)
                sketch_images.append(sketch)
        except Exception as e:
            print(f"Skipped {photo_file}, {sketch_file} due to error: {e}")
            continue

    return np.array(photo_images, dtype=np.float32), np.array(sketch_images, dtype=np.float32)

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

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + 100 * l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

def resume_training(photo_data, sketch_data):
    generator = tf.keras.models.load_model("C:/Users/user/Desktop/face sketch/pix2pix_generator.h5", custom_objects={'InstanceNormalization': InstanceNormalization})
    discriminator = build_discriminator()

    gen_optimizer = Adam(2e-4, beta_1=0.5)
    disc_optimizer = Adam(2e-4, beta_1=0.5)

    g_losses, d_losses = [], []

    for epoch in range(START_EPOCH, EPOCHS):
        photo_data, sketch_data = shuffle(photo_data, sketch_data)

        for i in range(0, len(photo_data), BATCH_SIZE):
            input_images = photo_data[i:i+BATCH_SIZE]
            target_images = sketch_data[i:i+BATCH_SIZE]

            input_images = tf.convert_to_tensor(input_images, dtype=tf.float32)
            target_images = tf.convert_to_tensor(target_images, dtype=tf.float32)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(input_images, training=True)

                disc_real_output = discriminator([input_images, target_images], training=True)
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

        if (epoch + 1) % 50 == 0:
            sample = generator(tf.convert_to_tensor([photo_data[0]]), training=False)[0].numpy()
            sample = (sample + 1) * 127.5
            Image.fromarray(sample.squeeze().astype(np.uint8)).save(os.path.join(output_dir, f"sample_epoch_{epoch+1}.png"))

            generator.save(os.path.join(output_dir, f"generator_epoch_{epoch+1}.h5"))
            discriminator.save(os.path.join(output_dir, f"discriminator_epoch_{epoch+1}.h5"))

    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Resumed Training Loss")
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.show()

    generator.save(os.path.join(output_dir, "pix2pix_generator_final.h5"))
    discriminator.save(os.path.join(output_dir, "pix2pix_discriminator_final.h5"))
    print("\u2705 Training complete. Final models saved.")

if __name__ == "__main__":
    photo_dir = r"C:/Users/user/Desktop/face sketch/archive/photos"
    sketch_dir = r"C:/Users/user/Desktop/face sketch/archive/sketches"
    photos, sketches = load_dataset(photo_dir, sketch_dir)
    print(f"\u2705 Loaded {len(photos)} image pairs.")
    resume_training(photos, sketches)