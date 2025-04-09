import numpy as np
import tensorflow as tf
from PIL import Image
import os

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

# Load Generator
generator = tf.keras.models.load_model("C:/Users/user/Desktop/face sketch/retraining_outputs/pix2pix_generator_final.h5", custom_objects={'InstanceNormalization': InstanceNormalization})

# Preprocess input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((96, 96), Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1
    image = np.expand_dims(image, axis=(0, -1))  # (1, 96, 96, 1)
    return image

# Generate sketch
def generate_sketch(image_tensor):
    sketch = generator.predict(image_tensor)[0]
    sketch = (sketch + 1) * 127.5
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    return Image.fromarray(sketch.squeeze())

# --- Run the code ---
if __name__ == "__main__":
    input_path = "C:/Users/user/Desktop/face sketch/test/test2.jpg"  # Replace with your test photo path
    output_path = "C:/Users/user/Desktop/face sketch/New folder/output_sketch.jpg"  # Replace with your desired output path

    if not os.path.exists(input_path):
        print(f"Input file '{input_path}' not found.")
    else:
        input_tensor = preprocess_image(input_path)
        result = generate_sketch(input_tensor)
        result.save(output_path)
        result.show()
        print(f" Sketch saved to '{output_path}'")
        print(" Sketch generated successfully.")  