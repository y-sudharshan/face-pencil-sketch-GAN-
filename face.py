import os
import numpy as np
import re
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ========= CONFIG =========
IMG_HEIGHT, IMG_WIDTH = 96, 96
CHANNELS = 1
EPOCHS = 300
BATCH_SIZE = 16
SAVE_INTERVAL = 50

# Make sure these paths are valid
photo_dir = r"C:/Users/user/Desktop/face sketch/archive/photos"
sketch_dir = r"C:/Users/user/Desktop/face sketch/archive/sketches"


# ========= LOAD DATA =========
def load_dataset(photo_dir, sketch_dir):
    photos, sketches = [], []

    # Verify directories exist
    if not os.path.exists(photo_dir):
        raise FileNotFoundError(f"Photo directory doesn't exist: {photo_dir}")
    if not os.path.exists(sketch_dir):
        raise FileNotFoundError(f"Sketch directory doesn't exist: {sketch_dir}")
        
    photo_files = sorted(os.listdir(photo_dir))
    sketch_files = sorted(os.listdir(sketch_dir))

    print(f"Found {len(photo_files)} photos and {len(sketch_files)} sketches")
    
    if len(photo_files) == 0 or len(sketch_files) == 0:
        raise ValueError("No files found in one or both directories")

    print("Sample photo filenames:", photo_files[:3])
    print("Sample sketch filenames:", sketch_files[:3])

    # Create a mapping for sketch files using regex pattern
    # Pattern to match: f-039-01-sz1.jpg
    sketch_pattern = re.compile(r'f-(\d+)-(\d+)')
    sketch_map = {}
    
    # Map each sketch file using the ID extracted from the filename
    for sketch_file in sketch_files:
        match = sketch_pattern.search(sketch_file)
        if match:
            # Extract numbers from sketch filename
            sketch_id = f"{match.group(1)}{match.group(2)}"
            sketch_map[sketch_id] = sketch_file
            # Also store it with just the first number as key (for more flexible matching)
            sketch_map[match.group(1)] = sketch_file
        else:
            # Fallback to standard basename
            basename = os.path.splitext(sketch_file)[0]
            sketch_map[basename] = sketch_file
    
    print(f"Created sketch mapping with {len(sketch_map)} entries")
    
    # Debugging: show some entries in the sketch map
    print("Sample sketch map entries:")
    count = 0
    for key, value in sketch_map.items():
        if count < 5:
            print(f"  Key: {key} -> Sketch: {value}")
            count += 1
        else:
            break

    # Pattern to match numbers in photo filenames
    photo_pattern = re.compile(r'(\d+)')
    
    matched_count = 0
    for photo_name in photo_files:
        photo_path = os.path.join(photo_dir, photo_name)
        
        # Try multiple matching strategies
        matched_sketch = None
        
        # Strategy 1: Extract ID using regex
        matches = photo_pattern.findall(photo_name)
        if matches:
            for match in matches:
                if match in sketch_map:
                    matched_sketch = sketch_map[match]
                    break
        
        # Strategy 2: Use basename
        if not matched_sketch:
            basename = os.path.splitext(photo_name)[0]
            if basename in sketch_map:
                matched_sketch = sketch_map[basename]
        
        # If no match found, continue to next photo
        if not matched_sketch:
            print(f"No matching sketch for photo: {photo_name}")
            continue

        sketch_path = os.path.join(sketch_dir, matched_sketch)
        
        if not os.path.exists(photo_path):
            print(f"Photo file doesn't exist: {photo_path}")
            continue
            
        if not os.path.exists(sketch_path):
            print(f"Sketch file doesn't exist: {sketch_path}")
            continue

        try:
            photo = Image.open(photo_path).convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
            sketch = Image.open(sketch_path).convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
            photos.append(np.array(photo))
            sketches.append(np.array(sketch))
            matched_count += 1
            if matched_count % 10 == 0:
                print(f"Processed {matched_count} image pairs...")
        except Exception as e:
            print(f"Failed to load: {photo_path}, {sketch_path} | {e}")

    if len(photos) == 0:
        raise ValueError("❌ No image pairs loaded. Check directory paths and file matching.")

    photos = (np.expand_dims(np.array(photos), -1).astype("float32") - 127.5) / 127.5
    sketches = (np.expand_dims(np.array(sketches), -1).astype("float32") - 127.5)
    print(f"✅ Loaded {len(photos)} image pairs.")
    print(f"Photo array shape: {photos.shape}, Sketch array shape: {sketches.shape}")
    return photos, sketches


# ========= GENERATOR =========
def build_generator():
    # Generator code unchanged
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    e1 = Conv2D(64, 4, strides=2, padding='same')(inputs)
    e1 = LeakyReLU(0.2)(e1)

    e2 = Conv2D(128, 4, strides=2, padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = LeakyReLU(0.2)(e2)

    e3 = Conv2D(256, 4, strides=2, padding='same')(e2)
    e3 = BatchNormalization()(e3)
    e3 = LeakyReLU(0.2)(e3)

    d1 = Conv2DTranspose(128, 4, strides=2, padding='same')(e3)
    d1 = BatchNormalization()(d1)
    d1 = tf.keras.layers.ReLU()(d1)
    d1 = Concatenate()([d1, e2])

    d2 = Conv2DTranspose(64, 4, strides=2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = tf.keras.layers.ReLU()(d2)
    d2 = Concatenate()([d2, e1])

    outputs = Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh')(d2)
    return Model(inputs, outputs)


# ========= DISCRIMINATOR =========
def build_discriminator():
    # Discriminator code unchanged
    input_photo = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    input_sketch = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    merged = Concatenate()([input_photo, input_sketch])

    x = Conv2D(64, 4, strides=2, padding='same')(merged)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, 4, padding='same', activation='sigmoid')(x)
    return Model([input_photo, input_sketch], x)


# ========= SAVE SAMPLE IMAGE =========
def save_sample(epoch, generator, photos, output_dir="pix2pix_output"):
    os.makedirs(output_dir, exist_ok=True)
    generated = generator.predict(photos[:1], verbose=0)[0, :, :, 0]
    output = (generated + 1) * 127.5
    plt.imshow(output.astype("uint8"), cmap="gray")
    plt.axis("off")
    plt.savefig(f"{output_dir}/epoch_{epoch}.png")
    plt.close()


# ========= MAIN =========
if __name__ == "__main__":
    try:
        # Check if directories exist
        if not os.path.exists(photo_dir):
            print(f"❌ Photo directory doesn't exist: {photo_dir}")
            exit(1)
        if not os.path.exists(sketch_dir):
            print(f"❌ Sketch directory doesn't exist: {sketch_dir}")
            exit(1)
        
        # Print sample files for debugging
        print("\nSample photos:")
        for f in sorted(os.listdir(photo_dir))[:5]:
            print(f"  {f}")
            
        print("\nSample sketches:")
        for f in sorted(os.listdir(sketch_dir))[:5]:
            print(f"  {f}")
            
        photos, sketches = load_dataset(photo_dir, sketch_dir)
        
        # Adjust batch size if needed
        actual_batch_size = min(BATCH_SIZE, photos.shape[0])
        if actual_batch_size < BATCH_SIZE:
            print(f"⚠️ Warning: Reducing batch size from {BATCH_SIZE} to {actual_batch_size} due to dataset size")
            BATCH_SIZE = actual_batch_size

        generator = build_generator()
        discriminator = build_discriminator()
        discriminator.compile(loss="binary_crossentropy", optimizer=Adam(2e-4, 0.5), metrics=["accuracy"])

        # Print model summaries
        print("Generator summary:")
        generator.summary()
        print("\nDiscriminator summary:")
        discriminator.summary()

        discriminator.trainable = False
        input_photo = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
        fake_sketch = generator(input_photo)
        validity = discriminator([input_photo, fake_sketch])
        combined = Model(input_photo, validity)
        combined.compile(loss="binary_crossentropy", optimizer=Adam(2e-4, 0.5))

        # Calculate the output shape of the discriminator (12, 12, 1)
        # or use discriminator.output_shape to get it dynamically
        output_shape = discriminator.output_shape[1:]
        print(f"Discriminator output shape: {output_shape}")

        for epoch in range(EPOCHS):
            # Safety check
            if photos.shape[0] < BATCH_SIZE:
                print("❌ Not enough images to train. Please check your dataset.")
                break

            idx = np.random.randint(0, photos.shape[0], BATCH_SIZE)
            real_photos = photos[idx]
            real_sketches = sketches[idx]

            fake_sketches = generator.predict(real_photos, verbose=0)
            
            # Use the actual discriminator output shape
            valid = np.ones((BATCH_SIZE,) + output_shape)
            fake = np.zeros((BATCH_SIZE,) + output_shape)

            d_loss_real = discriminator.train_on_batch([real_photos, real_sketches], valid)
            d_loss_fake = discriminator.train_on_batch([real_photos, fake_sketches], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            g_loss = combined.train_on_batch(real_photos, valid)

            print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss[0]:.4f}, Acc: {d_loss[1]*100:.2f}% | G Loss: {g_loss:.4f}")

            if epoch % SAVE_INTERVAL == 0:
                save_sample(epoch, generator, photos)
                generator.save(f"pix2pix_generator_epoch_{epoch}.h5")
                discriminator.save(f"pix2pix_discriminator_epoch_{epoch}.h5")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()