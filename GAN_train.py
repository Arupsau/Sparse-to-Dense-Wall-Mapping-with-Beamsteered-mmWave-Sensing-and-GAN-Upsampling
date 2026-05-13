import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# Paths
radar_dir = "./radar_pcd_frames"
depth_gt_path = "./0_4.pkl"

# Hyperparameters
H, W = 128, 128    # must match what you used in radar_points_to_grid
BATCH_SIZE = 4

# 2.1 Load radar images (.npy)
radar_files = sorted(glob.glob(os.path.join(radar_dir, "frame_*_image.npy")))
print("Found radar image frames:", len(radar_files))

radar_imgs = []
for f in radar_files:
    img = np.load(f)   # (H, W)
    if img.shape != (H, W):
        # resize if needed
        img = tf.image.resize(img[..., None], (H, W)).numpy().squeeze(-1)
    radar_imgs.append(img.astype(np.float32))

radar_imgs = np.stack(radar_imgs, axis=0)  # (N, H, W)
radar_imgs = radar_imgs[..., None]  
print("Radar imgs shape:", radar_imgs.shape)



import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime

# ====================
# CONFIG
# ====================
H, W = 128, 128
RAW_H, RAW_W = 480, 640    # RealSense depth cloud resolution
depth_gt_path = "./0_4.pkl"

# ====================
# Load depth PCD PKL
# ====================
with open(depth_gt_path, "rb") as f:
    depth_data = pickle.load(f)

print("Loaded depth entries:", len(depth_data))

# =====================
# Convert depth PCD → 2D depth images
# =====================
def depth_pcd_to_depth_image(pcd, W=640, H=480):
    pts = np.vstack([pcd['f0'], pcd['f1'], pcd['f2']]).T  # (307200,3)
    pts_img = pts.reshape(H, W, 3)
    depth = np.linalg.norm(pts_img, axis=2)
    return depth.astype(np.float32)

depth_images_raw = []
depth_timestamps = []

for pcd, t in depth_data:
    depth_images_raw.append(depth_pcd_to_depth_image(pcd))
    depth_timestamps.append(t)

depth_images_raw = np.array(depth_images_raw)  # (N,480,640)
print("Depth images raw:", depth_images_raw.shape)

# ======================
# Resize to GAN input size
# ======================
depth_imgs = []
for di in depth_images_raw:
    di_resized = tf.image.resize(di[..., None], (H, W)).numpy().squeeze(-1)
    di_resized = di_resized / (np.max(di_resized) + 1e-6)
    depth_imgs.append(di_resized)

depth_imgs = np.stack(depth_imgs)   # (N,128,128)
print("Depth images resized:", depth_imgs.shape)
synced_depth = []
for i in range(len(radar_imgs)):
    depth_idx = min(i * 3, len(depth_imgs) - 1)
    synced_depth.append(depth_imgs[depth_idx])

depth_imgs = np.stack(synced_depth)
depth_imgs = np.stack(depth_imgs, axis=0)[..., None]   # (N, H, W, 1)

print("After sync:", depth_imgs.shape)
print("Radar Shape", radar_imgs.shape)



dataset = tf.data.Dataset.from_tensor_slices((radar_imgs, depth_imgs))
dataset = dataset.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)



def down_block(x, filters, batch_norm=True):
    x = layers.Conv2D(filters, 4, strides=2, padding='same',
                      use_bias=not batch_norm)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    return x

def up_block(x, skip, filters, dropout=False):
    x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(0.5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Concatenate()([x, skip])
    return x
def build_generator(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    d1 = down_block(inputs, 64, batch_norm=False)   # (64,64)
    d2 = down_block(d1, 128)                        # (32,32)
    d3 = down_block(d2, 256)                        # (16,16)
    d4 = down_block(d3, 512)                        # (8,8)
    d5 = down_block(d4, 512)                        # (4,4)
    d6 = down_block(d5, 512)                        # (2,2)

    # Bottleneck
    b = layers.Conv2D(512, 4, strides=2, padding='same')(d6)  # (1,1)
    b = layers.Activation('relu')(b)

    # Decoder
    u1 = up_block(b, d6, 512, dropout=True)         # (2,2)
    u2 = up_block(u1, d5, 512, dropout=True)        # (4,4)
    u3 = up_block(u2, d4, 512, dropout=True)        # (8,8)
    u4 = up_block(u3, d3, 256)                      # (16,16)
    u5 = up_block(u4, d2, 128)                      # (32,32)
    u6 = up_block(u5, d1, 64)                       # (64,64)

    outputs = layers.Conv2DTranspose(
        1, 4, strides=2, padding='same', activation='sigmoid'
    )(u6)                                           # (128,128,1)

    return Model(inputs, outputs, name="generator")

generator = build_generator(input_shape=(H, W, 1))
generator.summary()


def build_discriminator(input_shape=(128, 128, 1)):
    inp = layers.Input(shape=input_shape, name="radar_input")
    tar = layers.Input(shape=input_shape, name="depth_target")

    x = layers.Concatenate()([inp, tar])  # (H,W,2)

    x = down_block(x, 64, batch_norm=False)
    x = down_block(x, 128)
    x = down_block(x, 256)
    x = down_block(x, 512)

    x = layers.Conv2D(1, 4, strides=1, padding='same')(x)  # PatchGAN logits

    return Model([inp, tar], x, name="discriminator")

discriminator = build_discriminator(input_shape=(H, W, 1))
discriminator.summary()


bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_fake, gen_output, target, lambda_L1=100.0):
    # Adversarial loss: wants disc_fake to be "real"
    gan_loss = bce(tf.ones_like(disc_fake), disc_fake)
    # L1 reconstruction loss
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total = gan_loss + lambda_L1 * l1_loss
    return total, gan_loss, l1_loss

def discriminator_loss(disc_real, disc_fake):
    real_loss = bce(tf.ones_like(disc_real), disc_real)
    fake_loss = bce(tf.zeros_like(disc_fake), disc_fake)
    return real_loss + fake_loss

gen_optimizer  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


@tf.function
def train_step(radar_batch, depth_batch):
    with tf.GradientTape(persistent=True) as tape:
        gen_output = generator(radar_batch, training=True)

        disc_real = discriminator([radar_batch, depth_batch], training=True)
        disc_fake = discriminator([radar_batch, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_fake, gen_output, depth_batch
        )
        disc_loss = discriminator_loss(disc_real, disc_fake)

    gen_grads = tape.gradient(gen_total_loss, generator.trainable_variables)
    disc_grads = tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    return {
        "gen_total": gen_total_loss,
        "gen_gan": gen_gan_loss,
        "gen_l1": gen_l1_loss,
        "disc": disc_loss
    }


EPOCHS = 10  # tune as needed

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for step, (radar_batch, depth_batch) in enumerate(dataset):
        losses = train_step(radar_batch, depth_batch)

    print(f"  G_total={losses['gen_total']:.4f} | "
          f"G_L1={losses['gen_l1']:.4f} | "
          f"G_GAN={losses['gen_gan']:.4f} | "
          f"D={losses['disc']:.4f}")
    

generator.save("generator_wall_gan.h5")
discriminator.save("discriminator_wall_gan.h5")


# Take some sample frames
idx = 0  # change index to inspect different frames
radar_imgs = np.flip(radar_imgs, axis=1)
sample_radar = radar_imgs[idx:idx+1]       # (1,H,W,1)
sample_depth_gt = depth_imgs[idx]          # (H,W,1)

pred_depth = generator.predict(sample_radar)[0, ..., 0]  # (H,W)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Radar Input")
plt.imshow(sample_radar[0, ..., 0], cmap='jet')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Depth GT")
plt.imshow(sample_depth_gt[..., 0], cmap='jet')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Predicted Depth")
plt.imshow(pred_depth, cmap='jet')
plt.axis('off')

plt.tight_layout()
plt.show()

