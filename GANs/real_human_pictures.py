# -*- coding: utf-8 -*-
"""real_human_pictures.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZzKeyly2hTx1_p2nsHB21bPDW9wePRKJ
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image as Img
from keras import Input
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
import os
from PIL import Image
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

PIC_DIR = '/content/drive/MyDrive/5000_images'
TOTAL_IMAGES = 500

ORIG_WIDTH = 178
ORIG_HEIGHT = 208
diff = (ORIG_HEIGHT - ORIG_WIDTH) // 2

WIDTH = 64
HEIGHT = 64

crop_rect = (0, diff, ORIG_WIDTH, ORIG_HEIGHT - diff)

images = []
for pic_file in tqdm(os.listdir(PIC_DIR)[:TOTAL_IMAGES]):
    pic_path = os.path.join(PIC_DIR, pic_file)

    if pic_file.endswith(('.png', '.jpg', '.jpeg')):
        pic = Image.open(pic_path).crop(crop_rect)
        pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
        images.append(np.uint8(pic))

print(f"Loaded {len(images)} images.")

images = np.array(images) / 255
print(images.shape)

plt.figure(1, figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.show()

LATENT_DIM = 32
CHANNELS = 3

def create_generator():
    gen_input = Input(shape=(LATENT_DIM, ))

    x = Dense(128 * 16 * 16)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(128, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator

def create_discriminator():
    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    x = Conv2D(128, 3)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = Adam(
        learning_rate=.0001
    )

    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return discriminator

generator = create_generator()
generator.summary()

discriminator = create_discriminator()
discriminator.trainable = False
discriminator.summary()

gan_input = Input(shape=(LATENT_DIM, ))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

optimizer = Adam(learning_rate=.0001,decay=1e-8)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')

gan.summary()

import time
iters = 2000
batch_size = 2
RES_DIR = 'res2'
FILE_PATH = '%s/generated_%d.png'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

CONTROL_SIZE_SQRT = 6
control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT**2, LATENT_DIM)) / 2

start = 0
d_losses = []
a_losses = []
images_saved = 0
for step in range(iters):
    start_time = time.time()
    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    generated = generator.predict(latent_vectors)

    real = images[start:start + batch_size]
    combined_images = np.concatenate([generated, real])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += .05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)
    d_losses.append(d_loss)

    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
    a_losses.append(a_loss)

    start += batch_size
    if start > images.shape[0] - batch_size:
        start = 0

    if step % 50 == 49:
        gan.save_weights('/gan.weights.h5')
        control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
        control_generated = generator.predict(control_vectors)

        for i in range(CONTROL_SIZE_SQRT ** 2):
            x_off = i % CONTROL_SIZE_SQRT
            y_off = i // CONTROL_SIZE_SQRT
            control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT, :] = control_generated[i, :, :, :]
        im = Img.fromarray(np.uint8(control_image * 255))
        im.save(FILE_PATH % (RES_DIR, images_saved))
        images_saved += 1

plt.figure(1, figsize=(12, 8))
plt.subplot(121)
plt.plot(d_losses, color='red')
plt.xlabel('epochs')
plt.ylabel('discriminant losses')
plt.subplot(122)
plt.plot(a_losses)
plt.xlabel('epochs')
plt.ylabel('adversary losses')
plt.show()

REC_DIR = '/content/res2'

import imageio
import os
images_to_gif = []
if os.path.isdir(RES_DIR) and len(os.listdir(RES_DIR)) > 0:
    for filename in os.listdir(RES_DIR):
        images_to_gif.append(imageio.imread(RES_DIR + '/' + filename))
    imageio.mimsave('trainnig_visual.gif', images_to_gif)
    shutil.rmtree(RES_DIR)
else:
    print(f"No images found in {RES_DIR}")
