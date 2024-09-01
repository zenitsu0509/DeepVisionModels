# -*- coding: utf-8 -*-
"""fractured_detection_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KdoC4OVFn0szZgCw3uI5X7xA5tmLdB55
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

!kaggle datasets download -d vuppalaadithyasairam/bone-fracture-detection-using-xrays

!unzip /content/bone-fracture-detection-using-xrays.zip

def load_images_and_labels(main_folder):
    data = []
    labels = []
    classes = os.listdir(main_folder)

    for label in classes:
        class_folder = os.path.join(main_folder, label)
        if os.path.isdir(class_folder):

            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (224, 224))
                    data.append(image)
                    labels.append(label)
    return np.array(data), np.array(labels)

main_folder = '/content/archive (6)/train'
data, labels = load_images_and_labels(main_folder)

plt.imshow(data[0])
plt.title(labels[0])

labels[0]

test, test_labels = load_images_and_labels('/content/archive (6)/val')

len(test)

def preprocess_images(images, image_size=(128, 128)):
    processed_images = []
    for image in images:
        image = cv2.resize(image, image_size)
        image = image.astype('float32') / 255.0
        processed_images.append(image)
    return np.array(processed_images)

def encode_labels(labels):
  labels_encoded = []
  for label in labels:
    if label == 'fractured':
        label = 1
    else :
        label = 0
    labels_encoded.append(label)
  return labels_encoded

image_size = (128, 128)
data_processed = preprocess_images(data, image_size)

labels_encoded =  encode_labels(labels)

labels_encoded = np.array(labels_encoded)
labels_encoded

x_train, x_val, y_train, y_val = train_test_split(data_processed, labels_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

test_data = preprocess_images(test,image_size)

test_labels_encoded = encode_labels(test_labels)

test_labels_encoded = np.array(test_labels_encoded)

X,Y,x,y = train_test_split(test_data,test_labels_encoded,test_size=0.2,random_state=42)

p = np.concatenate((X, Y), axis=0)
h = np.concatenate((x, y), axis=0)

model.evaluate(p,h)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(p)

y_pred.shape

y_pred[0]

plt.imshow(p[0])

cm = confusion_matrix(np.round(y_pred),h)

import seaborn as sns

sns.heatmap(cm,annot=True,fmt='d')