# -*- coding: utf-8 -*-
"""handwritten_char_predictor

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/handwritten-char-predictor-15ef1eee-094d-426a-9dd0-9ff87cfb2689.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20240821/auto/storage/goog4_request%26X-Goog-Date%3D20240821T094009Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D4b7be50ea6a8f176c78703057f93e3565220eac848c89833a31bcf1c4614a4c4c5505663e1c7f989b87b9604231b392f778121d1173be4b1379589bef5b629fd4e182debc9baf56cfd320bf81927d842a1cb7d0c7550885f86c43b569327f289f9815e6bf04904a3e7bed6b7872e1828a6af55dc941d8b7f57803d013423ffe6629fe15d44a838126964f6f57631db01074e72563f81e5fceab656ff7072f231436a3f53d4610201e8d5e1ce296b04c1f6080b374156acfee979ff885fffe91d039d414029ad1b157f9b3534f84119176a0004353044c14c5ff19b56927efe62d24fb0056860f0055425fc145bbfb5873b4b2d285fd1a25f847434546cafd156
"""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d sachinpatel21/az-handwritten-alphabets-in-csv-format

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

!unzip /content/az-handwritten-alphabets-in-csv-format.zip

# load data
df = pd.read_csv("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv")

df.head()

df['0'].unique()

pixel_data =  df.iloc[0].drop('0').values

image_array = pixel_data.reshape(28, 28)


plt.figure(figsize=(5,5))
plt.imshow(image_array, cmap='gray')
plt.show()

X = df.drop('0', axis=1)
y = df['0']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(26, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model.evaluate(X_test,y_test)

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes, labels=np.arange(26))

plt.figure(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[chr(i) for i in range(65, 91)])
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())  # Use the current axes with larger figure size
plt.show()

model.save("my_model_char.keras")

