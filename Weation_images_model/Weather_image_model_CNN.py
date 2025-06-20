

import pandas as pd
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import keras
import random
from PIL import Image
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d jehanbhathena/weather-dataset

import zipfile
zip_ref = zipfile.ZipFile('/content/weather-dataset.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

datadir = '/content/dataset'

climates = {cl: len(os.listdir(os.path.join(datadir, cl))) for cl in os.listdir(datadir)}

image_folder = '/content/dataset/rime'
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]
plt.figure(figsize=(10, 10))
plt.title('Rime images')
for i in range(9):
    plt.subplot(3, 3, i + 1)
    image_path = os.path.join(image_folder, image_files[i])
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')

plt.show()

sns.set(rc={"figure.figsize":(12, 6)})
sns.barplot(x=list(climates.keys()), y=list(climates.values()))

fig = plt.figure(figsize=(12, 12))
rows = 4
columns = 4

for i, cl in enumerate(climates.keys()):
    img_name = os.listdir(os.path.join(datadir, cl))[random.randrange(0, 100)]
    img_path = os.path.join(datadir, cl, img_name)
    img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
    fig.add_subplot(rows, columns, i+1)
    plt.axis('off')
    plt.imshow(img)
    plt.title(cl)

plt.tight_layout()
plt.show()

filepaths = []
labels = []
for i in climates.keys():
    img_path = datadir+'/'+i
    for imgs in os.listdir(img_path):
        filepaths.append(os.path.join(img_path, imgs))
        labels.append(i)
Fseries = pd.Series(filepaths, name='filepaths')
Lseries = pd.Series(labels, name='labels')
df = pd.concat([Fseries, Lseries], axis=1)

df.head()

df['labels'].value_counts().plot(kind='pie', autopct='%1.1f%%')
_ = plt.ylabel('')

"""# **Lets Process the image for trainng and testing**"""

def process_images(filepaths,target_size=(224,224)):
  images = []
  for path in filepaths:
    img = load_img(path, target_size=target_size)
    img = img_to_array(img)
    img = img/255.0
    images.append(img)
  return np.array(images)

X_train, X_test, y_train, y_test = train_test_split(df['filepaths'], df['labels'], test_size=0.2, random_state=42)

X_train = process_images(X_train)
X_test = process_images(X_test)

X_train.shape

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

y_train = to_categorical(y_train_encoded)
y_test = to_categorical(y_test_encoded)

y_train.shape

"""# **Lets Build the model**"""

model = Sequential()
model.add(Conv2D(16,(2,2),activation='relu',input_shape=(224,224,3)))
model.add(MaxPool2D(2,2))
model.add(BatchNormalization())
model.add(Conv2D(32,(2,2),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(11,activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_split=0.1,validation_data=(X_test, y_test))

model.evaluate(X_test, y_test)

model.save('weather.keras')

y_pred = model.predict(X_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_test_classes, y_pred_classes))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

mse = np.mean((y_test - y_pred)**2)
print("Mean Squared Error:", mse)

