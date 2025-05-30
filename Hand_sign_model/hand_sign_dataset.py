!kaggle datasets download -d datamunge/sign-language-mnist

!unzip /content/sign-language-mnist.zip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

from IPython.display import Image
Image("/content/amer_sign2.png")

Image("/content/amer_sign3.png")

Image('/content/american_sign_language.PNG')

train = pd.read_csv("/content/sign_mnist_train.csv")
test = pd.read_csv("/content/sign_mnist_test.csv")

train.shape

train.isna().sum()

test.isna().sum()

labels = train['label'].values

unique_value = np.array(labels)
np.unique(unique_value)

plt.figure(figsize = (18,8))
sns.countplot(x =labels)

train.drop('label', axis = 1, inplace = True)

for i in range(5):
  plt.subplot(1,5,i+1)
  plt.imshow(train.iloc[i].values.reshape(28,28), cmap = 'gray')
  plt.axis('off')
  plt.tight_layout()
  plt.title(labels[i])
plt.show()

images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])

from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
labels

plt.imshow(images[5].reshape(28,28))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 43)

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

plt.imshow(x_train[5].reshape(28,28))

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))

model.add(Dense(24, activation = 'softmax'))

model.compile(loss ='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=30, batch_size=64)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy graph")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])

plt.show()

test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

y_pred = model.predict(test_images)

from sklearn.metrics import accuracy_score

accuracy_score(test_labels, y_pred.round())

y_pred = y_pred[0:500]
test_labels = test_labels[0:500]

from sklearn.metrics import confusion_matrix
plt.figure(figsize=(10, 10))
cm = confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
sns.heatmap(cm, annot=True)

model.save("sigh_model.keras")

