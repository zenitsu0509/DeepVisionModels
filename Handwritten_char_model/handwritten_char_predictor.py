
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

