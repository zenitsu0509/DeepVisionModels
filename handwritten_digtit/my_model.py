# -*- coding: utf-8 -*-
"""My_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oGfb14V9zwl3QO7ndcKEtcfg69ntgOWs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf
from PIL import ImageOps

(X_train, y_train), (X_test, y_test) = mnist.load_data()

num_images = 5
images = X_train[:num_images]

plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

X_train = np.concatenate((X_train,X_test))
y_train = np.concatenate((y_train,y_test))

model.fit(X_train,y_train,batch_size=32,epochs=20,validation_data=(X_test,y_test),validation_split=0.2)

model.save('my_model.keras')

model = tf.keras.models.load_model('/content/my_model.keras')

def preprocess_image(image_data):
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
    plt.imshow(image, cmap='gray')
    plt.title("Original Captured Image")
    plt.axis('off')
    plt.show()
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image = np.array(image).astype('float32') / 255.0
    image = image.reshape(1, 28, 28, 1)

    return image

def predict_digit(image_data):
    image = preprocess_image(image_data)
    prediction = model.predict(image)
    print(f"Prediction Probabilities: {prediction}")

    predicted_digit = np.argmax(prediction)
    print(f"Predicted Digit: {predicted_digit}")

from google.colab import output
output.register_callback('notebook.predict_digit', predict_digit)

canvas_html = """
<canvas id="canvas" width=256 height=256 style="border:1px solid #000000;"></canvas>
<br>
<button onclick="clearCanvas()">Clear</button>
<button onclick="predictDigit()">Predict</button>
<script>
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var drawing = false;

canvas.addEventListener('mousedown', function(e) {
    drawing = true;
    draw(e);
});

canvas.addEventListener('mousemove', function(e) {
    if (drawing) {
        draw(e);
    }
});

canvas.addEventListener('mouseup', function() {
    drawing = false;
    ctx.beginPath();
});

canvas.addEventListener('mouseleave', function() {
    drawing = false;
    ctx.beginPath();
});

function draw(e) {
    ctx.lineWidth = 20;  // Increase line width for better visibility
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';  // Ensure the stroke color is white (if the background is black)
    ctx.lineTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.getBoundingClientRect().left, e.clientY - canvas.getBoundingClientRect().top);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}


function predictDigit() {
    var dataURL = canvas.toDataURL('image/png');
    var img = new Image();
    img.src = dataURL;
    img.onload = function() {
        // Convert image to base64
        var canvas2 = document.createElement('canvas');
        canvas2.width = img.width;
        canvas2.height = img.height;
        var ctx2 = canvas2.getContext('2d');
        ctx2.drawImage(img, 0, 0);
        var imageData = canvas2.toDataURL('image/png');
        google.colab.kernel.invokeFunction('notebook.predict_digit', [imageData], {});
    };
}
</script>
"""
display(widgets.HTML(canvas_html))

