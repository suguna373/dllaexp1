import pandas as pd
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)
x_train: (60000, 28, 28)
y_train: (60000,)
x_test: (10000, 28, 28)
y_test: (10000,)
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape
IMAGE_CHANNELS = 1
print('IMAGE_WIDTH:', IMAGE_WIDTH);
print('IMAGE_HEIGHT:', IMAGE_HEIGHT);
print('IMAGE_CHANNELS:', IMAGE_CHANNELS);
IMAGE_WIDTH: 28
IMAGE_HEIGHT: 28
IMAGE_CHANNELS: 1
pd.DataFrame(x_train[0])


x_train = x_train / 255.0
x_test = x_test / 255.0
pd.DataFrame(x_train[0])


x_train = x_train / 255.0
x_test = x_test / 255.0
pd.DataFrame(x_train[0])

x_train = x_train / 255.0
x_test = x_test / 255.0
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step
import numpy as np
# Reshape the data to add a channel dimension
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# Define CNN architecture with stride, pooling, and filter details
model = Sequential()
# Convolutional Layer 1
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
 input_shape=(28, 28, 1), padding="valid", name="Conv1"))
# Max Pooling Layer 1
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="Pool1"))
# Convolutional Layer 2
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
 padding="same", name="Conv2"))
# Max Pooling Layer 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="Pool2"))
# Convolutional Layer 3
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', 
 padding="same", name="Conv3"))
# Flatten Layer
model.add(Flatten(name="Flatten"))
# Fully Connected Layer
model.add(Dense(64, activation='relu', name="Dense1"))
# Dropout Layer (Regularization)
model.add(Dropout(0.5, name="Dropout"))
# Output Layer (10 classes for digits 0-9)
model.add(Dense(10, activation='softmax', name="Output"))
model.summary()
Model: "sequential_3"

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
