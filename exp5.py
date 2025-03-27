from tensorflow.keras.datasets import mnist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)

(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape
IMAGE_CHANNELS = 1
print('IMAGE_WIDTH:', IMAGE_WIDTH)
print('IMAGE_HEIGHT:', IMAGE_HEIGHT)
print('IMAGE_CHANNELS:', IMAGE_CHANNELS)

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu',
           input_shape=(28, 28, 1), padding="valid", name="Conv1"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="Pool1"),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
           padding="same", name="Conv2"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid", name="Pool2"),
    Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu',
           padding="same", name="Conv3"),
    Flatten(name="Flatten"),
    Dense(64, activation='relu', name="Dense1"),
    Dropout(0.5, name="Dropout"),
    Dense(10, activation='softmax', name="Output")
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
