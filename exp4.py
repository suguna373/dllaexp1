from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import normalize
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
history = model.fit(x_train, y_train,
 epochs=100,
 batch_size=32,
 validation_data=(x_test, y_test))
test_loss = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
