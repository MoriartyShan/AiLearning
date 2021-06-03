#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf
print(tf.__name__, tf.__version__)
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_all, x_test = x_train_all / 255.0, x_test / 255.0
x_valid = x_train_all[:5000]
x_train = x_train_all[5000:]
y_valid = y_train_all[:5000]
y_train = y_train_all[5000:]
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(150, activation='sigmoid'))
model.add(tf.keras.layers.Dense(100, activation='sigmoid'))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_all, y_train_all, batch_size=1, epochs=100, validation_data=(x_valid, y_valid))
model.evaluate(x_test, y_test)
