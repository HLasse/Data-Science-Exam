import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense


print("version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(10,)))
model.add(Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

adam = tf.keras.optimizers.Adam(amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

a = np.random.randint(0,3,(3, 10))
hist = model.fit(a, a, validation_split=0.2, epochs=1)
