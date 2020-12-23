import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import pickle
from keras.layers import Conv2D, MaxPooling2D


pickle_import = open(r"/PetImages/X.pickle", 'rb')
x = pickle.load(pickle_import)

pickle_import = open(r"/PetImages/y.pickle", 'rb')
y = pickle.load(pickle_import)

x = x / 255.0 # Normalization / Scaling down the features, so there are no big numbers and easier to train neural netwrok

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=x.shape[1:])) #Neurons
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

model.fit(x, y, batch_size=32, epochs=100, validation_split=0.3)

model.save(r"C:\Users\sadi\PycharmProjects\DeepLearning_Cats_Dogs\PetImages\FINALML.model")
