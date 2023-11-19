import tensorflow as tf
import os
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import optimizers
import matplotlib.pyplot as plt

Base_Directory = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay"
train_dir = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay/train"
test_dir = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay/test"
validation_dir = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay/val"
train_dir_generated = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay/train_generated"
test_dir_generated = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay/test_generated"
validation_dir_generated = "C:/Users/korad/OneDrive/Desktop/EDI/TY/Chest_XRay/val_generated"

# initilizing a Sequential model
model = models.Sequential()
# adding input layer with the 3 x 3 kernel with the activation function relu, also defining input shape of image which is 150 x 150 pixels and 3 channels(RGB)
model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (150, 150, 3)))
# max pooling the feature matrix by a 2 x 2 matrix
model.add(layers.MaxPooling2D((2, 2)))
# same way adding 3 more hidden layers
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# converting the data into a 1D array so the it can be given as input for next layers(dense)
model.add(layers.Flatten())
# defining a dense layer with dimensionality of output vector as 512
# dense layer is used for changing the dimension of the vectors by using every neuron.
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='relu'))

print(model.summary())

#compiling a model
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# generating the dataset by resizing it
# or simply put we are just normalizing the pixel size of data from range [0,255] to [0,1] so that computation is stable.

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), save_to_dir = train_dir_generated, batch_size=20, class_mode='binary')
validation_generator = val_datagen.flow_from_directory(validation_dir, target_size=(150, 150), save_to_dir = validation_dir_generated , batch_size=20, class_mode='binary')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break
    
# training a model using the generated data and validating it using generated validation data which is resized
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=5, validation_data=validation_generator, validation_steps=50)

model.save('Pneumonia.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
