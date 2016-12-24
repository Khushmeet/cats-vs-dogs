from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Dense, Flatten, Dropout, ZeroPadding2D
import os
import numpy as np

TEST_DIR = 'data/test/'

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 150, 150), activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory('data/train',
                                                    target_size=(150, 150),
                                                    batch_size=16,
                                                    class_mode='binary')

model.fit_generator(train_generator,
                    samples_per_epoch=2500,
                    nb_epoch=10)

model.save_weights('cat-dog-vgg19.h5')

test_imgs = [TEST_DIR + i for i in os.listdir(TEST_DIR)]

loaded_imgs = [load_img(i) for i in test_imgs]

test = [img_to_array(img) for img in loaded_imgs]

test_datagen = ImageDataGenerator(rescale=1. / 255, shuffle=False)

test_x = test_datagen.flow(test, batch_size=32, seed=None)

predictions = model.predict_generator(test_x, val_samples=12500)
print(predictions)
