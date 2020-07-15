# -*- coding: utf-8 -*-

"""
    load data
    build CNN
"""

import os
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


import config


def load_data(data_dir):
    """
        load data
        dataset from：
            - traing data：https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip
            - test data：https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip
        IrfanView for ppm view：https://www.irfanview.com/
    """
    # gain all subdirectory,every subdirectory as one label
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith('.ppm')]
        for f in file_names:
            # read data by fixed 
            image_data = image.load_img(f, target_size=(config.img_rows, config.img_cols))
            # pixel value change to 0-1
            image_data = image.img_to_array(image_data) / 255
            images.append(image_data)
            labels.append(int(d))

    print('{} numbers of data be loaded '.format(len(images)))
    # display all classes display
    plt.figure(figsize=(15, 8))
    sns.countplot(pd.Series(labels))
    plt.show()

    return images, labels


def display_traffic_signs(images, labels):
    """
        show data from all classes
    """
    # obtain class
    unique_labels = set(labels)

    plt.figure(figsize=(10, 10))
    for i, label in enumerate(unique_labels):
        # select the first image in every class
        image_data = images[labels.index(label)]
        plt.subplot(8, 8, i + 1)
        plt.axis('off')
        # label,sample numbers in this  class
        plt.title('Label {0} ({1})'.format(label, labels.count(label)))
        plt.imshow(image_data)
    plt.tight_layout()
    plt.show()


def process_data(images, labels):
    """
        process loaded data and label, as CNN input
    """
    X = np.array(images)
    y = np.array(labels)
    y = keras.utils.to_categorical(y, config.n_classes)

    return X, y


def build_cnn():
    """
        CNN structure
    """
    print('build CNN')
    input_shape = (config.img_rows, config.img_cols, 3)
    model = Sequential()

    # First layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Second layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Dense
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(config.n_classes))
    model.add(Activation('softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    print(model.summary())

    return model

