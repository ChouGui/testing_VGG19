import os
import sys
import random
import cv2
import time
import shutil
import pickle
import numpy as np

import keras
import tensorflow as tf

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras import optimizers


# Train a model
def train():
    # Variables, 25 epochs so far
    epochs = 100
    batch_size = 64
    train_samples = 5000  # 10 categories with 5000 images in each category
    validation_samples = 250  # 10 categories with 1000 images in each category
    # validation_split=0.4
    img_width, img_height = 125, 130
    # os.rmdir("DB/train/.ipynb_checkpoints")
    # os.rmdir("DB/test/.ipynb_checkpoints")
    # Create a data generator for training
    # Making real time data augmentation
    train_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Create a data generator for validation
    validation_data_generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Create a train generator
    train_generator = train_data_generator.flow_from_directory(
        str(train_path),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical')
    # Create a test generator
    validation_generator = validation_data_generator.flow_from_directory(
        str(test_path),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical')

    # Get the model (2 categories)
    # model = tf.keras.applications.VGG19(include_top = False, input_shape = (125,130,3))

    # lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5)
    from tensorflow.keras.applications import VGG19  # For Transfer Learning

    # base_model = VGG19(weights='imagenet', input_shape=(125, 130, 3), include_top=False)
    base_model = VGG19(weights=None, input_shape=(125, 130, 3), include_top=False)
    inputs = keras.Input(shape=(125, 130, 3))
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # model.compile(optimizer = 'adam', loss=tf.keras.losses.CategoricalCrossentropy())
    # model.compile(optimizer = 'adam', loss='mean_squared_error')
    # model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())

    model.fit(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        epochs=epochs)
    # Save model to disk
    model.save('models/VGG19_ep' + str(epochs) + '_bs' + str(batch_size) + '_ts' + str(train_samples) + '_vs' + str(
        validation_samples) + '.h5')
    print('Saved model to disk!')
    # Get labels
    labels = train_generator.class_indices
    # Invert labels
    classes = {}
    for key, value in labels.items():
        classes[value] = key.capitalize()
    # Save classes to file
    with open('models/classes.pkl', 'wb') as file:
        pickle.dump(classes, file)
    print('Saved classes to disk!')

t1 = time.perf_counter()
# Run the training of the model
train()
t2 = time.perf_counter()
print(f"computation time {t2 - t1:0.4f} seconds")

# Evaluate the model
def evaluate():
    # Load the model
    model = keras.models.load_model('/content/models/VGG19_ep100_bs64_ts5000_vs250.h5')
    # Load classes
    classes = {}
    with open('models/classes.pkl', 'rb') as file:
        classes = pickle.load(file)
    # Get a list of categories
    str_test = str(test_path)
    categories = os.listdir(str_test)
    # Get a category a random
    category = random.choice(categories)
    # Print the category
    print(category)
    # Get images in a category
    images = os.listdir(str_test + '/' + category)
    # Randomize images to get different images each time
    random.shuffle(images)
    # Loop images
    blocks = []
    for i, name in enumerate(images):
        # Limit the evaluation
        if i > 6:
            break;
        # Print the name
        print(name)
        # Get the image
        image = cv2.imread(str_test + '/' + category + '/' + name)
        # Get input reshaped and rescaled
        input = np.array(image).reshape((1, 125, 130, 3)).astype('float32') / 255
        # Get predictions
        predictions = model.predict(input).ravel()
        # Print predictions
        print(predictions)
        # Get the class with the highest probability
        prediction = np.argmax(predictions)
        # Check if the prediction is correct
        correct = True if classes[prediction].lower() == category else False
        # Draw the image and show the best prediction
        image = cv2.resize(image, (128, 128))
        cv2.putText(image, '{0}: {1} %'.format(classes[prediction], str(round(predictions[prediction] * 100, 2))),
                    (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, '{0}: {1} %'.format(classes[prediction], str(round(predictions[prediction] * 100, 2))),
                    (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (65, 105, 225), 2)
        cv2.putText(image, '{0}'.format('CORRECT!' if correct else 'WRONG!'), (12, 50), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (0, 0, 0), 2)
        cv2.putText(image, '{0}'.format('CORRECT!' if correct else 'WRONG!'), (10, 48), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    (0, 255, 0) if correct else (0, 0, 255), 2)

        # Append the image
        blocks.append(image)

    # Display images and predictions
    row1 = np.concatenate(blocks[0:3], axis=1)
    row2 = np.concatenate(blocks[3:6], axis=1)

    from google.colab.patches import cv2_imshow
    cv2_imshow(np.concatenate((row1, row2), axis=0))
    # cv2.imshow('Predictions', np.concatenate((row1, row2), axis=0))
    cv2.imwrite('res/predictions.jpg', np.concatenate((row1, row2), axis=0))
    cv2.waitKey(0)

t1 = time.perf_counter()
# Run the evaluation of the model
evaluate()
t2 = time.perf_counter()
print(f"computation time {t2 - t1:0.4f} seconds")
