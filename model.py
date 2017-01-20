import tensorflow as tf
import numpy as np
import csv
import os
import skimage.io
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))

def angles_to_labels(angles):
    """
    angles: numpy array of np.float64 angles (from driving_log.csv)
    """
    min_angle = -1.0
    max_angle = 1.0
    increment = 0.05

    for i in range(angles.size):
        label = 0
        x = min_angle
        while angles[i] > x and x <= max_angle:
            label += 1
            x += increment
        angles[i] = label

    return angles

def load_data(driving_log, img_dir):
    """
    driving_log: string for driving_log.csv file
    img_dir: string for driving data IMG directory
    """
    steering_angles = np.genfromtxt(driving_log, delimiter=",", usecols=(3,3), unpack=True, dtype=np.float64)[0]
    
    center_images = []
    for img in os.listdir(img_dir):
        # img_data = normalize_grayscale(skimage.io.imread(img_dir + img, as_grey=True))
        img_data = skimage.io.imread(img_dir + img)
        center_images.append(img_data)
    
    X_train = np.array(center_images)
    # X_train = normalize_grayscale(X_train)
    y_train = steering_angles
    
    return X_train, y_train

def main(_):
    X_train, y_train = load_data('driving_data/driving_log.csv', 'driving_data/IMG/')

    print("Converting angles to labels")
    y_train = angles_to_labels(y_train)
    print("One-hot encoding labels")
    lb = preprocessing.LabelBinarizer()
    y_train = lb.fit_transform(y_train)

    print("features shape: {}".format(X_train.shape))
    print("labels shape: {}".format(y_train.shape))
    
    nb_classes = y_train.shape[1]
    print("Number of classes: {}".format(nb_classes))
    input_shape = X_train.shape[1:]

    # NVIDIA CNN architecture
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.fit(X_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs, validation_split=0.2, shuffle=True)

    # Save model to JSON file
    json_string = model.to_json()
    json_file = open('model.json', 'w')
    json_file.write(json_string)

    # Save weights to file
    model.save_weights('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
