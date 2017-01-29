import tensorflow as tf
import numpy as np
import cv2
import skimage.io
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

flags = tf.app.flags
FLAGS = flags.FLAGS

# Command line flags
flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")
# flags.DEFINE_integer('training_size', 1000, "Number of samples to process before going to the next epoch.")
# flags.DEFINE_integer('val_samples', 100, "Number of samples to use from validation generator at the end of every epoch.")

def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((image_data - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))

def gen_data(steering, center_img_names, img_dir, batch_size=32, test_size=0.1, test=False):
    """
    steering: numpy array of steering angles (np.float64)
    center_img_names: numpy array of center image file names (str)
    img_dir: string for driving data IMG directory
    batch_size: integer
    """
    total = len(steering)

    while True:
        features = []
        labels = []

        while len(labels) < batch_size:
            i = np.random.randint(total)
            image_name = center_img_names[i].split('/')[-1]
            image = skimage.io.imread(img_dir + image_name)
            steering_angle = steering[i]
            
            # 50% chance to flip image
            if np.random.randint(2) == 1:
                image = cv2.flip(image, 1)
                steering_angle = -steering_angle

            features.append(image)
            labels.append(steering_angle)

        yield (np.array(features), np.array(labels))

def nvidia_model(input_shape):
    # NVIDIA CNN architecture
    # http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model = Sequential()
    # Normalization
    model.add(Lambda(lambda x: x/127.5 - 1.0,
                     input_shape=input_shape,
                     name='Normalization'))
    # # Pooling to reduce training time
    # model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    # Convolutional layers with dropout to prevent overfitting
    dropout = 0.5
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(Dropout(dropout))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), activation='elu'))
    model.add(Dropout(dropout))
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(dropout))
    # Output
    model.add(Dense(1, activation='elu'))

    # opt = SGD()
    opt = Adam()
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    return model

def main(_):
    driving_log = 'driving_data/driving_log.csv'
    steering_angles = np.array([np.genfromtxt(driving_log, delimiter=',', usecols=(3), unpack=True, dtype=np.float64)])
    center_image_names = np.array([np.genfromtxt(driving_log, delimiter=',', usecols=(0), unpack=True, dtype=str)])
    # left_image_names = np.genfromtxt(driving_log, delimiter=',', usecols=(1), unpack=True, dtype=str)[0]
    # right_image_names = np.genfromtxt(driving_log, delimiter=',', usecols=(2), unpack=True, dtype=str)[0]

    data = np.append(steering_angles.T, center_image_names.T, axis=1)
    np.random.shuffle(data)
    steering_angles = data.T[0].astype(np.float64)
    center_image_names = data.T[1]

    input_shape = (160, 320, 3)
    model = nvidia_model(input_shape)
    model.summary()

    print('Batch size: ', FLAGS.batch_size)
    print('Epochs: ', FLAGS.epochs)

    test_size = 0.1
    samples_per_epoch = int(len(steering_angles) - (len(steering_angles) * test_size))
    nb_val_samples = int(len(steering_angles) * test_size)

    model.fit_generator(
        gen_data(steering_angles, center_image_names, 'driving_data/IMG/', batch_size=FLAGS.batch_size, test_size=test_size, test=False),
        samples_per_epoch=samples_per_epoch,
        nb_epoch=FLAGS.epochs,
        validation_data=gen_data(steering_angles, center_image_names, 'driving_data/IMG/', batch_size=FLAGS.batch_size, test_size=test_size, test=True),
        nb_val_samples=nb_val_samples)

    # Save model to JSON file
    json_string = model.to_json()
    json_file = open('model.json', 'w')
    json_file.write(json_string)

    # Save weights to file
    model.save_weights('model.h5')

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
