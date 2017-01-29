## Project: Use Deep Learning to Clone Driving Behavior
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview

The good team at Udacity has developed a [driving simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/January/58752736_udacity-sdc-udacity-self-driving-car-simulator-dominique-default-windows-desktop-64-bit-4/udacity-sdc-udacity-self-driving-car-simulator-dominique-default-windows-desktop-64-bit-4.zip) for this project. The goal is to build a convolutional neural network that can learn to drive the track on its own. This means recording lots of driving data in the simulator, then feeding that data into the neural network to train it.

### Approach

Recording driving data using the simulator takes a very long time, so I decided to use the [sample data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) that was provided. This data only includes driving in the middle of the lane, thus I have recorded additional data to train for recovery by recording when the car is off the track and recovering properly (on either side of the track).

My model is based off of the [NVIDIA CNN architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The layers are:

- Normalization
- Convolution (5x5 kernel with 24 output filters, 2x2 stride, ELU activation)
- Dropout (0.5 probability)
- Convolution (5x5 kernel with 36 output filters, 2x2 stride, ELU activation)
- Dropout (0.5 probability)
- Convolution (5x5 kernel with 48 output filters, 2x2 stride, ELU activation)
- Dropout (0.5 probability)
- Convolution (3x3 kernel with 64 output filters, 2x2 stride, ELU activation)
- Dropout (0.5 probability)
- Convolution (3x3 kernel with 64 output filters, 2x2 stride, ELU activation)
- Dropout (0.5 probability)
- Fully connected (100 neurons, ELU activation)
- Fully connected (50 neurons, ELU activation)
- Fully connected (10 neurons, ELU activation)
- Output (1 neuron, ELU activation)

The dropout layers after each convolutional layer are to reduce overfitting. I tried using stochastic gradient descent, but the Adam optimizer is producing better results.

To prepare the data, I extract the steering angles and image names from the driving log CSV file, combine them into a matrix and shuffle it, then separate them again.

For training, I use a generator that is able to split the data into training and test sets by providing appropriate arguments. This was a challenge because the driving data is neatly ordered, hence the prior shuffling of the data. Inside the generator, each image has a 50% chance of being flipped (along with the steering angle) to provide a better balance of left and right turning data.

*TODO Provide example images from the dataset

### Usage

To train the model, run `python model.py --batch_size [integer] --epochs [integer]`. The model is saved as model.json and the weights are saved as model.h5.

To use the model, launch the simulator and enter autonomous mode, then run `python drive.py model.json`.