import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

''' 
#Generator_CNN
#Discriminator_CNN
Generator_model
Adversarial_model

Data conveyor
- preprocessing
- upscale using bicubic
Training algorithm
Plot/show images module
#PSNR/SSIM
Testing

'''


if __name__ == "main":

