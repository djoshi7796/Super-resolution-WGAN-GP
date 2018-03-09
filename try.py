'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

#8-3-18 : Code susceptible to errors on account of using different modules and libraries for image-array interconversion. #TODO fix it. 

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from scipy.misc import toimage
from PIL import Image
import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    
    def generator_m(self):
        if self.G:
            return self.G
        self.G = Sequential()
        inputshape = (self.img_rows/2, self.img_cols/2, self.channel)
        self.G.add(Conv2D(16, 3, padding='same', input_shape=inputshape))
        self.G.add(Activation('relu'))
        self.G.add(MaxPooling2D((2,2), padding='same'))
        
        self.G.add(Conv2D(8, 3, padding='same'))
        self.G.add(Activation('relu'))
        self.G.add(MaxPooling2D((2,2), padding='same'))
        
        self.G.add(Conv2D(8, 3, padding='same'))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D((2, 2)))
        
        self.G.add(Conv2D(8, 3, padding='same'))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D((2, 2)))
        
        self.G.add(Conv2D(16, 3, padding='same'))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D((2, 2)))
        
        self.G.add(Conv2D(self.channel, 3, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator_m())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

def load_data():

    (Origin_train, _), (Origin_test, _) = mnist.load_data() # load MNIST dataset

    Down_train = [] # create scaled down images
    for x in Origin_train:
        im = toimage(x)
        im = im.resize((14,14), Image.ANTIALIAS)
        a = np.asarray(im)
        Down_train.append(a)

    Down_train = np.asarray(Down_train)

    Down_test = [] # same with test data

    for x in Origin_test:
        im = toimage(x)
        im = im.resize((14,14), Image.ANTIALIAS)
        a = np.asarray(im)
        Down_test.append(a)

    Down_test = np.asarray(Down_test)
    # normalize data
    Down_train = Down_train.astype('float32') / 255.
    Down_test = Down_test.astype('float32') / 255.
    Origin_train = Origin_train.astype('float32') / 255.
    Origin_test = Origin_test.astype('float32') / 255.
    # add 1 extra dimension so single input vector looks like [[[]]]
    Down_train = np.reshape(Down_train, (len(Down_train), 14, 14, 1))
    Down_test = np.reshape(Down_test, (len(Down_test), 14, 14, 1))
    Origin_train = np.reshape(Origin_train, (len(Origin_train), 28, 28, 1))
    Origin_test = np.reshape(Origin_test, (len(Origin_test), 28, 28, 1))

    return Origin_train, Down_train

class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        
        self.origin, self.down = load_data()

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator_m()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        
        #TODO shape jhols in train. Make 3D to 4D somewhere, probably where the noise thing is commented

        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
                
            #We probably should'nt use this line as we want the original and downsampled batches to be in sync (assuming this line creates batches of random images
            #images_train = self.origin[np.random.randint(0,self.origin.shape[0], size=batch_size), :, :, :]

            images_train = self.origin[batch_size]  #pick the first 'batch_size' number of images
            images_to_gen = self.down[batch_size]   #same for images to be generated
            
            #Again, no noise here
            #noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            
            images_fake = self.generator.predict(images_to_gen)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)
 
            #noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            y = np.ones([batch_size, 1])
            a_loss = self.adversarial.train_on_batch(images_to_gen, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.origin.shape[0], samples)
            images = self.origin[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)

