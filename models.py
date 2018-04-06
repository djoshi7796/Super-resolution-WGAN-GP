import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import keras
from keras.models import Sequential, Model
from keras.layers import Input
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization #Please don't use in the first layer of the discriminator 
from keras.optimizers import Adam, RMSprop
from keras import initializers
from preprocessing import load_image, show

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

class WGAN_GP():
	
	def __init__(self, rows, cols, channels):
		
		self.img_rows = rows
		self.img_cols = cols
		self.channel = channels
		self.G = None
		self.D = None
		self.GM = None	#gen Model
		self.AM = None	#adversarial model
		
	
	def generator(self):
		
		if self.G:
			return self.G
		input_shape = (self.img_rows, self.img_cols, self.channel)
		self.G = Sequential()
		
		self.G.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',kernel_initializer=initializers.random_normal(mean=0,stddev=0.001),bias_initializer='zeros',padding='same',use_bias=True,input_shape=input_shape))
		self.G.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu',kernel_initializer=initializers.random_normal(mean=0,stddev=0.001),bias_initializer='zeros',padding='same',use_bias=True))
		self.G.add(Conv2D(filters=3, kernel_size=(3, 3), activation='linear',kernel_initializer=initializers.random_normal(mean=0,stddev=0.001),bias_initializer='zeros',padding='same',use_bias=True))
		return self.G	

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
	
	def discriminator_model(self):
		
		if self.DM:
			return self.DM
		optimizer = Adams(lr=0.0001, beta_1=0, beta_2=0.9)
		weight = i
		self.DM.compile('binary_crossentropy',optimizer)	
	
	def adversarial_model(self):
		
		if self.AM:
			return self.AM
		optimizer = Adams(lr=0.0001, beta_1=0, beta_2=0.9)
		self.AM.compile('binary_crossentropy',optimizer)
		

	
class CelebA_WGAN_GP():

	def __init__(self):
		self.img_rows = 178
		self.img_cols = 218
		self.channel = 3
		self.D = self.WGAN_GP.discriminator_model()
		self.A = self.WGAN_GP.adversarial_model()
		self.G = self.WGAN_GP.generator()
        self.x_train = None #data sample
        self.y_train = None #noise sample
        
        self.x_dim = 116412
        self.z_dim = 116412

        self.x = K.placeholder(K.float32, [None, self.x_dim], name = 'x')
        self.z = K.placeholder(K.float32, [None, self.z_dim], name = 'z')        


        self.g_loss = K.mean(self.d_)
        self.d_loss = K.mean(self.d) - K.mean(self.d_)

        epsilon = K.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat)

        ddx = K.gradients(d_hat, x_hat)[0]
        print(ddx.get_shape().as_list())
        ddx = K.sqrt(K.reduce_sum(K.square(ddx), axis=1)) #redo this!!!!!!!
        ddx = K.reduce_mean(K.square(ddx - 1.0) * scale)

        self.d_loss = self.d_loss + ddx

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):     




	def train(self, train_steps = 2000, batch_size = 256):
		

        

        for t in range (train_steps):
            d_iters = 5
            if t % 50 == 0 or t < 5:
                d_iters = 10	
            
            for _ in range(d_iters):
                #assuming that I'll get separate arrays for the original data and the bicubic data
                x_train  # = give data in batches of size batch size	
                y_train # = give the noisy images to batchsize, gen_model(batch_size)
            #Not sure about this yet
                self.A() #Why discriminator model here
            
            self.G()  # why generator here       
            
            

        pass
			#add an adam optimizer in the adversarial model to use the new weights

if __name__ == "__main__":

	gan = WGAN_GP(178, 218, 3)
	model = gan.generator()
	image = load_image("202431.png",(178,218)).reshape(1, 178, 218, 3)
	image_pred = load_image("202431_predicted.png",(178,218)).reshape(1, 178, 218, 3)
	image = model.predict(image)
	show(image)
	'''Data, Labels = load_images()
	print(image.shape)
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",model.predict(image))
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~",model.predict(image_pred))
	'''
		
