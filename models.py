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
		#complete this	
	
	def adversarial_model(self):
		
		if self.AM:
			return self.AM
		optimizer = Adams(lr=0.0001, beta_1=0, beta_2=0.9)
		self.AM.compile('binary_crossentropy',optimizer)
		#complete this as well
		#use adam wala optimizer
		

	
class CelebA_WGAN_GP():

	def __init__(self):
		self.img_rows = 178
		self.img_cols = 218
		self.channel = 3
		self.D = self.WGAN_GP.discriminator_model()
		self.A = self.WGAN_GP.adversarial_model()
		self.G = self.WGAN_GP.generator()

	def train(self, train_steps = 2000, batch_size = 256):
		for t in range(train_steps / 5):
			for i in range(batch_size):
				pass
				#some variables thing i didn't quite understand 
			#add an adam optimizer in the adversarial model to use the new weights
		#test maybe??

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
		
