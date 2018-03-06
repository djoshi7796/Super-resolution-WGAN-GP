from PIL import Image
import cv2
import os
import math
import numpy as np
from skimage.measure import compare_ssim
import imutils
import sys


def load_images(path, size):
	x_images = []
	y_images = []
	for file in os.listdir(name):
		if os.path.splitext(file)[1] != ext:
			continue
		image = Image.open(name+file)
		if image.mode != "RGB":
			image.convert("RGB")
		x_image = image.resize((size[0]//2, size[1]//2))
		x_image = x_image.resize(size, Image.BICUBIC)
		x_image = np.array(x_image)
		#print("^^^^^^^^^", x_image.shape)
		y_image = image.resize(size)
		y_image = np.array(y_image)
		x_images.append(x_image)
		y_images.append(y_image)
		x_images = np.array(x_images)
		#print("%%%%%", x_images.shape)
		y_images = np.array(y_images)
		x_images = x_images / 255
		y_images = y_images / 255
		return x_images, y_images

def load_image(path, size):
	image = Image.open(path)
	image = np.array(image) / 255
	return image

def psnr(target, ref):	
	# assume RGB image
	
	diff = ref - target
	diff = diff.flatten('C')
	
	rmse = math.sqrt(np.mean(diff ** 2.))
	
	return 20 * math.log10(255. / rmse)

def ssim(target, ref):

    imageA = cv2.imread(ref)
    imageB = cv2.imread(target)

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))   
    return format(score)

 
def show(image):
    """
    画像データを表示
    # 引数
        image : Numpy array, 画像データ
    """
    image = image[0] * 255
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    if(False):
        name, ext = os.path.splitext(name)
        image.save(name+"_bicubic","PNG")
    image.show()
