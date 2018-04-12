import os
import numpy as np
from PIL import Image
#from interface import show
import cv2
import math
from skimage.measure import compare_ssim
import imutils
import sys
#from tqdm import tqdm

global SCALE
global size

def to_dirname(name):
    if name[-1:] == '/':
        return name
    else:
        return name + '/'


def load_image(name, size, resize = 1):
    SCALE=2
    image = Image.open(name)
    print("before resizing", image.size)
    if(resize):
        image = image.resize((size[0]//SCALE, size[1]//SCALE))
        print("after resizing", image.size)
        image = image.resize(size, Image.BICUBIC)
    image = np.array(image)    
    image = image/255    
    image = np.array([image])    
    return image


def load_images(name, size, ext='.png'):
    SCALE=2
    x_images = []
    y_images = []
    for file in os.listdir(name):
        if os.path.splitext(file)[1] != ext:
            continue
        image = Image.open(name+file)
        if image.mode != "RGB":
            image.convert("RGB")
        x_image = image.resize((size[0]//SCALE, size[1]//SCALE))
        x_image = x_image.resize(size, Image.BICUBIC)
        x_image = np.array(x_image)
        #print("^^^^^^^^^", x_image)
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

def load_batch(dirname, size, batch, total, ext = '.png') :
    SCALE=2
    while 1:
        for i in range(total//batch):
            x_images = []
            y_images = []
            start = i*batch
            end = (i+1)*batch
            for file in os.listdir(dirname)[start:end]:
                if os.path.splitext(file)[1] != ext:
                    continue
                image = Image.open(dirname+file)
    #            if image.mode != "RGB":
    #                image.convert("RGB")
                x_image = image.resize((size[0]//SCALE, size[1]//SCALE))
                x_image = x_image.resize(size, Image.BICUBIC)
                x_image = np.array(x_image)
                #y_image = cv2.resize(size)
                y_image = image.resize(size)
                y_image = np.array(y_image)
                #print("^^^^^^^^^", x_image.shape, y_image.shape)
                x_images.append(x_image)
                y_images.append(y_image)
            x_images = np.array(x_images)
            #print("%%%%%", x_images.shape)
            y_images = np.array(y_images)
            
            x_images = x_images / 255
            y_images = y_images / 255
            yield x_images, y_images

def show(image, original, name, pred=False):
    """
    画像データを表示
    # 引数
        image : Numpy array, 画像データ
    """
    image = image[0] * 255
    print("in show image : ", image.shape)
    psnr_val = psnr(image, original)
    ssim_val = ssim(image, original)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    if(pred):
        name, ext = os.path.splitext(name)
        image.save(name+"_pred.png","PNG")
    image.show()
    return psnr_val, ssim_val


def psnr(target, ref):	
    # assume RGB image

    diff = ref - target
    diff = diff.flatten('C')
    print('difference is ', diff)
    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

def get_input():
    """
    標準入力から文字列を取得
    # 戻り値
        value : String, 入力値
    """
    value = input('>> ')
    value = value.rstrip()
    if value == 'q':
        exit()
    return value

def ssim(target, ref):

    imageA = ref
    imageB = target
#    print("Type of original image : ", ref)
#    print("Type of modified image : ", target)

    imageB = imageB.astype(np.uint8)
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    #print("SSIM: {}".format(score))   
    return format(score)
