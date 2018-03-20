import os
import numpy as np
from PIL import Image
from interface import show
import cv2
#from tqdm import tqdm
global SCALE
global size

def to_dirname(name):
    if name[-1:] == '/':
        return name
    else:
        return name + '/'


def load_image(name, size, resize = 0):
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
            print("^^^^^^^^^", x_image.shape, y_image.shape)
            x_images.append(x_image)
            y_images.append(y_image)
        x_images = np.array(x_images)
        #print("%%%%%", x_images.shape)
        y_images = np.array(y_images)
        
        x_images = x_images / 255
        y_images = y_images / 255
        yield x_images, y_images
'''    for i in range(total//batch):
       data = np.array
       ref = np.array
       start = i*batch
       end = (i+1)*batch
       #img_name = [x for x in os.listdir(dirname)[start:end]]
       img = [load_image(dirname + x, size, 0) for x in os.listdir(dirname)[start:end]]
       ref = ref(img).reshape(batch,size[1],size[0],3)
       img_resize = [load_image(dirname + x, size, 1) for x in os.listdir(dirname)[start:end]]
       data = data(img_resize).reshape(batch,size[1],size[0],3)
       yield ref[0]
'''            

if __name__ == "__main__":
    dirname = "/home/group16/Downloads/CelebA/Align/celeba_test/"    
    size = [178, 218]
    out = load_batch(dirname, size, 2, 4)
    #a,b = next(out)
    #print(len(out))
    for i in out:
        a,b = i
        print(a.shape)
        for i in range(a.shape[0]):
            img = a[i].reshape(3, 218, 178)
            show(img, img, 'randomcrap', False)
            #print(a[i].shape)
            #show(b[i], b[i], 'randomcrap', False)
        
    print(out)
