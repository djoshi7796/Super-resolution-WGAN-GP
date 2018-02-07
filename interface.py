import numpy as np
from PIL import Image
import math
import os

def show(image, original, name, pred=False):
    """
    画像データを表示
    # 引数
        image : Numpy array, 画像データ
    """
    image = image[0] * 255
    psnr_val = psnr(original, image)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    if(pred):
        name, ext = os.path.splitext(name)
        image.save(name+"_bicubic","PNG")
    image.show()
    return psnr_val


def psnr(target, ref):	
    # assume RGB image

    diff = ref - target
    diff = diff.flatten('C')

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
