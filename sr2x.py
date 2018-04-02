import os
from argparse import ArgumentParser
from modules.file import load_model
from modules.image import load_image
from modules.interface import show, get_input
from PIL import Image
import numpy as np

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None)
    parser.add_argument('-s', '--suffix', type=str, default=None)
    return parser.parse_args()


def single(model, name, size):
    print("path :", os.path)
    if os.path.isfile(name) == False:
        print('File not exist')
        return
    image_original = np.array(Image.open(name))
    image_upscaled = load_image(name=name, size=size, resize=1)
    test_psnr, test_ssim = show(image_upscaled, image_original,name,pred = False)
    print(" &&&&&&&&&&&&&&&&&&&&&& input image type and shape ", type(image_upscaled))
    prediction = model.predict(image_upscaled)

    #image = image.reshape(218,178,3)
    print("predicted image", prediction.shape)
    pred_psnr, pred_ssim = show(prediction, image_original,name,pred = True)
    print("PSNR : ", test_psnr, pred_psnr)
    print("SSIM : ", test_ssim, pred_ssim)


#target, ref are numpy arrays


def continuous(model, size):
    print('Enter the file name (*.jpg)')
    while True:
        value = get_input()
        if os.path.isfile(value) == False:
            print('File not exist')
            continue
        image = load_image(name=value, size=size)
        show(image)
        prediction = model.predict(image)
        show(prediction)


def main():
    args = get_args()
    suffix = args.suffix
    if args.input:
        filename = args.input
    else:
        filename = False
    model = load_model('model'+suffix+'.json')
    model.load_weights('weights'+suffix+'.hdf5')
    size = (model.input_shape[2], model.input_shape[1])
    print(" ***** size ***** ", size)
    if filename:
        single(model=model, name=filename, size=size)
    else:
        continuous(model=model, size=size)


if __name__ == '__main__':
    main()
