#from model import UNet as DDPM
from dip import dip_train
from ddpm import train

use_dip = False

if __name__ == "__main__":

    # the image to train model
    image_path = './data/7d8034fc-54fd-460d-a58e-3e83722fe225.jpg'

    if use_dip:
        # with dip
        dip_model = dip_train(image_path)
    else:
        # without dip
        dip_model = None

    train(dip_model, image_path)
