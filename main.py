#from model import UNet as DDPM
from dip import dip_train
from ddpm import train

if __name__ == "__main__":
    dip_model = dip_train('./data/7d8034fc-54fd-460d-a58e-3e83722fe225.jpg')

    train(dip_model)
