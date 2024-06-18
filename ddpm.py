import copy
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet

class SingleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, transform=None):
        self.image = Image.open(image_path)
        self.transform = transform
        if self.transform:
            self.image = self.transform(self.image)

    def __len__(self):
        return 8  # Only one image

    def __getitem__(self, idx):
        return self.image

device = "cuda" if torch.cuda.is_available() else "cpu"

ch = 128
ch_mult = [1, 2, 2, 2]
attn = [1]
num_res_blocks = 2
dropout = 0.1

# Define variables based on the flags
beta_1 = 1e-4  # start beta value
beta_T = 0.02  # end beta value
T = 1000  # total diffusion steps
mean_type = 'epsilon'  # predict variable
var_type = 'fixedlarge'  # variance type

# Training
lr = 2e-4  # target learning rate
grad_clip = 1.0  # gradient norm clipping
total_steps = 30000  # total training steps
img_size = 32  # image size
warmup = 5000  # learning rate warmup
batch_size = 8  # batch size
num_workers = 1  # workers of Dataloader
ema_decay = 0.9999  # ema decay rate
parallel = False  # multi gpu training

# Logging & Sampling
logdir = './logs/DDPM_CIFAR10_EPS'  # log directory
sample_size = 1  # sampling size of images
sample_step = 2500  # frequency of sampling

# Evaluation
save_step = 2500  # frequency of saving checkpoints, 0 to disable during training

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def warmup_lr(step):
    return min(step, warmup) / warmup

def train(dip_model, image_path):
    # dataset
    dataset = SingleImageDataset(
        image_path=image_path,
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)

    # model setup
    net_model = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, dip_model, beta_1, beta_T, T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, beta_1, beta_T, T, img_size,
        mean_type, var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, beta_1, beta_T, T, img_size,
        mean_type, var_type).to(device)
    if parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    if not os.path.exists(os.path.join(logdir, 'sample')):
        os.makedirs(os.path.join(logdir, 'sample'))
    x_T = torch.randn(sample_size, 3, img_size, img_size)
    x_T = x_T.to(device)
    if dip_model is not None:
        x_T = F.interpolate(x_T, scale_factor=2, mode='bilinear', align_corners=False)
        x_T = dip_model(x_T)
        x_T = F.interpolate(x_T, scale_factor=0.5, mode='bilinear', align_corners=False)

    datalooper = iter(dataloader)
    x_0 = next(datalooper).to(device)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    with trange(total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()

            use_dip = True
            loss = trainer(x_0, use_dip).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, ema_decay)

            # log
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if sample_step > 0 and step % sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0_2 = ema_sampler(x_T)
                    grid = (x_0_2[0] + 1) / 2
                    path = os.path.join(
                        logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                net_model.train()

            # save
            if save_step > 0 and step % save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(logdir, 'ckpt.pt'))

