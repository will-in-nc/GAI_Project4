import copy
import json
import os
import warnings

import torch
from absl import app, flags
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from utils.both import get_inception_and_fid_score

from dip import dip_train

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
total_steps = 25000  # total training steps
img_size = 32  # image size
warmup = 5000  # learning rate warmup
batch_size = 128  # batch size
num_workers = 4  # workers of Dataloader
ema_decay = 0.9999  # ema decay rate
parallel = False  # multi gpu training

# Logging & Sampling
logdir = './logs/DDPM_CIFAR10_EPS'  # log directory
sample_size = 64  # sampling size of images
sample_step = 1000  # frequency of sampling

# Evaluation
save_step = 5000  # frequency of saving checkpoints, 0 to disable during training
eval_step = 0  # frequency of evaluating model, 0 to disable during training
num_images = 50000  # the number of generated images for evaluation
fid_use_torch = False  # calculate IS and FID on gpu
fid_cache = './stats/cifar10.train.npz'  # FID cache

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, warmup) / warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, num_images, batch_size, desc=desc):
            batch_size = min(batch_size, num_images - i)
            x_T = torch.randn((batch_size, 3, img_size, img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, fid_cache, num_images=num_images,
        use_torch=fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train(dip_model):
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

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
    grid = (make_grid(next(iter(dataloader))[0][:sample_size]) + 1) / 2

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
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
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
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
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

            # evaluate
            if eval_step > 0 and step % eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))


def eval():
    # model setup
    model = UNet(
        T=T, ch=ch, ch_mult=ch_mult, attn=attn,
        num_res_blocks=num_res_blocks, dropout=dropout)
    sampler = GaussianDiffusionSampler(
        model, beta_1, beta_T, T, img_size=img_size,
        mean_type=mean_type, var_type=var_type).to(device)
    if parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(logdir, 'ckpt.pt'))

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(logdir, 'samples_ema.png'),
        nrow=16)
