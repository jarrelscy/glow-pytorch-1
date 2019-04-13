from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
from dataset import Jp2ImageFolderDataset 
import argparse
import os
import torch, gc
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
torch.backends.cudnn.benchmark = True
from model import Glow

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--batch', default=14, type=int, help='batch size')
parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
parser.add_argument(
    '--n_flow', default=28, type=int, help='number of flows in each block'
)
parser.add_argument('--n_block', default=6, type=int, help='number of blocks')
parser.add_argument(
    '--no_lu',
    action='store_true',
    help='use plain convolution instead of LU decomposed version',
)
parser.add_argument(
    '--affine', action='store_true', help='use affine coupling instead of additive'
)
parser.add_argument('--load_path',type=str, default='',  help='load path')
parser.add_argument('--startiter', default=0, type=int, help='start iteration')
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--img_size', default=256, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
parser.add_argument('--path', default='//data/jp2/', type=str, help='Path to image directory')


def sample_data(path, batch_size, image_size):
    #transform = transforms.Compose(
    #    [
    #        lambda x: x - 0.5,
    #    ]
    #)

    dataset = Jp2ImageFolderDataset(path, imsize=args.img_size) 
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size 

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

def train_iter(model, image, optimizer, args):
    n_bins = 2. ** args.n_bits
    image = image.to(device)
    log_p, logdet = model(image + torch.rand_like(image) / n_bins)
    loss, log_p, log_det = calc_loss(log_p.mean(dim=0), logdet.mean(dim=0), args.img_size, n_bins)
    model.zero_grad()
    loss.backward()
    # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
    warmup_lr = args.lr
    optimizer.param_groups[0]['lr'] = warmup_lr
    optimizer.step()
    return loss.item(), log_p.item(), log_det.item(), warmup_lr
def check_save(model_single, optimizer,args, z_sample, i, save=False):
    if i % 10 == 0 or save:
        with torch.no_grad():
            utils.save_image(
                model_single.reverse(z_sample).cpu().data,
                f'sample/{str(i + 1).zfill(6)}.png',
                normalize=True,
                nrow=10,
                range=(-1, 1),
            )

    if i % 1000 == 0 or save:
        torch.save(
            model_single.state_dict(), f'checkpoint/model_{str(i + 1).zfill(6)}.pt'
        )
        torch.save(
            optimizer.state_dict(), f'checkpoint/optimizer.pth'
        )
def train(args, model, optimizer):
    first_batch = next(iter(sample_data(args.path, 2, args.img_size)))
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2. ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(1, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))
    try:
        with tqdm(range(args.startiter, args.iter)) as pbar:
            for i in pbar:
                image = next(dataset)
                if i == args.startiter:
                    loss, log_p, log_det, warmup_lr = train_iter(model_single, first_batch, optimizer, args)
                    model = model_single # nn.DataParallel(model_single)
                else:
                    loss, log_p, log_det, warmup_lr = train_iter(model, image, optimizer, args)

                pbar.set_description(
                    f'Loss: {loss:.5f}; logP: {log_p:.5f}; logdet: {log_det:.5f}; lr: {warmup_lr:.7f}'
                )
                check_save(model_single, optimizer, args, z_sample, i)
    except (KeyboardInterrupt, SystemExit):
        check_save(model_single, optimizer, args, z_sample, i, save=True)
        raise


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.load_path) > 0:
        args.startiter = int(args.load_path[:-3].split('_')[-1])
    print(args)

    model_single = Glow(
        1, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    ).cpu()
    if len(args.load_path) > 0:
        model_single.load_state_dict(torch.load(args.load_path,  map_location=lambda storage, loc: storage))
        
        model_single.initialize()
        gc.collect()
        torch.cuda.empty_cache()
    model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if len(args.load_path) > 0:
        optim_path = '/'.join(args.load_path.split('/')[:-1])
        optimizer.load_state_dict(torch.load(os.path.join(optim_path, 'optimizer.pth'),  map_location=lambda storage, loc: storage))
        gc.collect()
        torch.cuda.empty_cache()
    
    train(args, model, optimizer)
