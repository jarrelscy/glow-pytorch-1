from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
from dataset import Jp2ImageFolderDataset 
import argparse
import os, random
import torch, gc
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
torch.backends.cudnn.benchmark = True
from model import Glow
import multiprocessing as mp
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
import torch.utils.data.distributed
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP 
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp
from apex.parallel import Reducer
amp.register_float_function(torch,"inverse")
amp_handle = amp.init(enabled=True)


parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--batch', default=6, type=int, help='batch size')
parser.add_argument('--iter', default=20000000, type=int, help='maximum iterations')
parser.add_argument(
    '--n_flow', default=32, type=int, help='number of flows in each block'
)
parser.add_argument('--n_block', default=7, type=int, help='number of blocks')
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
parser.add_argument('--n_bits', default=8, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--img_size', default=256, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
parser.add_argument('--path', default='//data/jp2/', type=str, help='Path to image directory')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')



def sample_data(path, batch_size, image_size, rank=None):
    #transform = transforms.Compose(
    #    [
    #        lambda x: x - 0.5,
    #    ]
    #)

    dataset = Jp2ImageFolderDataset(path, imsize=args.img_size) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank)
    loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=4, pin_memory=True)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, sampler=train_sampler,  batch_size=batch_size, num_workers=4, pin_memory=True
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

def train_iter(model, image, optimizer, args, i):
    n_bins = 2. ** args.n_bits
    image = image.to(device)
    model.zero_grad()
    log_p, logdet = model(image + torch.rand_like(image) / n_bins)
    loss, log_p, log_det = calc_loss(log_p.mean(dim=0), logdet.mean(dim=0), args.img_size, n_bins)
    
    with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    # loss.backward()    
    warmup_lr = args.lr * min(1, (i-args.startiter) * args.batch / (5000))
    # warmup_lr = args.lr
    optimizer.param_groups[0]['lr'] = warmup_lr
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss.item(), log_p.item(), log_det.item(), warmup_lr
def check_save(model_single, optimizer,args, z_sample, i, save=False):
    if i % 50 == 0 or save:
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
            model_single.state_dict(), f'checkpoint/model.pt' # _{str(i + 1).zfill(6)}
        )
        torch.save(
            optimizer.state_dict(), f'checkpoint/optimizer.pth'
        )
def train(args, model, optimizer):
    
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2. ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(1, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device)) # .half())
    try:
        with tqdm(range(args.startiter, args.iter)) as pbar:
            for i in pbar:
                image = next(dataset) # .half()
                loss, log_p, log_det, warmup_lr = train_iter(model, image, optimizer, args, i)
                
                fp.write( f'i: {i}; Loss: {loss:.5f}; logP: {log_p:.5f}; logdet: {log_det:.5f}; lr: {warmup_lr:.7f}\n')
                fp.flush()
                pbar.set_description(
                        f'Rank: {args.local_rank} Loss: {loss:.5f}; logP: {log_p:.5f}; logdet: {log_det:.5f}; lr: {warmup_lr:.7f}'
                    )
                
                if args.local_rank == 0:                    
                    check_save(model.module, optimizer, args, z_sample, i)
    except (KeyboardInterrupt, SystemExit):
        if args.local_rank == 0:
            check_save(model.module, optimizer, args, z_sample, i, save=True)
        raise


if __name__ == '__main__':
    seed = 1234
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    mp.set_start_method('spawn')
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.local_rank) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.local_rank)
    
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    
    if len(args.load_path) > 0:
        if '_' in args.load_path:
            args.startiter = int(args.load_path[:-3].split('_')[-1])
    print(args)
    fp = open(f'log_{args.local_rank}.txt', 'a')

    model_single = Glow(
        1, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    ).cpu()
    if len(args.load_path) > 0:
        model_single.load_state_dict(torch.load(args.load_path,  map_location=lambda storage, loc: storage))        
        model_single.initialize()
        gc.collect()
        torch.cuda.empty_cache()
        model_single = model_single.to(device)
    else:
        model_single = model_single.to(device)
        first_batch = next(iter(sample_data(args.path, 8, args.img_size, rank=0)))
        with torch.no_grad():
            print (args.local_rank, first_batch.mean(), first_batch.std())
            log_p, logdet = model_single(first_batch.to(device))
            print (args.local_rank, log_p)
            log_p, logdet = model_single(first_batch.to(device))
            print (args.local_rank, log_p)
    
    dp_device_ids = [args.local_rank]
    
    model = DDP(model_single, allreduce_always_fp32=True) #, device_ids=dp_device_ids, output_device=args.local_rank)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # if len(args.load_path) > 0:
    #    optim_path = '/'.join(args.load_path.split('/')[:-1])
    #    optimizer.load_state_dict(torch.load(os.path.join(optim_path, 'optimizer.pth'),  map_location=lambda storage, loc: storage))
    #    gc.collect()
    #    torch.cuda.empty_cache()
    
    train(args, model, optimizer)
