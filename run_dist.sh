#!/bin/sh
python -m torch.distributed.launch --nproc_per_node=2 train-ddp.py --load_path $1 --startiter $2
