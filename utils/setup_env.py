import os
import subprocess

import torch
from torch import distributed as torch_dist


def setup_print(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        if is_master:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def initialize_DDP(launcher='pytorch'):
    if launcher == 'slurm':
        rank, local_rank, world_size = _initialize_DDP_slurm()
    elif launcher == 'pytorch':
        rank, local_rank, world_size = _initialize_DDP_pytorch()
    else:
        rank, local_rank, world_size = 0, 0, 1

    if launcher in ['slurm', 'pytorch']:
        setup_print(local_rank == 0)
        print('Distributed environment initialized.')
    else:
        print('Not initialize distributed environment.')
    return rank, local_rank, world_size


def _initialize_DDP_pytorch():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch_dist.init_process_group(backend='nccl')
    world_size = torch_dist.get_world_size()
    return rank, local_rank, world_size


def _initialize_DDP_slurm():
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    local_rank = proc_id % num_gpus
    torch.cuda.set_device(local_rank)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')

    # specify master port
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr

    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    torch_dist.init_process_group(backend='nccl')

    return proc_id, local_rank, ntasks
