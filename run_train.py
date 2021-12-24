# Copyright 2021 Lucas Fidon and Suprosanna Shit

import os
import math
import torch
from monai.data import DataLoader, DistributedSampler
from src.data.factory import get_dataset
from src.training.evaluator import get_evaluator
from src.training.trainer import get_trainer
from src.networks.factory import get_network
from src.loss.factory import get_loss_function
from src.optimizers.factory import get_optimizer, get_lr_scheduler
from src.utils.definitions import *
from config.loader import load_config
from dataset_config.loader import load_dataset_config
import logging
import sys
from tensorboardX import SummaryWriter
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--data_config',
                    required=True,
                    help='config file (.yml) containing the hyper-parameters for the dataset. '
                         'See /dataset_config for examples.')
parser.add_argument('--exp_name',
                    default=None,
                    help='name for the experiment. If None is given use the exp_name in the config file.')
parser.add_argument('--seed',
                    default=112,
                    type=int,
                    help='random split used to split the training data into training/validation.')
parser.add_argument('--resume',
                    default=None,
                    help='checkpoint of the last epoch of the model.')
parser.add_argument('--device',
                    default='cuda',
                    help='device to use for training')
parser.add_argument('--num_workers',
                    default=0,
                    type=int,
                    help='Number of workers to use in the data loaders multi-processing. '
                         'Use 0 to deactivate multi-processing (default).')
parser.add_argument('--cuda_visible_device',
                    nargs='*',
                    type=int,
                    default=[0,1],
                    help='list of index where skip conn will be made.')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--fp16',
                    action='store_true',
                    help='(Optional) activate the mixed-precision.')
parser.add_argument('--cache',
                    action='store_true',
                    help='(Optional) Use a PersistentDataset. '
                         'Preprocessed data will be saved in the cache directory defined in /utils/definitions.py. '
                         'This allows for faster training at the cost of disk memory.')
parser.add_argument('--verbose',
                    action='store_true',)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    # Load the config files
    config = load_config(config_file=args.config)
    data_config = load_dataset_config(config_file=args.data_config, verbose=True)

    # Set device options
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.local_rank = int(os.environ["RANK"])
        args.gpu = 'cuda:%d' % args.local_rank

    device = torch.device(args.device)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.rank = torch.distributed.get_rank()
        print('Distributed:', args.distributed, "GPUs:", args.gpu)

    # Create the loss function
    loss = get_loss_function(config=config, data_config=data_config)

    # Create the data loaders
    train_ds, val_ds = get_dataset(
        config=config,
        data_config=data_config,
        seed=args.seed,
        mode='split',
        use_persistent_dataset=args.cache,
    )
    
    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
        val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=config['optimization']['batch_size'],
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,  # the val_batch_size is used for the sliding window inferer
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=val_sampler,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config['optimization']['batch_size'],
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,  # the val_batch_size is used for the sliding window inferer
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    # Create the network
    net = get_network(
        config,
        in_channels=data_config['info']['in_channels'],
        n_class=data_config['info']['n_class'],
        device=device,
    )
    if args.verbose:
        import numpy as np
        print(net)
        # Print the number of parameters
        trainable_model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        n_parameters = int(sum([np.prod(p.size()) for p in trainable_model_parameters]))
        print('\nTotal number of parameters: %.2fM' % (n_parameters / 1e6))

    net_wo_dist = net
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, broadcast_buffers=False, device_ids=[args.gpu],) #find_unused_parameters=True
        net_wo_dist = net.module

    # Set the optimization options
    optimizer = get_optimizer(config, network=net_wo_dist)
    iter_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(config, optimizer=optimizer, iter_per_epoch=iter_per_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net_wo_dist.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        last_epoch = math.floor(scheduler.last_epoch/iter_per_epoch)

    # Create the summary writter
    if args.exp_name is None:
        exp_name = config['log']['exp_name']
    else:
        exp_name = args.exp_name
    writer = SummaryWriter(
        log_dir=os.path.join(SAVE_DIR, '%s_split%d' % (exp_name, args.seed)),
    )

    evaluator = get_evaluator(
        config=config,
        data_config=data_config,
        val_loader=val_loader,
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        exp_name=exp_name,
        seed=args.seed,
        device=device,
    )
    trainer = get_trainer(
        config=config,
        data_config=data_config,
        train_loader=train_loader,
        net=net,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        evaluator=evaluator,
        exp_name=exp_name,
        seed=args.seed,
        device=device,
        fp16=args.fp16,
        verbose=args.verbose,
        distributed=args.distributed,
    )

    if args.resume:
        evaluator.state.epoch = last_epoch
        trainer.state.epoch = last_epoch
        trainer.state.iteration = iter_per_epoch * last_epoch

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    trainer.run()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
