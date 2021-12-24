# Copyright 2021 Lucas Fidon and Suprosanna Shit

import torch
from src.optimizers.asam import ASAM, SAM
from src.optimizers.lookahead import Lookahead

SUPPORTED_OPTIMIZERS = [
    'SGD_Nesterov',
    'Adam',
    'ASAM',  # SGD Nesterov + ASAM (Adaptive Shape Aware Minimization)
    'SGDP',
    'AdamP',
    'AdamP_Lookahead',
]
SUPPORTED_LR_SCHEDULERS = [
    "polynomial_decay",
    "constant",
]

def get_optimizer(config, network):
    optim_name = config['optimization']['optimizer']

    print('\n*** Optimizer\nUse %s as optimizer.' % optim_name)

    # Check config parameters for the loss
    if not optim_name in SUPPORTED_OPTIMIZERS:
        raise ArgumentError(
            'Optimizer name %s is not supported. Please choose an optimizer name in %s' % \
            (optim_name, SUPPORTED_OPTIMIZERS)
        )
    # Create the optimizer
    if optim_name == 'SGD_Nesterov':
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=float(config['optimization']['lr']),
            momentum=float(config['optimization']['momentum']),
            weight_decay=float(config['loss']['weight_decay']),
            nesterov=True,
        )
    elif optim_name == 'Adam':
        # It is recommended to use Adam with a linear lr warmup
        # from 0 to lr_init=3e-3 during 2 / (1 - 0.999) = 2000 iterations
        # based on: "On the Adequacy of Untuned Warmup for Adaptive Optimization", 2019.
        # Not sure if the polynomial lr decay helps with Adam (never tested).
        # recommended momentum: 0.95
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=float(config['optimization']['lr']),
            betas=(float(config['optimization']['momentum']), 0.999),  # recommended: momentum=0.95
            eps=1e-5,
            weight_decay=float(config['loss']['weight_decay']),
            amsgrad=False,
        )
    elif optim_name == 'ASAM':
        base_optimizer = torch.optim.SGD(
            network.parameters(),
            lr=float(config['optimization']['lr']),
            momentum=float(config['optimization']['momentum']),
            weight_decay=float(config['loss']['weight_decay']),
            nesterov=True,
        )
        optimizer = ASAM(
            optimizer=base_optimizer,
            model=network,
            rho=0.5,  # default 0.5
            eta=0.01,  # default 0.01
        )
    elif optim_name == 'SGDP':
        # You first need to install adamp
        # python -m pip install adamp
        from adamp import SGDP
        optimizer = SGDP(
            network.parameters(),
            lr=float(config['optimization']['lr']),
            momentum=float(config['optimization']['momentum']),
            weight_decay=float(config['loss']['weight_decay']),
            nesterov=True,
        )
    elif optim_name == 'AdamP':
        # You first need to install adamp
        # python -m pip install adamp
        from adamp import AdamP
        optimizer = AdamP(
            network.parameters(),
            lr=float(config['optimization']['lr']),
            betas=(float(config['optimization']['momentum']), 0.999),
        )
    elif optim_name == 'AdamP_Lookahead':
        # You first need to install adamp
        # python -m pip install adamp
        from adamp import AdamP
        base_optimizer = AdamP(
            network.parameters(),
            lr=float(config['optimization']['lr']),
            betas=(float(config['optimization']['momentum']), 0.999),
        )
        optimizer = Lookahead(base_optimizer)
    else:
        optimizer = None

    return optimizer


def get_lr_scheduler(config, optimizer, iter_per_epoch):
    lr_scheduler_name = config['optimization']['lr_scheduler']

    # Check config parameters for the loss
    if not lr_scheduler_name in SUPPORTED_LR_SCHEDULERS:
        raise ArgumentError(
            'Learning rate scheduler name %s is not supported. Please choose a scheduler name in %s' % \
            (lr_scheduler_name, SUPPORTED_LR_SCHEDULERS)
        )

    # Create the lr scheduler
    if lr_scheduler_name == 'polynomial_decay':
        num_warmup_epoch = float(config['optimization']['warm_epoch'])
        warm_lr_init = float(config['optimization']['warm_lr_init'])
        warm_lr_final = float(config['optimization']['lr'])
        num_warmup_iter = num_warmup_epoch * iter_per_epoch
        num_after_warmup_iter = config['optimization']['max_epochs'] * iter_per_epoch
        def lr_lambda_polynomial(iter: int):
            if iter < num_warmup_epoch * iter_per_epoch:
                lr_lamda0 = warm_lr_init / warm_lr_final
                return lr_lamda0 + (1 - lr_lamda0) * iter / num_warmup_iter
            else:
                # The total number of epochs is num_warmup_epoch + max_epochs
                return (1 - (iter - num_warmup_iter) / num_after_warmup_iter) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_polynomial)

    elif lr_scheduler_name == 'constant':
        num_warmup_epoch = float(config['optimization']['warm_epoch'])
        warm_lr_init = float(config['optimization']['warm_lr_init'])
        warm_lr_final = float(config['optimization']['lr'])
        num_warmup_iter = num_warmup_epoch * iter_per_epoch
        def lr_lambda_polynomial(iter: int):
            if iter < num_warmup_epoch * iter_per_epoch:
                lr_lamda0 = warm_lr_init / warm_lr_final
                return lr_lamda0 + (1 - lr_lamda0) * iter / num_warmup_iter
            else:
                return 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_polynomial)

    else:
        scheduler = None

    return scheduler

