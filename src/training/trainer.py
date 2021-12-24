# Copyright 2021 Lucas Fidon and Suprosanna Shit

import os
from torch.nn.functional import interpolate
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, CheckpointSaver, MeanDice
from monai.transforms import (
    Compose,
    AsDiscreted,
    AsDiscrete,
)
import torch
from torch.nn.utils import clip_grad_norm_
from src.utils.definitions import *


def get_trainer(config, data_config, train_loader, net, loss, optimizer, scheduler, writer,
                  evaluator, exp_name, seed, device, fp16=False, verbose=False, distributed=False):
    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=False,
            epoch_level=False,
        ),
        ValidationHandler(
            validator=evaluator,
            interval=config['optimization']['val_interval'],
            epoch_level=True,
        ),
        CheckpointSaver(
            save_dir=os.path.join(SAVE_DIR, '%s_split%d' % (exp_name, seed)),
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
            save_interval=1,
            n_saved=1),
        TensorBoardStatsHandler(
            writer,
            tag_name="train_loss",
            output_transform=lambda x: x["loss"],
            global_epoch_transform=lambda x: scheduler.last_epoch,  # TODO: check the iteration stuff
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="lr",
            output_transform=lambda x: x["lr"],
            global_epoch_transform=lambda x: scheduler.last_epoch,  # TODO: check the iteration stuff
        ),
    ]
    if verbose:
        train_handlers += [
            StatsHandler(
                tag_name="train_loss",
                output_transform=lambda x: x["loss"],
            ),
            StatsHandler(
                tag_name="lr",
                output_transform=lambda x: x["lr"],
                key_var_format="{}: {:.10f} "
            ),
        ]

    train_post_transform = Compose([
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=True,
            n_classes=data_config['info']['n_class'],
        ),
    ])

    key_train_metric = None

    # Computing dice scores for training volumes takes too much time.
    # It is enough to monitor the loss function values.
    # post_pred = AsDiscrete(  # DiceMetric requires y and y_pred to be in a one-hot format since MONAI 0.5.0
    #     argmax=True,
    #     to_onehot=True,
    #     n_classes=data_config['info']['n_class'],
    # )
    # post_label = AsDiscrete(  # DiceMetric requires y and y_pred to be in a one-hot format since MONAI 0.5.0
    #     argmax=False,
    #     to_onehot=True,
    #     n_classes=data_config['info']['n_class_all'],
    # )
    # if data_config['info']['n_class_all'] > data_config['info']['n_class']:  # Partial segmentations detected
    #     from src.handlers.marginalized_mean_dice import MarginalizedMeanDice
    #     key_train_metric={
    #         "train_mean_dice": MarginalizedMeanDice(
    #             include_background=False,
    #             output_transform=lambda x: (post_pred(x["pred"]), post_label(x["label"])),
    #             labels_superset_map=data_config['info']['labels_superset']
    #         ),
    #     }
    # else:  # Fully supervised learning
    #     key_train_metric={
    #         "train_mean_dice": MeanDice(
    #             include_background=False,
    #             output_transform=lambda x: (post_pred(x["pred"]), post_label(x["label"])),
    #         ),
    #     }

    trainer = DynUNetTrainer(
        config=config,
        data_config=data_config,
        device=device,
        max_epochs=config['optimization']['max_epochs'] + config['optimization']['warm_epoch'],
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        post_transform=train_post_transform,
        key_train_metric=key_train_metric,
        train_handlers=train_handlers,
        amp=fp16,
        distributed=distributed,
    )

    return trainer


# define customized trainer
class DynUNetTrainer(SupervisedTrainer):
    def __init__(self, config, data_config, **kwargs):
        self.image_keys = data_config['info']['image_keys']
        self.label_key = 'label'
        self.distributed = kwargs.pop('distributed')

        # Set the gradient clipping
        if 'gradient_clip_l2_norm' in list(config['optimization'].keys()):
            self.use_gradient_clip = True
            self.gradient_max_l2_norm = config['optimization']['gradient_clip_l2_norm']
        else:
            self.use_gradient_clip = False

        # Initialize superclass things
        super().__init__(**kwargs)

        # Check if we use a one-step (default) or a two-step optimizer (like SAM or ASAM)
        self.two_step_optimizer = False
        if 'ascent_step' in dir(self.optimizer):
            self.two_step_optimizer = True

    def get_batch(self, batchdata):
        return tuple([batchdata[key] for key in self.image_keys]), batchdata[self.label_key]

    def _iteration(self, engine, batchdata):
        inputs, targets = self.get_batch(batchdata)
        inputs = torch.cat(inputs, 1)
        inputs = inputs.to(engine.state.device,  non_blocking=False)
        targets = targets.to(engine.state.device, non_blocking=False)
        net_wo_ddp = self.network
        if self.distributed:
            net_wo_ddp = self.network.module

        def _compute_loss(preds, label):
            labels = [label] + [interpolate(label, pred.shape[2:]) for pred in preds[1:]]
            return sum([0.5 ** i * self.loss_function(p, l) for i, (p, l) in enumerate(zip(preds, labels))])

        self.network.train()  # make sure the network is in training mode
        self.optimizer.zero_grad()

        if self.amp and self.scaler is not None:  # mixed-precision
            #todo ASAM/SAM are not supported with mixed precision
            # problem is that the gradient scaler does not support closure in the optimization step

            # Runs the forward pass with autocasting.
            with torch.cuda.amp.autocast():
                predictions = self.inferer(inputs, self.network)
                loss = _compute_loss(
                    [predictions] + net_wo_ddp.get_feature_maps(), targets
                )

            # Create scaled gradients.
            self.scaler.scale(loss).backward()

            # Unscale the gradients.
            self.scaler.unscale_(self.optimizer)
            if self.use_gradient_clip:  # Clip the gradient
                clip_grad_norm_(
                    net_wo_ddp.parameters(),
                    max_norm=self.gradient_max_l2_norm,
                    norm_type=2,
                )

            # Unscale the gradients if it was not done before
            # and call optimizer.step() while skipping gradients that contain infs or NaNs
            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

        else:
            # Compute loss and gradient
            predictions = self.inferer(inputs, self.network)
            loss = _compute_loss(
                [predictions] + net_wo_ddp.get_feature_maps(), targets
            ).mean()
            loss.backward()
            if self.use_gradient_clip:  # Clip the gradient
                clip_grad_norm_(
                    net_wo_ddp.parameters(),
                    max_norm=self.gradient_max_l2_norm,
                    norm_type=2,
                )

            # Optimizer step
            if self.two_step_optimizer:  # for SAM / ASAM
                # We need to define a closure operation that performs a new forward-backward pass
                def closure():
                    pred = self.inferer(inputs, self.network)
                    loss = _compute_loss([pred] + net_wo_ddp.get_feature_maps(), targets).mean()
                    loss.backward()
                    if self.use_gradient_clip:  # Clip the gradient
                        clip_grad_norm_(
                        net_wo_ddp.parameters(),
                        max_norm=self.gradient_max_l2_norm,
                        norm_type=2,
                    )
                    return loss
            else:
                closure = None
            self.optimizer.step(closure)

        # Call the lr scheduler to update the lr at each iteration
        lr = self.optimizer.param_groups[0]['lr']
            
        return {"image": inputs, "label": targets, "pred": predictions, "loss": loss.item(), "lr":lr}
