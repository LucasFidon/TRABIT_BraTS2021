# Copyright 2021 Lucas Fidon and Suprosanna Shit

import os
from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointSaver, TensorBoardStatsHandler, MeanDice
from monai.inferers import SlidingWindowInferer
import torch
from monai.transforms import (
    Compose,
    AsDiscreted,
    AsDiscrete,
)
from src.utils.definitions import *


def get_evaluator(config, data_config, val_loader, net, optimizer, scheduler, writer, exp_name, seed, device):
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=os.path.join(SAVE_DIR, '%s_split%d' % (exp_name, seed)),
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
            save_key_metric=True,
            key_metric_n_saved=5,
            save_interval=1,
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="val_dice",
            global_epoch_transform=lambda x: scheduler.last_epoch,  # TODO: check the iteration stuff
        ),
    ]

    val_post_transform = Compose([
        AsDiscreted(
            keys=("pred", "label"),
            argmax=(True, False),
            to_onehot=True,
            n_classes=data_config['info']['n_class']
        ),
    ])
    post_pred = AsDiscrete(  # DiceMetric requires y and y_pred to be in a one-hot format since MONAI 0.5.0
        argmax=True,
        to_onehot=True,
        n_classes=data_config['info']['n_class'],
    )
    post_label = AsDiscrete(  # DiceMetric requires y and y_pred to be in a one-hot format since MONAI 0.5.0
        argmax=False,
        to_onehot=True,
        n_classes=data_config['info']['n_class_all'],
    )
    if data_config['info']['n_class_all'] > data_config['info']['n_class']:  # Partial segmentations detected
        print('Use the marginalized Mean Dice for evaluation')
        from src.handlers.marginalized_mean_dice import MarginalizedMeanDice
        key_val_metric={
            "val_dice": MarginalizedMeanDice(
                include_background=False,
                output_transform=lambda x: (post_pred(x["pred"]), post_label(x["label"])),
                labels_superset_map=data_config['info']['labels_superset']
            ),
        }
    else:  # Fully supervised learning
        key_val_metric={
            "val_dice": MeanDice(
                include_background=False,
                output_transform=lambda x: (post_pred(x["pred"]), post_label(x["label"])),
            ),
        }
    evaluator = DynUNetEvaluator(
        config=config,
        data_config=data_config,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer(
            roi_size=config['data']['patch_size'],
            sw_batch_size=config['optimization']['val_batch_size'],  # the batch size to run window slices
            overlap=0.5,
            mode="gaussian",
            sigma_scale=0.125,
            padding_mode="constant",
            cval=0.,
        ),
        post_transform=val_post_transform,
        key_val_metric=key_val_metric,
        val_handlers=val_handlers,
        amp=False,
    )

    return evaluator


# Define customized evaluator
class DynUNetEvaluator(SupervisedEvaluator):
    def __init__(self, config, data_config, **kwargs):
        # Set parameters using the config file
        # Could do something about the choice of inference-time augmentation.
        self.image_keys = data_config['info']['image_keys']
        self.label_key = 'label'

        # Initialize superclass things
        super().__init__(**kwargs)

    def get_batch(self, batchdata):
        return tuple([batchdata[key] for key in self.image_keys]), batchdata[self.label_key]
        
    def _iteration(self, engine, batchdata):
        inputs, targets = self.get_batch(batchdata)
        inputs = torch.cat(inputs, 1)
        inputs = inputs.to(engine.state.device,  non_blocking=False)
        targets = targets.to(engine.state.device,  non_blocking=False)

        def _compute_pred():
            pred = self.inferer(inputs, self.network)
            n_pred = 1

            # Perform on the fly flipping augmentation
            flip_dims = [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4)]
            for dims in flip_dims:
                flip_input = torch.flip(inputs, dims=dims)
                pred += torch.flip(
                            self.inferer(flip_input, self.network),
                            dims=dims,
                        )
                n_pred += 1
            pred /= n_pred
            return pred

        # execute forward computation
        self.network.eval()
        with torch.no_grad():
            if self.amp:
                with torch.cuda.amp.autocast():
                    predictions = _compute_pred()
            else:
                predictions = _compute_pred()
        return {"image": inputs, "label": targets, "pred": predictions}
