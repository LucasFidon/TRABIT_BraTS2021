"""
Adapted from https://github.com/Project-MONAI/MONAI/blob/dev/monai/handlers/mean_dice.py
"""

from typing import Callable, Union

import torch

from monai.handlers.iteration_metric import IterationMetric
from monai.utils import MetricReduction
from src.metrics.marginalized_mean_dice import MarginalizedDiceMetric


class MarginalizedMeanDice(IterationMetric):
    """
    Computes Dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        include_background: bool = True,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = "cpu",
        save_details: bool = True,
        labels_superset_map = None,
    ) -> None:
        """
        Args:
            include_background: whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.
            save_details: whether to save metric computation details per image, for example: mean dice of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.
        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        metric_fn = MarginalizedDiceMetric(
            include_background=include_background,
            reduction=MetricReduction.NONE,
            labels_superset_map=labels_superset_map,
        )
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            device=device,
            save_details=save_details,
        )