# Copyright 2021 Lucas Fidon and Suprosanna Shit

from monai.transforms import MapTransform
from dataset_config.loader import load_brats_data_config


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the necrotic and non-enhancing tumor
    label 4 is the ehancing tumor.
    Convert label 4 to label 3.

    In case label 3 is found aklready present, we assume that the conversion
    has been performed already and this transformation does nothing.
    """
    def __init__(self, **kwargs):
        # super(ConvertToMultiChannelBasedOnBratsClassesd, self).__init__(**kwargs)
        config = load_brats_data_config()
        self.ET_label = config['info']['labels']['ET']
        super().__init__(**kwargs)

    def __call__(self, data):
        d = dict(data)
        label3_present = (d[self.keys[0]] == 3).sum() > 0
        if not label3_present:
            d[self.keys[0]][d[self.keys[0]] == self.ET_label] = 3
        return d
