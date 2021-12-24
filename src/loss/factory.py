# Copyright 2021 Lucas Fidon and Suprosanna Shit

from src.loss.dice_ce import DiceCELoss

SUPPORTED_LOSS = [
    'Dice_CE',
    'marginalized_Dice_CE',
    'Leaf_Dice_CE',
    'WassDice_CE_BraTS',
    'WassDice_CE_BraTSv2',
]

def get_loss_function(config, data_config):
    loss_name = config['loss']['loss_name']

    # Check config parameters for the loss
    if not loss_name in SUPPORTED_LOSS:
        raise ArgumentError(
            'Loss name %s is not supported. Please choose a loss name in %s' % \
            (loss_name, SUPPORTED_LOSS)
        )

    print('\n*** Loss function\nUse %s as loss function.' % loss_name)

    # Create the loss
    if loss_name == 'Dice_CE':
        loss = DiceCELoss()
    elif loss_name == 'marginalized_Dice_CE':
        # You need to install the python package first
        from label_set_loss_functions.loss import MarginalizedDiceCE
        loss = MarginalizedDiceCE(
            labels_superset_map=data_config['info']['labels_superset'],
        )
    elif loss_name == 'Leaf_Dice_CE':
        # You need to install the python package first
        from label_set_loss_functions.loss import LeafDiceCE
        loss = LeafDiceCE(
            labels_superset_map=data_config['info']['labels_superset'],
        )
    elif loss_name == 'WassDice_CE_BraTS':
        # You need to install the python package first
        # pip install git+https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss.git
        from src.loss.gwdl_ce import GWDLCELossBraTS
        loss = GWDLCELossBraTS()
    elif loss_name == 'WassDice_CE_BraTSv2':
        # You need to install the python package first
        # pip install git+https://github.com/LucasFidon/GeneralizedWassersteinDiceLoss.git
        from src.loss.gwdl_ce import GWDLCELossBraTSv2
        loss = GWDLCELossBraTSv2()
    else:
        loss = None

    return loss

