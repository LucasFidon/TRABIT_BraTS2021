"""
@brief  Please, put all the user specific hyperparameters and paths here.
"""

# PATHS
SAVE_DIR = './runs'
# Folder where to save persistent dataset.
# I recommend to have it in \tmp, other wise you can get the error:
# OSError: [Errno 18] Invalid cross-device link
CACHE_DIR = '/tmp/monai_cache_dir'

# POST PROCESSING FOR BRATS (INFERENCE)
THRESHOLD_ET = 50
