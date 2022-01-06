from yacs.config import CfgNode as CN

_C = CN()

_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = '../data/PIPAL(processed)'
_C.DATASETS.NUM_WORKERS = 4
_C.DATASETS.BATCH_SIZE = 16

_C.TRAIN = CN()

_C.TRAIN.START_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 5

_C.TRAIN.WEIGHT_DIR = ''
_C.TRAIN.LOG_DIR = ''

_C.TRAIN.RESUME = CN()
_C.TRAIN.RESUME.NET_D = ''
_C.TRAIN.RESUME.NET_G = ''

_C.TRAIN.LEARNING_RATE = 1e-5

_C.MODEL = CN()
_C.MODEL.LATENT_DIM = 100
_C.MODEL.FIXED_FEAT_EXTRACTOR = True
_C.MODEL.FEAT_EXTRACTOR_LEVEL = 'low'
_C.MODEL.TRANSFORMER_LAYERS = 1
_C.MODEL.TRANSFORMER_DIM = 128
_C.MODEL.MHA_NUM_HEADS = 4
_C.MODEL.FEAT_DIM = 1024
_C.MODEL.HEAD_DIM = 128

cfg = _C


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
