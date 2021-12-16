from yacs.config import CfgNode as CN

_C = CN()

_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = '../data/PIPAL(processed)'
_C.DATASETS.NUM_WORKERS = 10
_C.DATASETS.BATCH_SIZE = 16

_C.TRAIN = CN()

_C.TRAIN.START_EPOCH = 0
_C.TRAIN.NUM_EPOCHS = 25

_C.TRAIN.WEIGHT_DIR = ''
_C.TRAIN.LOG_DIR = ''

_C.TRAIN.RESUME = CN()
_C.TRAIN.RESUME.NET_D = ''
_C.TRAIN.RESUME.NET_G = ''

_C.TRAIN.LEARNING_RATE = CN()
_C.TRAIN.LEARNING_RATE.NET_D = 1e-5
_C.TRAIN.LEARNING_RATE.NET_G = 5e-5

_C.TRAIN.CRITERION_WEIGHT = CN()
_C.TRAIN.CRITERION_WEIGHT.ERRD_REAL_ADV = 1
_C.TRAIN.CRITERION_WEIGHT.ERRD_REAL_CLF = 1
_C.TRAIN.CRITERION_WEIGHT.ERRD_REAL_QUAL = 1
_C.TRAIN.CRITERION_WEIGHT.ERRD_FAKE_ADV = 1
_C.TRAIN.CRITERION_WEIGHT.ERRD_FAKE_CLF = 1
_C.TRAIN.CRITERION_WEIGHT.ERRG_ADV = 1
_C.TRAIN.CRITERION_WEIGHT.ERRG_CLF = 1
_C.TRAIN.CRITERION_WEIGHT.ERRG_QUAL = 1
_C.TRAIN.CRITERION_WEIGHT.ERRG_CONT = 1

_C.MODEL = CN()
_C.MODEL.LATENT_DIM = 100
_C.MODEL.INCEPTION_DIMS = 2048

cfg = _C


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
