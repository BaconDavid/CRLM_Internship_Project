# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.IS_CUDA = True


_C.DATA = CN()
_C.DATA.Data_dir = None
_C.DATA.Train_dir = None
_C.DATA.Valid_dir = None
_C.DATA.Test_dir = None

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.lr = 0.001
# The all important scales for the stuff
_C.TRAIN.batch_size = 16
_C.TRAIN.num_epochs = 20
_C.TRAIN.optimizer = 'Adam'
_C.TRAIN.loss = 'CrossEntropy'

_C.VALID = CN()
_C.VALID.batch_size = 16
_C.VALID.loss = 'CrossEntropy'\

_C.TEST = CN()
_C.TEST.batch_size = 16
_C.TEST.loss = 'CrossEntropy'

_C.DATASET = CN()
_C.DATASET.WeightedRandomSampler = False
_C.DATASET.num_classes = 2
_C.DATASET.num_channels = 3

_C.MODEL = CN()
_C.MODEL.name = 'Resnet10'
_C.MODEL.pretrained = False
_C.MODEL.num_classes = 2
_C.MODEL.num_in_channels = 1
_C.MODEL.dropout = 0.5
_C.MODEL.num_out_channels = 1

_C.LOG = CN()
_C.LOG.log_dir = None
_C.LOG.log_file = None

_C.Preprocess = CN()
_C.Preprocess.resize_shape = None
#_C.Preprocess.resize_width = 256
#_C.Preprocess.resize_height = 256

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`