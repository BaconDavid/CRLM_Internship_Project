# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.DEVICE = 'cuda'


_C.DATA = CN()
# The path to the data directory
_C.DATA.Data_dir = None
_C.DATA.Train_dir = None
_C.DATA.Valid_dir = None
_C.DATA.Test_dir = None
_C.DATA.Data_mask_dir = None

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.lr = 0.001
# The all important scales for the stuff
_C.TRAIN.batch_size = 16
_C.TRAIN.num_epochs = 20
_C.TRAIN.optimizer = 'Adam'
_C.TRAIN.loss = 'CrossEntropy'
_C.TRAIN.data_aug = True
_C.TRAIN.Debug = False
_C.TRAIN.scheduler = False
_C.TRAIN.scheduler_name = 'WarmupCosineSchedule'
_C.TRAIN.drop_out = 0.5
_C.TRAIN.weight_decay = 0.001


_C.VALID = CN()
_C.VALID.batch_size = 16
_C.VALID.loss = 'CrossEntropy'
_C.VALID.data_aug = False

_C.LOSS = CN()
_C.LOSS.loss_mode = 'classification'
_C.LOSS.loss_name = 'CrossEntropy'
_C.LOSS.loss_lm = 32
_C.LOSS.loss_coverage = 0.5
_C.LOSS.alpha = 0.5


_C.TEST = CN()
_C.TEST.batch_size = 16
_C.TEST.loss = 'CrossEntropy'


_C.DATASET = CN()
_C.DATASET.WeightedRandomSampler = False
_C.DATASET.num_classes = 2
_C.DATASET.num_channels = 3
_C.DATASET.mask = True


_C.MODEL = CN()
_C.MODEL.name = 'Resnet10'
_C.MODEL.pretrained = False
_C.MODEL.freeze_layers = None
_C.MODEL.pretrained_path = None
_C.MODEL.num_class = 2
_C.MODEL.num_in_channels = 1
_C.MODEL.drop_out = 0.5
_C.MODEL.num_out_channels = 1
_C.MODEL.weight_decay = 0.0001
_C.MODEL.Drop_block = True
_C.MODEL.block_size = 5
_C.MODEL.drop_prob = 0.9
_C.MODEL.v2 = False
_C.MODEL.task = 'classification'
_C.MODEL.feature_model = 'Resnet10'
_C.MODEL.feature_dims = 512


_C.LOG = CN()
_C.LOG.log_dir = None
_C.LOG.log_file = None

_C.Preprocess = CN()
_C.Preprocess.resize_shape = None
#_C.Preprocess.resize_width = 256
#_C.Preprocess.resize_height = 256
_C.Preprocess.normalize = 'NormalizeIntensity()'
_C.Preprocess.padding_size = None

_C.LABEL = CN()
_C.LABEL.label_name = None


_C.visual_im = CN()
_C.visual_im.visual_im = True
_C.visual_im.visual_out_path = None
_C.visual_im.slice = 5

_C.WEIGHT = CN()
_C.WEIGHT.weight_dir = None

_C.SAVE = CN()
_C.SAVE.save_dir = None
_C.SAVE.save_name = None
_C.SAVE.fold = None

_C.Augmentation = CN()
_C.Augmentation.RandZoom = 'True'
_C.Augmentation.RandRotate = 'True'
_C.Augmentation.RandFlip = 'True'
_C.Augmentation.Resize = (256,256,64)
_C.Augmentation.NormalizeIntensity = True
_C.Augmentation.ToTensor = True
_C.Augmentation.CenterSpatialCrop = (256,256,64)
_C.Augmentation.SpatialPad = (256,256,64)
_C.Augmentation.RandSpatialCrop = (256,256,64)


_C.WarmupCosineSchedule = CN()
_C.WarmupCosineSchedule.warmup_steps = 500
_C.WarmupCosineSchedule.t_total = 1000




def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`