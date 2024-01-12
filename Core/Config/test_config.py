from config import get_cfg_defaults
from main import shit as my_project

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./Resnet10_local.yaml")
    cfg.freeze()
    #print(cfg.TRAIN.scheduler_param)
    print(type(cfg.SAVE.fold))

