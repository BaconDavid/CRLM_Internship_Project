from config import get_cfg_defaults
from main import shit as my_project

if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file("./Resnet10_local.yaml")

    print(cfg.LOSS.SelectiveLoss.SelectiveNetLoss.loss)
    cfg.TEST.batch_size = 10
    print(cfg.Scheduler.WarmupCosineScheduler.t_total)
    cfg.freeze()


    #print(cfg.TRAIN.scheduler_param)
    print(list(cfg.MODEL.Resnet10.block_inplanes))

