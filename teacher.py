import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from models.experimental import attempt_load
from models.yolo import Model


class TeacherModel(object):
    def __init__(self, opt):
        self._init_model(opt.teacher)

    def _init_model(self, weights):
        pretrained = weights.endswith('.pt')
        if pretrained:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
            model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get(
                'anchors')).to(device)  # create
            exclude = ['anchor'] if (opt.cfg or hyp.get(
                'anchors')) and not opt.resume else []  # exclude keys
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(
                state_dict, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(state_dict, strict=False)  # load
            logger.info('Transferred %g/%g items from %s' %
                        (len(state_dict), len(model.state_dict()), weights))  # report
        else:
            model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get(
                'anchors')).to(device)  # create
        with torch_distributed_zero_first(rank):
            check_dataset(data_dict)  # check

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--teacher', type=str,
                        default='weights/yolov5l.pt', help='teacher weights path')
    opt = parser.parse_args()
    teacher = TeacherModel(opt)
    