from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
import numpy as np
import torch
from utils.torch_utils import select_device

class TeacherModel(object):
    def __init__(self, conf_thres=0.5, iou_thres=0.3, imgsz=640, training=False):
        self.model = None
        self.device = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.training = training

    def init_model(self, weights, device, nc):
        device = select_device(device)
        t_model = torch.load(weights, map_location=torch.device('cpu'))
        if t_model.get("model", None) is not None:
            t_model = t_model["model"]
        t_model.to(device)
        t_model.float()
        self.model = t_model
        self.device = device

        if self.training:
            self.model.train()
        else:
            self.model.eval()
        self.stride = int(self.model.stride.max())
        self.nc = nc

    def generate_batch_targets(self, imgs, tar_size=[640, 640]):
        targets = []
        with torch.no_grad():
            if self.training:
                preds = self.model(imgs)
            else:
                preds = self.model(imgs)[0]
        if not self.training:

            for img_id in range(imgs.shape[0]):

                pred = preds[img_id:img_id+1]
                pred = non_max_suppression(
                    pred, self.conf_thres, self.iou_thres, distill=True, agnostic=False)

                for det in pred:  # detections per image
                    gn = torch.tensor(tar_size)[[1, 0, 1, 0]]
                    if len(det):
                        # Rescale boxes from img_size to img0 size
                        det[:, :4] = scale_coords(
                            imgs[img_id].unsqueeze(0).shape[2:], det[:, :4], tar_size).round()

                        for value in reversed(det):
                            xyxy, cls_id = value[:4], value[5]
                            logits = value[-self.nc:].logit().tolist()
                            xywh = (xyxy2xywh(torch.tensor(xyxy.cpu()).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            line = [img_id, int(cls_id)]
                            line.extend(xywh)
                            line.extend(logits)
                            targets.append(line)

            return torch.tensor(np.array(targets), dtype=torch.float32), None
        else:
            return [], preds


if __name__ == '__main__':

    teacher = TeacherModel(conf_thres=0.0001)

    teacher.init_model('weights/yolov5m-voc.pt', select_device('0'), 2, 20)

    # img0 = cv2.imread('../xingren.jpg')
    # img0, bboxes = teacher.predict(img0)
    # cv2.imshow('winname', img0)
    # cv2.waitKey(0)

    imgs = torch.rand((2, 3, 640, 640)).to(teacher.device)
    targets = teacher.generate_batch_targets(imgs)
