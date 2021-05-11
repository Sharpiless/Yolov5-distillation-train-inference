from models.experimental import attempt_load
from models.yolo_distill import Model
from utils.torch_utils import select_device, intersect_dicts
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.datasets import letterbox
from utils.plots import colors, plot_one_box

import numpy as np
import torch


class TeacherModel(object):
    def __init__(self, conf_thres=0.5, iou_thres=0.3, imgsz=640):
        self.model = None
        self.device = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

    def init_model(self, weights, device, batch_size, nc, teacher_cfg):

        self.device = select_device(device, batch_size=batch_size)

        # load checkpoint
        ckpt = torch.load(weights, map_location=self.device)
        self.model = Model(teacher_cfg or ckpt['model'].yaml, ch=3, nc=nc).to(
            self.device)  # create
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, self.model.state_dict(), exclude=['anchor'])  # intersect
        self.model.load_state_dict(state_dict, strict=False)  # load
        self.model.eval()
        self.stride = int(self.model.stride.max())
        self.nc = nc

    def preprocess(self, img0):
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, img0):
        img = self.preprocess(img0)
        pred = self.model(img)[0]
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, distill=True, agnostic=False)

        bboxes = []
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0.shape).round()

                for value in reversed(det):
                    xyxy, conf, cls_id = value[:4], value[4], value[5]
                    xywh = (xyxy2xywh(torch.tensor(xyxy.cpu()).view(1, 4)
                                      ) / gn).view(-1).tolist()  # normalized xywh
                    lbl = int(cls_id)
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])
                    label = f'{lbl} {conf:.2f}'
                    line = [x1, y1, x2, y2, lbl]
                    logits = value[-self.nc:].tolist()
                    tmp = np.argmax(logits)
                    bboxes.append(line)
                    plot_one_box(xyxy, img0, label=label, color=colors(
                        int(cls_id), True), line_thickness=2)
        return img0, bboxes

    def generate_targets(self, imgs, tar_size=[640, 640]):
        targets = []
        with torch.no_grad():
            for img_id in range(imgs.shape[0]):
                img = imgs[img_id].unsqueeze(0)
                pred = self.model(img)[0]
                pred = non_max_suppression(
                    pred, self.conf_thres, self.iou_thres, distill=True, agnostic=False)

                for i, det in enumerate(pred):  # detections per image
                    gn = torch.tensor(tar_size)[[1, 0, 1, 0]]
                    if len(det):
                        # Rescale boxes from img_size to img0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], tar_size).round()

                        for value in reversed(det):
                            xyxy, conf, cls_id = value[:4], value[4], value[5]
                            logits = value[-self.nc:].tolist()
                            xywh = (xyxy2xywh(torch.tensor(xyxy.cpu()).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            line = [img_id, int(cls_id)]
                            line.extend(xywh)
                            line.extend(logits)
                            targets.append(line)

        return torch.tensor(np.array(targets), dtype=torch.float32)

    def generate_batch_targets(self, imgs, tar_size=[640, 640]):
        targets = []
        preds = self.model(imgs)[0]
        with torch.no_grad():
            for img_id in range(imgs.shape[0]):

                pred = preds[img_id]                
                pred = non_max_suppression(
                    pred, self.conf_thres, self.iou_thres, distill=True, agnostic=False)

                for i, det in enumerate(pred):  # detections per image
                    gn = torch.tensor(tar_size)[[1, 0, 1, 0]]
                    if len(det):
                        # Rescale boxes from img_size to img0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], tar_size).round()

                        for value in reversed(det):
                            xyxy, conf, cls_id = value[:4], value[4], value[5]
                            logits = value[-self.nc:].tolist()
                            xywh = (xyxy2xywh(torch.tensor(xyxy.cpu()).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            line = [img_id, int(cls_id)]
                            line.extend(xywh)
                            line.extend(logits)
                            targets.append(line)

        return torch.tensor(np.array(targets), dtype=torch.float32)


if __name__ == '__main__':

    import cv2
    from utils.torch_utils import select_device

    teacher = TeacherModel()

    teacher.init_model('weights/yolov5l.pt', '0', 1, 20, 'models/yolov5l.yaml')
    img0 = cv2.imread('data/images/bus.jpg')
    img0, bboxes = teacher.predict(img0)
    cv2.imshow('winname', img0)
    cv2.waitKey(0)
    # teacher.init_model('weights/yolov5l.pt', '0', 2, 20, 'models/yolov5l.yaml')
    # imgs = torch.rand((2, 3, 640, 640)).to(teacher.device)
    # targets = teacher.generate_batch_targets(imgs)
    