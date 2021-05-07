import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class Detector(object):

    def __init__(
        self, model_path, input_size,
        device='0', conf_thres=0.5, iou_thres=0.5
    ):

        self.weights = model_path
        self.imgsz = input_size
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.model = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._init_model()

    def _init_model(self):
        self.model = attempt_load(self.weights, map_location=self.device)
        self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

    def preprocess(self, img0):
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img0):

        img = self.preprocess(img0)
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

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
                    lbl = self.names[int(cls_id)]
                    x1, y1 = int(xyxy[0]), int(xyxy[1])
                    x2, y2 = int(xyxy[2]), int(xyxy[3])
                    label = f'{lbl} {conf:.2f}'
                    line = [x1, y1, x2, y2, lbl]
                    bboxes.append(line)
                    plot_one_box(xyxy, img0, label=label, color=colors(
                        int(cls_id), True), line_thickness=2)
        return img0, bboxes

    def generate_targets(self, imgs, tar_size):

        targets = []
        with torch.no_grad():
            for img_id in range(imgs.shape[0]):
                img = imgs[img_id].unsqueeze(0)
                pred = self.model(img, augment=False)[0]
                # Apply NMS
                pred = non_max_suppression(
                    pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

                for i, det in enumerate(pred):  # detections per image

                    gn = torch.tensor(tar_size)[[1, 0, 1, 0]]
                    if len(det):
                        # Rescale boxes from img_size to img0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], tar_size).round()

                        for value in reversed(det):
                            xyxy, conf, cls_id = value[:4], value[4], value[5]
                            logits = value[-len(self.names):].tolist()
                            xywh = (xyxy2xywh(torch.tensor(xyxy.cpu()).view(1, 4)
                                              ) / gn).view(-1).tolist()  # normalized xywh
                            line = [img_id, int(cls_id)]
                            line.extend(xywh)
                            line.extend(logits)
                            targets.append(line)
        return torch.tensor(np.array(targets), dtyp)


if __name__ == '__main__':

    img0 = cv2.imread('data/images/bus.jpg')
    det = Detector(model_path='weights/yolov5l.pt',
                   input_size=640, conf_thres=0.2)
    result, bboxes = det.detect(img0.copy())
    # cv2.imshow('result', result)
    for x1, y1, x2, y2, lbl in bboxes:
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.putText(img0, lbl, (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('resultw', img0)
    cv2.waitKey(0)
