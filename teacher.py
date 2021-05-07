from models.experimental import attempt_load
from models.yolo_distill import Model
from utils.torch_utils import select_device, intersect_dicts
from utils.general import non_max_suppression
from utils.datasets import letterbox
from utils.plots import colors, plot_one_box

import torch


class TeacherModel(object):
    def __init__(self, conf_thres=0.5, iou_thres=0.3):
        self.model = None
        self.device = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def init_model(self, weights, device, batch_size, nc, teacher_cfg):

        self.device = select_device(device, batch_size=batch_size)
        cuda = self.device.type != 'cpu'

        pretrained = weights.endswith('.pt')
        if pretrained:
            # load checkpoint
            ckpt = torch.load(weights, map_location=self.device)
            self.model = Model(teacher_cfg or ckpt['model'].yaml, ch=3, nc=nc).to(
                self.device)  # create
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = intersect_dicts(
                state_dict, self.model.state_dict(), exclude=['anchor'])  # intersect
            self.model.load_state_dict(state_dict, strict=False)  # load
            self.model.eval()

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

    def predict(self, img0):

        pred = self.model(img_tensor)[0]
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


if __name__ == '__main__':

    from utils.torch_utils import select_device

    teacher = TeacherModel()

    teacher.init_model('weights/yolov5l.pt', '0', 1, 20, 'models/yolov5l.yaml')
    img0 = cv2.imread('data/images/')
    teacher.predict(img0)
