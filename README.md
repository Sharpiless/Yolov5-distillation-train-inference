# 代码地址：

https://github.com/Sharpiless/Yolov5-distillation-train-inference

<a align="left" href="https://apps.apple.com/app/id1452689527" target="_blank">
<img width="800" src="https://user-images.githubusercontent.com/26833433/98699617-a1595a00-2377-11eb-8145-fc674eb9b1a7.jpg"></a>

![](https://github.com/Sharpiless/Yolov5-distillation-train-inference/blob/main/data/images/YOLOv5%E8%92%B8%E9%A6%8F.png)

# 蒸馏训练：

```bash
python train_distill.py --weights yolov5s.pt \
    --teacher weights/yolov5l_voc.pt --distill_ratio 0.001 \
    --teacher-cfg model/yolov5l.yaml --data data/voc.yaml \
    --epochs 30 --batch-size 16
```

# 训练参数:

> --weights：预训练模型

> --teacher：教师模型权重

> --distill-ratio：蒸馏损失权重

> --with-gt-loss：是否同时使用ground truth

> --full-output-loss：是否使用[《Object detection at 200 Frames Per Second》](https://arxiv.org/pdf/1805.06361.pdf)中的损失

这篇文章分别对这几个损失函数做出改进，具体思路为只有当teacher network的objectness value高时，才学习bounding box坐标和class probabilities。

![](https://github.com/Sharpiless/Yolov5-distillation-train-inference/blob/main/data/images/full_loss.png)

# 准备数据集：

默认会启用 data/voc.yaml 自动下载VOC数据集进行训练

或者手动运行 data/scripts/get_voc2007.sh 下载

如需修改成自己的数据集，则只需要修改yaml路径即可

# 实验结果：

数据集：

VOC2007（补充的无标签数据使用VOC2012）

GPU：2080Ti*1

Batch Size：16

Epoches：30

Baseline：Yolov5s

Teacher model：Yolov5l（mAP 0.5:0.95 = 0.541）


这里假设VOC2012中新增加的数据为无标签数据（2k张）。

| 原模型     | 教师模型    | VOC2007 | VOC2012 | mAP 0.5:0.95 |
|---------|---------|---------|---------|--------------|
| Yolov5s | 无       | 原始标签    | 不使用     | 0.487        |
| Yolov5s | Yolov5l | 蒸馏训练    | 不使用     | 0.449        |
| Yolov5s | Yolov5l | 蒸馏训练    | 蒸馏训练    | 暂无         |
| Yolov5s | Yolov5l | 原始标签    | 蒸馏训练    | 0.486        |

参数和细节正在完善，支持KL散度、L2 logits损失和Sigmoid蒸馏损失等

# 待做事项：

- [√] 修改logist输出作为蒸馏损失输入
- [√] 完善代码结构和相关参数设定
- [×] 查找为何蒸馏损失不起作用（或者收敛慢）的原因
- [×] 完善相关实验并测试精度
- [√] 修改dataloader加快训练速度
- [√] 修改teacher model的批量推理加快训练速度

# 可能存在的问题：

- 1.训练轮数太少没收敛，可能蒸馏训练收敛满最终结果高
- 2.教师模型是Yolov5l在VOC训练30轮得到的（mAP 0.5:0.95 = 0.541），质量比标注较差影响蒸馏训练的结果
- 3.可调整的参数还有很多（教师模型的检测、IOU阈值，蒸馏损失种类，蒸馏损失比率等）

# 我的公众号：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210310070958646.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

