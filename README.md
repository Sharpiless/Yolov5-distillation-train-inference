# 代码地址：

https://github.com/Sharpiless/Yolov5-distillation-train-inference

<a align="left" href="https://apps.apple.com/app/id1452689527" target="_blank">
<img width="800" src="https://user-images.githubusercontent.com/26833433/98699617-a1595a00-2377-11eb-8145-fc674eb9b1a7.jpg"></a>

# 蒸馏训练：

```bash
python train_distill.py --weights yolov5s.pt \
    --teacher weights/yolov5l_voc.pt --distill_ratio 0.001 \
    -- teacher-cfg model/yolov5l.yaml --data data/voc.yaml \
    --epochs 30 --batch-size 16
```

# 训练参数:

> --weights：预训练模型

> --teacher：教师模型权重

> --distill-ratio：蒸馏损失权重

# 准备数据集：

默认会启用 data/voc.yaml 自动下载VOC数据集进行训练

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
| Yolov5s | Yolov5l | 生成标签    | 不使用     | 0.449        |
| Yolov5s | Yolov5l | 生成标签    | 生成标签    | 暂无         |
| Yolov5s | Yolov5l | 原始标签    | 生成标签    | 0.486        |

参数和细节正在完善

# 待做事项：

- [√] 修改logist输出作为蒸馏损失输入
- [√] 完善代码结构和相关参数设定
- [×] 查找为何蒸馏损失不起作用（或者收敛慢）的原因
- [×] 完善相关实验并测试精度
- [√] 修改dataloader加快训练速度
- [×] 修改teacher model的批量推理加快训练速度

# 我的公众号：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210310070958646.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

