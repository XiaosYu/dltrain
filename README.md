# dltrain
这是一个 PyTorch 训练包，本项目提供了大量的高级训练 API 和低级接口，满足所有训练需求
<br/>
安装仅在Shell中使用pip安装即可
```bash
pip install dltrain
```
### example .1 在iris数据集上使用多层感知器进行分类
```python
from dltrain import TaskBuilder, SimpleTrainer

builder = TaskBuilder('iris')
builder.base.use_epoch(100).use_batch_size(8).use_device('cuda')
builder.model.use_mlp(4, 3)
builder.delineator.use_random_split(builder.dataset.use_iris())
builder.criterion.use_cross_entropy()
SimpleTrainer().run(builder.build())
```

### example .1.1 使用Accuracy对模型进行验证
```python
from dltrain import TaskBuilder, SimpleTrainer

builder = TaskBuilder('iris')
builder.base.use_epoch(100).use_batch_size(8).use_device('cuda')
builder.model.use_mlp(4, 3)
builder.delineator.use_random_split(builder.dataset.use_iris())
builder.criterion.use_cross_entropy()
builder.evaluation_handler.add_accuracy()
SimpleTrainer().run(builder.build())
```
ps: 如需指定只计算训练/测试则调用.add_accuracy()中内嵌role='train/eval'即可。如不需要输出最后的绘图则内嵌参数drawable=False即可。

### example .2 使用自建模型在mnist手写集数字上进行分类
```python
from dltrain import TaskBuilder, SimpleTrainer
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        def make_layer(i, o, k, s, p):
            return nn.Sequential(
                nn.Conv2d(i, o, k, s, p),
                nn.BatchNorm2d(o),
                nn.ReLU()
            )
        
        self.features = nn.Sequential(
            make_layer(1, 3, 7, 1, 0),
            make_layer(3, 16, 7, 1, 0),
            make_layer(16, 64, 7, 1, 0),
            make_layer(64, 128, 7, 1, 0)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1000),
            nn.Linear(1000, 10)
        )
        
    def forward(self, data):
        data = self.features(data)
        data = data.reshape(data.shape[0], -1)
        data = self.classifier(data)
        return data


builder = TaskBuilder('mnist')
builder.model.use_model(Model())
builder.base.use_device('cuda')
builder.optimizer.use_adam()
builder.criterion.use_cross_entropy()
builder.evaluation_handler.add_accuracy()
builder.delineator.use_train_eval(builder.dataset.use_mnist('./dataset', True),
                                  builder.dataset.use_mnist('./dataset', False))
SimpleTrainer().run(builder.build())
```
模型向导对象的use_model接口允许用户传入自己的模型

### Example .2.1 使用torchvision的自带模型
```python
from dltrain import TaskBuilder, SimpleTrainer

builder = TaskBuilder('mnist')
builder.model.use_pytorch_model('resnet18', num_classes=10)
builder.base.use_device('cuda')
builder.optimizer.use_adam()
builder.criterion.use_cross_entropy()
builder.evaluation_handler.add_accuracy()
builder.delineator.use_train_eval(builder.dataset.use_mnist('./dataset', True),
                                  builder.dataset.use_mnist('./dataset', False))
SimpleTrainer().run(builder.build())
```
正如所说的，模型向导提供了use_pytorch_model接口，允许用户直接调用torchvision.models下的所有模型并封装到一个名为PyTorchNativeCNN的类型下，该类型会让模型自适应所有的数据集输入输出
##### Suppose Model In TorchVision
```python
__Model__ = [
    googlenet, alexnet,

    resnet18, resnet34, resnet50, resnet101, resnet152,

    vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,

    vit_b_16, vit_h_14, vit_b_32, vit_l_16, vit_l_32,

    mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,

    efficientnet_v2_s, efficientnet_v2_l, efficientnet_v2_m, efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,

    densenet121, densenet161, densenet169, densenet201,

    regnet_x_8gf, regnet_x_1_6gf, regnet_y_8gf, regnet_y_400mf, regnet_y_128gf, regnet_y_1_6gf, regnet_x_3_2gf,
    regnet_x_16gf, regnet_x_32gf, regnet_x_400mf, regnet_y_800mf, regnet_x_800mf, regnet_y_3_2gf, regnet_y_16gf,
    regnet_y_32gf,

    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0,

    swin_b, swin_t, swin_s, swin_v2_b, swin_v2_t, swin_v2_s,

    mnasnet0_5, mnasnet1_0, mnasnet1_3, mnasnet0_75
]
```
### default about the arguments
| 参数名称             | 默认值                                                |
|------------------|----------------------------------------------------|
| optimizer        | Sgd(lr=0.01,momentum=0,dampening=0,weight_decay=0) |
| scheduler        | User-set                                           |
| criterion*       | None,Must be specified by the user                 |    
| model*           | None,Must be specified by the user                 |
| epoch            | 10                                                 |
| batch_size       | 16                                                 |
| seed             | 3407                                               |
| device           | cpu                                                |
| save_checkpoint  | False                                              |
| start_checkpoint | User-set                                           |
| delineator*      | None,Must be specified by the user                 |
| forward          | SimpleForward                                      |
| trainer          | SimpleTrainer                                      |
