import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_layers import Stage


class DeepLabLargeFOVBN(nn.Module):
    def __init__(self, in_dim, out_dim, weights='DeepLab', *args, **kwargs):
        super(DeepLabLargeFOVBN, self).__init__(*args, **kwargs)
        self.stages = []
        layers = []

        stage = [
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # TensorFlow 'SAME' behavior
            nn.MaxPool2d(3, stride=2)
        ]
        layers += stage
        self.stages.append(Stage(64, stage))

        stage = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # TensorFlow 'SAME' behavior
            nn.MaxPool2d(3, stride=2)
        ]
        layers += stage
        self.stages.append(Stage(128, stage))

        stage = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # TensorFlow 'SAME' behavior
            nn.MaxPool2d(3, stride=2)
        ]
        layers += stage
        self.stages.append(Stage(256, stage))

        stage = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding=1)
        ]
        layers += stage
        self.stages.append(Stage(512, stage))

        stage = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding=1),
            # must use count_include_pad=False to make sure result is same as TF
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        ]
        layers += stage
        self.stages.append(Stage(512, stage))
        self.stages = nn.ModuleList(self.stages)

        self.features = nn.Sequential(*layers)

        head = [
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(1024, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, out_dim, kernel_size=1)
        ]
        self.head = nn.Sequential(*head)

        self.weights = weights
        self.init_weights()

    def forward(self, x):
        N, C, H, W = x.size()
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return x

    def init_weights(self):
        for layer in self.head.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        if self.weights == 'DeepLab':
            pretrained_dict = torch.load('weights/vgg_deeplab_lfov/tf_deeplab.pth')
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'head.7' not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)
        elif self.weights == 'Seg':
            pretrained_dict = torch.load('weights/nyu_v2/tf_finetune_seg.pth')
            self.load_state_dict(pretrained_dict)
        elif self.weights == 'Normal':
            pretrained_dict = torch.load('weights/nyu_v2/tf_finetune_normal.pth')
            self.load_state_dict(pretrained_dict)
        elif self.weights == '':
            pass
        else:
            raise NotImplementedError


if __name__ == "__main__":
    net = DeepLabLargeFOVBN(3, 10)
    in_ten = torch.randn(1, 3, 321, 321)
    out = net(in_ten)
    print(out.size())
    print(net.stages[1])
