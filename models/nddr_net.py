import torch
import torch.nn as nn
import torch.nn.functional as F


class NDDR(nn.Module):
    def __init__(self, out_channels, init_weights=[0.9, 0.1], init_method='constant', activation='relu',
                 batch_norm=True, bn_before_relu=False):
        super(NDDR, self).__init__()
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        if init_method == 'constant':
            self.conv1.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[0],
                torch.eye(out_channels) * init_weights[1]
            ], dim=1).view(out_channels, -1, 1, 1))
            self.conv1.bias.data.fill_(0)
            self.conv2.weight = nn.Parameter(torch.cat([
                torch.eye(out_channels) * init_weights[1],
                torch.eye(out_channels) * init_weights[0]
            ], dim=1).view(out_channels, -1, 1, 1))
            self.conv2.bias.data.fill_(0)
        elif init_method == 'xavier':
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)
        elif init_method == 'kaiming':
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        else:
            raise NotImplementedError

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

        self.batch_norm = batch_norm
        self.bn_before_relu = bn_before_relu
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.05)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.05)

    def forward(self, feature1, feature2):
        x = torch.cat([feature1, feature2], 1)
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        if self.batch_norm and self.bn_before_relu:
            out1 = self.bn1(out1)
            out2 = self.bn2(out2)
        if self.activation:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        if self.batch_norm and not self.bn_before_relu:
            out1 = self.bn1(out1)
            out2 = self.bn2(out2)
        return out1, out2


class NDDRNet(nn.Module):
    def __init__(self, net1, net2, init_weights=[0.9, 0.1], init_method='constant', activation='relu', batch_norm=True, shortcut=False, bn_before_relu=False):
        super(NDDRNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        assert len(net1.stages) == len(net2.stages)
        
        self.num_stages = len(net1.stages)
        nddrs = []
        total_channels = 0
        for stage_id in range(self.num_stages):
            out_channels = net1.stages[stage_id].out_channels
            assert out_channels == net2.stages[stage_id].out_channels
            total_channels += out_channels
            nddr = NDDR(out_channels, init_weights, init_method, activation, batch_norm, bn_before_relu)
            nddrs.append(nddr)
        nddrs = nn.ModuleList(nddrs)

        self.shortcut = shortcut
        final_conv = None
        if shortcut:
            print("Using shortcut")
            conv = nn.Conv2d(total_channels, net1.stages[-1].out_channels, kernel_size=1)
            bn = nn.BatchNorm2d(net1.stages[-1].out_channels, momentum=0.05)
            if bn_before_relu:
                print("Using bn before relu")
                final_conv = [conv, bn, nn.ReLU()]
            else:
                final_conv = [conv, nn.ReLU(), bn]
            final_conv = nn.Sequential(*final_conv)

        self.nddrs = nn.ModuleDict({
            'nddrs': nddrs,
            'shortcut': final_conv,
        })
    
    def forward(self, x):
        N, C, H, W = x.size()
        y = x.clone()
        xs = []
        ys = []
        for stage_id in range(self.num_stages):
            x = self.net1.stages[stage_id](x)
            y = self.net2.stages[stage_id](y)
            x, y = self.nddrs['nddrs'][stage_id](x, y)
            if self.shortcut:
                xs.append(x)
                ys.append(y)
        if self.shortcut:
            _, _, h, w = xs[-1].size()
            x = torch.cat([F.interpolate(_x, (h, w), mode='bilinear', align_corners=True) for _x in xs[:-1]]+[xs[-1]], dim=1)
            y = torch.cat([F.interpolate(_y, (h, w), mode='bilinear', align_corners=True) for _y in ys[:-1]]+[ys[-1]], dim=1)
            x = self.nddrs['shortcut'](x)
            y = self.nddrs['shortcut'](y)
        x = self.net1.head(x)
        y = self.net2.head(y)
        
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        y = F.interpolate(y, (H, W), mode='bilinear', align_corners=True)
        return x, y
    

if __name__ == '__main__':
    from vgg16_lfov import DeepLabLargeFOV
    net1 = DeepLabLargeFOV(3, 40, weights='')
    net2 = DeepLabLargeFOV(3, 3, weights='')
    net = NDDRNet(net1, net2)
    if torch.cuda.is_available():
        net.cuda()
    in_ten = torch.randn(1, 3, 321, 321)
    if torch.cuda.is_available():
        in_ten = in_ten.cuda()
    out1, out2 = net(in_ten)
    print(dict(net.named_parameters()).keys())
    print(out1.size(), out2.size())
