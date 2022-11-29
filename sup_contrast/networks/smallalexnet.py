r"""
Adapted from https://github.com/HobbitLong/CMC/blob/f25c37e49196a1fe7dc5f7b559ed43c6fce55f70/models/alexnet.py
"""

import torch.nn as nn


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


class SmallAlexNet(nn.Module):
    def __init__(self, in_channel=3, feat_dim=128, no_final_proj=False):
        super(SmallAlexNet, self).__init__()

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # fc6
        blocks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 7 * 7, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(4096, feat_dim) if not no_final_proj else nn.Identity(),
            #L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, *, layer_index=-1, average=True):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:
            x = layer(x)

        # NEW: spatial averaging
        if average:
            if x.ndim == 4:
                x = x.mean(dim=-1)
            if x.ndim == 3:
                x = x.mean(dim=-1)

        return x



class SupConSmallAlexNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128):
        super(SupConSmallAlexNet, self).__init__()
        model_fun, dim_in = SmallAlexNet, 128
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        #feat = self.head(feat)
        feat = nn.functional.normalize(self.head(feat), dim=1)
        return feat


class SupCESmallAlexNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=676):
        super(SupCESmallAlexNet, self).__init__()
        model_fun, dim_in = SmallAlexNet(no_final_proj=True), 4096
        self.encoder = model_fun
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))



class SmallAlexnetLinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, layer_index=5, num_classes=10):
        super(SmallAlexnetLinearClassifier, self).__init__()
        assert layer_index in [5,6], "Not implemented for other layer indices"
        feat_dim = 4096
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
