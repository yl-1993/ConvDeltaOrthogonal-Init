import torch.nn as nn

import sys
sys.path.append("../")
import _ext.nn as enn


__all__ = ['van32', 'van128', 'van512', 'van2048',
           'van4096', 'van8192']


class Vanilla(nn.Module):

    def __init__(self, base, c, num_classes=10, conv_init='conv_delta_orthogonal'):
        super(Vanilla, self).__init__()
        self.init_supported = ['conv_delta_orthogonal', 'kaiming_normal']
        if conv_init in self.init_supported:
            self.conv_init = conv_init
        else:
            print('{} is not supported'.format(conv_init))
            self.conv_init = 'kaiming_normal'
        print('initialize conv by {}'.format(conv_init))
        self.base = base
        self.fc = nn.Linear(c, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if self.conv_init == self.init_supported[0]:
                    enn.init.conv_delta_orthogonal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def make_layers(depth):
    assert isinstance(depth, int)
    c = 256 if depth <= 256 else 128
    layers = []
    in_channels = 3
    for stride in [1, 2, 2]:
        conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1, stride=stride)
        layers += [conv2d, nn.Tanh()]
        in_channels = c
    for _ in range(depth):
        conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1)
        layers += [conv2d, nn.Tanh()]
    layers += [nn.AvgPool2d(8)] # For mnist is 7
    return nn.Sequential(*layers), c


def van32(**kwargs):
    """Constructs a 32 layers vanilla model.
    """
    model = Vanilla(*make_layers(32), **kwargs)
    return model


def van128(**kwargs):
    """Constructs a 128 layers vanilla model.
    """
    model = Vanilla(*make_layers(128), **kwargs)
    return model


def van512(**kwargs):
    """Constructs a 512 layers vanilla model.
    """
    model = Vanilla(*make_layers(512), **kwargs)
    return model


def van2048(**kwargs):
    """Constructs a 2048 layers vanilla model.
    """
    model = Vanilla(*make_layers(2048), **kwargs)
    return model


def van4096(**kwargs):
    """Constructs a 4096 layers vanilla model.
    """
    model = Vanilla(*make_layers(4096), **kwargs)
    return model


def van8192(**kwargs):
    """Constructs a 8192 layers vanilla model.
    """
    model = Vanilla(*make_layers(8192), **kwargs)
    return model
