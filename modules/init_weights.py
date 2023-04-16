import torch
import torch.nn as nn
from torch.nn import init


class init_weights(object):
    def __init__(self, net, init_type='normal'):
        self.net = net
        self.init_type = init_type

        if self.init_type == 'normal':
            self.net.apply(self.weights_init_normal)
        elif self.init_type == 'uniform':
            self.net.apply(self.weights_init_uniform)
        elif self.init_type == 'xavier':
            self.net.apply(self.weights_init_xavier)
        elif self.init_type == 'kaiming':
            self.net.apply(self.weights_init_kaiming)
        elif self.init_type == 'orthogonal':
            self.net.apply(self.weights_init_orthogonal)
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % self.init_type)

    def weights_init_normal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.uniform(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            init.uniform(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)

    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight.data, gain=1)
        elif classname.find('Linear') != -1:
            init.xavier_normal_(m.weight.data, gain=1)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
