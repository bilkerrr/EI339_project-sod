import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward())
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv3_3
        for k in range(15):
            x = self.vgg[k](x)
        conv3_3 = x
        module = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        conv3_3 = module(conv3_3)
        newNorm = L2Norm(256, 20)
        conv3_3 = newNorm(conv3_3)

        # apply vgg up to conv4_3
        for k in range(15, 22):
            x = self.vgg[k](x)
        conv4_3 = x
        module = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        conv4_3 = module(conv4_3)
        conv4_3 = self.L2Norm(conv4_3)

        # apply vgg up to conv5_3
        for k in range(22, 29):
            x = self.vgg[k](x)

        # apply vgg up to fc7
        for k in range(29, len(self.vgg)):
            x = self.vgg[k](x)
        conv7 = x
        module = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        conv7 = module(conv7)
        newNorm = L2Norm(1024, 20)
        conv7 = newNorm(conv7)

        cnt = 0
        global m1, m2, m3, conv8_2, conv9_2, conv10_2, conv11_2
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=False)
            if k % 2 == 1:
                cnt += 1
                if cnt == 1:
                    conv8_2_a = x
                   
                    module = nn.ConvTranspose2d(512, 256, kernel_size=2, padding=2, stride=2)
                    conv8_2 = module(conv8_2_a)
                    module = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                    conv8_2 = module(conv8_2)
                    module = nn.ReLU(inplace=False)
                    conv8_2 = module(conv8_2)
                    module = nn.ConvTranspose2d(256, 256, kernel_size=2, padding=2, stride=2)
                    conv8_2 = module(conv8_2)
                    module = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                    conv8_2 = module(conv8_2)
                    module = nn.ReLU(inplace=False)
                    conv8_2 = module(conv8_2)
                    module = nn.ConvTranspose2d(256, 256, kernel_size=2, padding=4, stride=3)
                    conv8_2 = module(conv8_2)
                    module = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                    conv8_2 = module(conv8_2)
                    newNorm = L2Norm(256, 10)
                    conv8_2 = newNorm(conv8_2)
                    # fuse
                    m1 = conv3_3 + conv8_2
                    module = nn.ReLU(inplace=False)
                    m1 = module(m1)
                    module = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                    m1 = module(m1)
                    module = nn.ReLU(inplace=False)
                    m1 = module(m1)
                elif cnt == 2:
                    conv9_2_a = x
                    module = nn.ConvTranspose2d(256, 256, kernel_size=2, padding=1, stride=2)
                    conv9_2 = module(conv9_2_a)
                    module = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                    conv9_2 = module(conv9_2)
                    module = nn.ReLU(inplace=False)
                    conv9_2 = module(conv9_2)
                    module = nn.ConvTranspose2d(256, 256, kernel_size=2, padding=1, stride=2)
                    conv9_2 = module(conv9_2)
                    module = nn.Conv2d(256, 256, kernel_size=3, padding=1)
                    conv9_2 = module(conv9_2)
                    module = nn.ReLU(inplace=False)
                    conv9_2 = module(conv9_2)
                    module = nn.ConvTranspose2d(256, 512, kernel_size=3, padding=2, stride=3)
                    conv9_2 = module(conv9_2)
                    
                    module = nn.Conv2d(512, 512, kernel_size=3, padding=1)
                    conv9_2 = module(conv9_2)
                    newNorm = L2Norm(512,10)
                    conv9_2 = newNorm(conv9_2)
                    # fuse
                    m2 = conv4_3 + conv9_2
                    module = nn.ReLU(inplace=False)
                    m2 = module(m2)
                    module = nn.Conv2d(512, 512, kernel_size=3, padding=1)
                    m2 = module(m2)
                    module = nn.ReLU(inplace=False)
                    m2 = module(m2)
                elif cnt == 3:
                    conv10_2_a = x

                    module = nn.ConvTranspose2d(256, 512, kernel_size=2, padding=1, stride=2)
                    conv10_2 = module(conv10_2_a)
                    module = nn.Conv2d(512, 512, kernel_size=3, padding=1)
                    conv10_2 = module(conv10_2)
                    module = nn.ReLU(inplace=False)
                    conv10_2 = module(conv10_2)
                    module = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, stride=2)                            
                    conv10_2 = module(conv10_2)
                    module = nn.Conv2d(512, 512, kernel_size=3, padding=1)
                    conv10_2 = module(conv10_2)
                    module = nn.ReLU(inplace=False)
                    conv10_2 = module(conv10_2)
                    module = nn.ConvTranspose2d(512, 1024, kernel_size=3, padding=1, stride=3)
                    conv10_2 = module(conv10_2)
                    module = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
                    conv10_2 = module(conv10_2)
                    newNorm = L2Norm(1024, 10)
                    conv10_2 = newNorm(conv10_2)
                    # fuse
                    m3 = conv7 + conv10_2
                    module = nn.ReLU(inplace=False)
                    m3 = module(m3)
                    module = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
                    m3 = module(m3)
                    module = nn.ReLU(inplace=False)
                    m3 = module(m3)
                elif cnt == 4:
                    conv11_2 = x

        sources = [m1, m2, m3, conv8_2_a, conv9_2_a, conv10_2_a, conv11_2]

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=False), conv7, nn.ReLU(inplace=False)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [14, 21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
