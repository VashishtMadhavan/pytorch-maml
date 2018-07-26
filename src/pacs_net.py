import numpy as np
import random
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from layers import *
alexnet_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'

class PACSNet(nn.Module):
    '''
    The base model for few-shot learning on PACS; Same as AlexNet
    '''

    def __init__(self, num_classes, loss_fn, num_in_channels=3):
        super(PACSNet, self).__init__()
        # Define the network
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),

            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),

            ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('relu4', nn.ReLU(inplace=True)),

            ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('drop6', nn.Dropout()),
            ('fc6', nn.Linear(256 * 6 * 6, 4096)),
            ('relu6', nn.ReLU(inplace=True)),
            ('drop7', nn.Dropout()),
            ('fc7', nn.Linear(4096, 4096)),
            ('relu7', nn.ReLU(inplace=True)),
            ('fc8', nn.Linear(4096, num_classes)),
        ]))
        
        # Define loss function
        self.loss_fn = loss_fn

        # Initialize weights to pretrained imagenet weights
        #self._init_weights()
        pretrain_dict = model_zoo.load_url(alexnet_url)
        import pdb; pdb.set_trace()
        self.load_state_dict(pretrain_dict)

    def forward(self, x, weights=None):
        ''' Define what happens to data in the net '''
        if weights == None:
            x = self.features(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
        else:
            # conv1
            x = conv2d(x, weights['features.conv1.weight'], weights['features.conv1.bias'])
            x = relu(x)
            x = maxpool(x, kernel_size=3, stride=2)
            # conv2
            x = conv2d(x, weights['features.conv2.weight'], weights['features.conv2.bias'])
            x = relu(x)
            x = maxpool(x, kernel_size=3, stride=2)
            # conv3
            x = conv2d(x, weights['features.conv3.weight'], weights['features.conv3.bias'])
            x = relu(x)
            # conv4
            x = conv2d(x, weights['features.conv4.weight'], weights['features.conv4.bias'])
            x = relu(x)
            # conv5
            x = conv2d(x, weights['features.conv5.weight'], weights['features.conv5.bias'])
            x = relu(x)
            x = maxpool(x, kernel_size=3, stride=2)

            # classifier trunk
            x = x.view(x.size(0), 256 * 6 * 6)
            x = dropout(x, p=0.5)
            x = linear(x, weights['classifier.fc6.weight'], weights['classifier.fc6.bias'])
            x = relu(x)
            x = dropout(x, p=0.5)
            x = linear(x, weights['classifier.fc7.weight'], weights['classifier.fc7.bias'])
            x = relu(x)
            x = linear(x, weights['classifier.fc8.weight'], weights['classifier.fc8.bias'])
        return x

    def net_forward(self, x, weights=None):
        return self.forward(x, weights)
    
    def _init_weights(self):
        ''' Set weights to Gaussian, biases to zero '''
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        #print('init weights')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                #m.bias.data.zero_() + 1
                m.bias.data = torch.ones(m.bias.data.size())
    
    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        # TODO: breaks if nets are not identical
        # TODO: won't copy buffers, e.g. for batch norm
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
