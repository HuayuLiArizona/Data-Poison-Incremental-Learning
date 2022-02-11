from torch import nn
import math

class Initializer:
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                try:
                    m.bias.data.fill_(0)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.normal_(0, stdv)
                try:
                    m.bias.data.fill_(0)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)