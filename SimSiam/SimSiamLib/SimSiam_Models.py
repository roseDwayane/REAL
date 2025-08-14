
import torch
import torch.nn as nn
import torch.nn.functional as F


def max_norm(module, max_value=1.0, axis=-2, epsilon=1.0e-8):
    if hasattr(module, 'weight'):
        w = module.weight.data
        norms = torch.norm(w, p=2, dim=axis, keepdim=True)
        desired = norms.clamp(0, max_value)
        w = w * (desired / (epsilon + norms))
        module.weight.data = w


def unit_norm(module, axis=-1, epsilon=1.0e-8, norm_value=1.0):
    if hasattr(module, 'weight'):
        w = module.weight.data
        norms = torch.norm(w, p=2, dim=axis, keepdim=True)
        w = w / (epsilon + norms) * norm_value
        module.weight.data = w


def zero_mean(module, axis=-1, epsilon=1.0e-8):
    if hasattr(module, 'weight'):
        w = module.weight.data
        means = torch.mean(w, dim=axis, keepdim=True)
        w = w - means
        module.weight.data = w


class EEGNet_PT452E_TC16_V3(nn.Module):
    """
    temporal_conv = 16
    spatial_conv = 32
    from EEGNet_V452, temporal_conv(padding='valid‘)
    # nn.Dropout2d(p=0.4, inplace=False)
    """
    def __init__(self, nb_classes=2,
                 kern_length=125,
                 fc_dim=512):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.fc_dim = fc_dim

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16,
                                                     kernel_size=(1, self.kern_length),
                                                     padding=(0, 0),
                                                     bias=False,
                                                     ),
                                           nn.BatchNorm2d(num_features=16))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32,
                                                    kernel_size=(30, 1),
                                                    stride=(30, 1),
                                                    padding=(0, 0),
                                                    bias=False,
                                                    groups=16),
                                          nn.BatchNorm2d(num_features=32),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             groups=32),
                                   nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=(1, 1),
                                             bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   # nn.Dropout2d(p=0.4, inplace=False)
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             groups=64),
                                   nn.Conv2d(in_channels=64, out_channels=128,
                                             kernel_size=(1, 1),
                                             bias=False),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   # nn.Dropout2d(p=0.4, inplace=False)
                                   )

        # 1x1卷积扩展维度至 dim (from MobileNet)
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=self.fc_dim,
                                               kernel_size=(1, 1), bias=False),
                                     nn.BatchNorm2d(num_features=self.fc_dim),
                                     nn.ELU(inplace=True)
                                     )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.fc_dim, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv1x1(x)

        x = self.gap(x)
        x = x.view(-1, self.fc_dim)

        x = self.fc(x)

        return x

    def __str__(self):
        return 'EEGNet_V453(%d)' % (self.fc_dim, )


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder#num_classes=dim, zero_init_residual=True)
        # print(self.encoder.fc.weight.shape)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]

        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(pred_dim, dim))  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()
