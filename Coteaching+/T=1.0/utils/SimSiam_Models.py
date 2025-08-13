import torch
import torch.nn as nn


def max_norm(module, max_value=1.0, axis=-2, epsilon=1.0e-8):
    if hasattr(module, 'weight'):
        w = module.weight.data
        norms = torch.norm(w, p=2, dim=axis, keepdim=True)
        desired = norms.clamp(0, max_value)
        w = w * (desired / (epsilon + norms))
        module.weight.data = w


class EEGNetResidualSeparableConv(nn.Module):
    """
    from EEGNet_V452
    """
    def __init__(self, in_channels=32,
                 out_channels=32,
                 kernel_size=(1, 15),
                 padding=(0, 0)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.separable_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                                                      kernel_size=self.kernel_size,
                                                      padding=self.padding,
                                                      bias=False, groups=self.in_channels),

                                            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                                      kernel_size=(1, 1),
                                                      bias=False),
                                            nn.BatchNorm2d(num_features=self.out_channels))

        self.residual = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                                kernel_size=(1, 1),
                                                bias=False),
                                      nn.BatchNorm2d(num_features=self.out_channels))

        self.activate = nn.ELU(inplace=True)

    def forward(self, x):
        if self.in_channels == self.out_channels:
            x = self.separable_conv(x) + x
        else:
            x = self.separable_conv(x) + self.residual(x)

        x = self.activate(x)

        return x


class EEGNet(nn.Module):
    """
    from EEGNet_V452
    """
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=1):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, self.kern_length // 2),
                                                     bias=False),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), stride=(30, 1), padding=(0, 0),
                                                    bias=False, groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=16),
                                   nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=(1, 1),
                                             bias=False),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(channels, 2), stride=(channels, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=32),
                                   nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Linear(64, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        # print(x.shape)
        x = self.spatial_conv(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)

        x = self.gap(x)
        # print(x.shape)
        x = x.view(-1, 64)
        # torch.flatten(x, 1)
        # print(x.shape)
        # print(self.fc.weight.shape)
        # x = self.dropout(x)
        x = self.fc(x)

        return x


class EEGNet_V452(nn.Module):
    """
    from EEGNet_V452
    """
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=4, l2_weight=0.001):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels
        self.l2_weight = l2_weight

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, self.kern_length // 2),
                                                     bias=False,
                                                     ),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), padding=(0, 0),
                                                    bias=False,
                                                    groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=16),
                                   nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=32),
                                   nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Linear(64, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)

        return x


class EEGNet_V452E(nn.Module):
    """
    1x1卷积扩展维度至 dim (from MobileNet)
    self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=dim,
                                           kernel_size=(1, 1), bias=False),
                                 nn.BatchNorm2d(num_features=dim))

    from EEGNet_V452
    """
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=1, dim=512):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels
        self.dim = dim

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, self.kern_length // 2),
                                                     bias=False,
                                                     ),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), padding=(0, 0),
                                                    bias=False,
                                                    groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=16),
                                   nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=32),
                                   nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        # 1x1卷积扩展维度至 dim (from MobileNet)
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=dim,
                                               kernel_size=(1, 1), bias=False),
                                     nn.BatchNorm2d(num_features=dim))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Linear(dim, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.gap(x)

        x = self.conv1x1(x)

        x = x.view(-1, self.dim)
        x = self.fc(x)

        return x


class EEGNet_V452_512(nn.Module):
    """
    from EEGNet_V452
    """
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=4, l2_weight=0.001):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels
        self.l2_weight = l2_weight

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, self.kern_length // 2),
                                                     bias=False,
                                                     ),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), padding=(0, 0),
                                                    bias=False,
                                                    groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=16),
                                   nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=32),
                                   nn.Conv2d(in_channels=32, out_channels=512,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=512),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Linear(512, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.gap(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x


class EEGNet_V453(nn.Module):
    """
    from EEGNet_V452, temporal_conv(padding='valid)
    """
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=4, l2_weight=0.001, fc_dim=256):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels
        self.l2_weight = l2_weight
        self.fc_dim = fc_dim

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, 0),
                                                     bias=False,
                                                     ),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), padding=(0, 0),
                                                    bias=False,
                                                    groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=16),
                                   nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=32),
                                   nn.Conv2d(in_channels=32, out_channels=self.fc_dim,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=self.fc_dim),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.fc_dim, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        # print(x.shape)
        x = self.spatial_conv(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)

        x = self.gap(x)
        # print(x.shape)
        x = x.view(-1, self.fc_dim)
        # torch.flatten(x, 1)
        x = self.fc(x)

        return x


class EEGNet_V452_Valid(nn.Module):
    """
    from EEGNet_V452
    self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                       kernel_size=(1, self.kern_length), padding=(0, 0),
                                       bias=False),
    """
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=1):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, 0),
                                                     bias=False),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), stride=(30, 1), padding=(0, 0),
                                                    bias=False, groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=16),
                                   nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=(1, 1),
                                             bias=False),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(channels, 2), stride=(channels, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=32),
                                   nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Linear(64, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)

        # max_norm constrains
        # # w0 = self.spatial_conv[0].weight.data.flatten()
        # max_norm(self.spatial_conv[0], max_value=1.0, axis=-2)
        # # w1 = self.spatial_conv[0].weight.data.flatten()
        # # print(w1-w0)

        x = self.spatial_conv(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)

        x = self.gap(x)
        # print(x.shape)
        x = x.view(-1, 64)
        # torch.flatten(x, 1)
        # print(x.shape)
        # print(self.fc.weight.shape)
        # x = self.dropout(x)
        x = self.fc(x)

        return x


class EEGNet_V452_Valid_Residual(nn.Module):
    """
    from EEGNet_V452
    self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                       kernel_size=(1, self.kern_length), padding=(0, 0),
                                       bias=False),
    """
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=1):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, 0),
                                                     bias=False),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), stride=(30, 1), padding=(0, 0),
                                                    bias=False, groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(EEGNetResidualSeparableConv(in_channels=16, out_channels=32,
                                                               kernel_size=(1, 15), padding=(0, 7)),
                                   nn.MaxPool2d(kernel_size=(channels, 2), stride=(channels, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(EEGNetResidualSeparableConv(in_channels=32, out_channels=64,
                                                               kernel_size=(1, 15), padding=(0, 7)),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        # self.conv3 = nn.Sequential(EEGNetResidualSeparableConv(in_channels=64, out_channels=128,
        #                                                        kernel_size=(1, 15), padding=(0, 7)))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)

        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x


class EEGNet_V452_Valid_Residual_512(nn.Module):
    """
    from EEGNet_V452
    self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                       kernel_size=(1, self.kern_length), padding=(0, 0),
                                       bias=False),
    """
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=1):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, 0),
                                                     bias=False),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), stride=(30, 1), padding=(0, 0),
                                                    bias=False, groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(EEGNetResidualSeparableConv(in_channels=16, out_channels=32,
                                                               kernel_size=(1, 15), padding=(0, 7)),
                                   nn.MaxPool2d(kernel_size=(channels, 2), stride=(channels, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(EEGNetResidualSeparableConv(in_channels=32, out_channels=64,
                                                               kernel_size=(1, 15), padding=(0, 7)),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        # self.conv3 = nn.Sequential(EEGNetResidualSeparableConv(in_channels=64, out_channels=128,
        #                                                        kernel_size=(1, 15), padding=(0, 7)))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.gap(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x


class EEGNet_V452_128(nn.Module):
    def __init__(self, nb_classes=2, kern_length=125, samples=251, channels=4, l2_weight=0.001):
        super().__init__()
        self.nb_classes = nb_classes
        self.kern_length = kern_length
        self.samples = samples
        self.channels = channels
        self.l2_weight = l2_weight

        """
         Define layers that would be used in your model, including conv, pooling, and fully, etc.
        """
        self.temporal_conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8,
                                                     kernel_size=(1, self.kern_length), padding=(0, self.kern_length // 2),
                                                     bias=False,
                                                     ),
                                           nn.BatchNorm2d(num_features=8))

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
                                                    kernel_size=(30, 1), padding=(0, 0),
                                                    bias=False,
                                                    groups=8),
                                          nn.BatchNorm2d(num_features=16),
                                          nn.ELU(inplace=True),
                                          nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=16),
                                   nn.Conv2d(in_channels=16, out_channels=32,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=32),
                                   nn.Conv2d(in_channels=32, out_channels=64,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ELU(inplace=True),

                                   nn.Conv2d(in_channels=64, out_channels=64,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=64),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ELU(inplace=True),

                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64,
                                             kernel_size=(1, 15), padding=(0, 7),
                                             bias=False, groups=64),
                                   nn.Conv2d(in_channels=64, out_channels=128,
                                             kernel_size=(1, 1), bias=False),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.ELU(inplace=True),
                                   nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                                   nn.Dropout2d(p=0.4, inplace=False))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.fc = nn.Linear(128, self.nb_classes)

    def forward(self, x):
        x = self.temporal_conv(x)
        # print(x.shape)
        x = self.spatial_conv(x)
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)

        x = self.gap(x)
        # print(x.shape)
        x = x.view(-1, 128)
        # torch.flatten(x, 1)
        # print(x.shape)
        # print(self.fc.weight.shape)
        # x = self.dropout(x)
        x = self.fc(x)

        return x


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
