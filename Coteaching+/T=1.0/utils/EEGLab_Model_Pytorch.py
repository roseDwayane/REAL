

import torch
from torch import nn


class EEGNetPytorch10(nn.Module):
    def __init__(self, nb_classes=4, kern_length=125, strides=(1, 1),
                 norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
        super().__init__()
        self.name = 'EEGNetPytorch10'

        self.stage0 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, kern_length),
                                              stride=strides, padding=(kern_length//2, 0)),
                                    nn.BatchNorm2d(8),

                                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(30, 1),
                                              stride=(30, 1)),
                                    nn.BatchNorm2d(16),
                                    nn.ELU(inplace=True),
                                    nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
                                    )

        self.stage1 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 9)),
                                    nn.BatchNorm2d(16),
                                    nn.ELU(inplace=True),

                                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 9)),
                                    nn.BatchNorm2d(16),
                                    nn.ELU(inplace=True),

                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1)),
                                    nn.BatchNorm2d(32),
                                    nn.ELU(inplace=True),

                                    nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
                                    )

        self.stage2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 9)),
                                    nn.BatchNorm2d(32),
                                    nn.ELU(inplace=True),

                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 9)),
                                    nn.BatchNorm2d(32),
                                    nn.ELU(inplace=True),

                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1)),
                                    nn.BatchNorm2d(64),
                                    nn.ELU(inplace=True),

                                    nn.AdaptiveAvgPool2d(1)
                                    )

        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(64, 128),
                                nn.ELU(inplace=True)
                                )

        self.softmax = nn.Sequential(nn.Linear(128, nb_classes),
                                     #nn.Softmax(dim=1)
                                     )

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x


#EEGNet_v25(nb_classes=2)
#tt = GroupNormalization(4, name='tt')
#print(tt.name)


"""
class MaxNorm(Constraint):
"""
"""MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Arguments:
      m: the maximum norm for the incoming weights.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
"""
