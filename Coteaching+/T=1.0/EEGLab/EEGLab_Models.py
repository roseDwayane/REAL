
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.layers import Permute, Reshape, Multiply

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import SpatialDropout2D, Dropout

from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm, UnitNorm

from tensorflow.keras.optimizers import SGD, Adam

import tensorflow as tf

import sys


def mnist_model(n_classes=10, input_shape=(28, 28, 1)):
    """define cnn model for MNIST.
    可能是TF2.0的bug, 正则项调整为kernel_constraint!
    然而, 在使用generator时, kernel_regularizer恢复正常. 奇怪!
    :return:
    """

    # 实例化Sequential
    model = Sequential()

    # Convolutional layer
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', input_shape=input_shape,
                     #kernel_regularizer=regularizers.l2(0.001),    # 会导致fit停滞？？？, 故换成kernel_constraint
                     kernel_constraint=max_norm(0.1))
              )
    model.add(BatchNormalization())     # Improvement to Learning
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2)))

    # Full connected layer
    model.add(Flatten())

    model.add(Dense(128, kernel_initializer='he_uniform',
              #kernel_regularizer=regularizers.l2(0.01),
              kernel_constraint=max_norm(0.5)
              ))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    #opt = SGD(lr=0.001, momentum=0.9)
    #opt = Adam(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet(nb_classes=4, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
    advised to do some model searching to get optimal performance on your
    particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.

    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)   # (N, H, W, C)

    """
    常规的kernel, H x W x Cin x Cout  H x W x F1 x (F1*D)
    H x w x 1 x D   
    """
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # Block2
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)

    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    # Dense
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)

    softmax = Activation('softmax', name='softmax')(dense)

    model = Model(inputs=input1, outputs=softmax, name='EEGNet')

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v2(nb_classes=4, Chans=64, Samples=128,
              dropoutRate=0.5, kernLength=64, F1=8,
              D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    """
    v2:
      Conv2D替换DepthwiseConv2D, SeparableConv2D
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)   # (N, H, W, C)

    """block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)"""
    block1 = Conv2D(F1*D, (Chans, 1), use_bias=False)(block1)                   # @v2

    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # Block2
    ##################################################################
    """block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)"""
    block2 = Conv2D(F2, (1, 16), use_bias=False)(block1)                        # @v2

    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)

    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    # Dense
    ##################################################################
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)

    softmax = Activation('softmax', name='softmax')(dense)

    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v2')

    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v3(nb_classes=4, Chans=64, Samples=128,
              dropoutRate=0.5, kernLength=64, F1=8,
              D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):

    """
    v2:
      Conv2D替换DepthwiseConv2D, SeparableConv2D
    v3:
      Block2添加1层Conv2D
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    x = Conv2D(F1, (1, kernLength), padding='same',
               input_shape=(Chans, Samples, 1),
               use_bias=False)(input1)
    x = BatchNormalization()(x)   # (N, H, W, C)

    """block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)"""

    x = Conv2D(F1*D, (Chans, 1), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    x = dropoutType(dropoutRate)(x)

    # Block2
    ##################################################################
    """block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)"""
    x = Conv2D(F2, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same')(x)              # +@v3
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 8))(x)
    x = dropoutType(dropoutRate)(x)

    # Dense
    ##################################################################
    flatten = Flatten(name='flatten')(x)

    # softmax
    ##################################################################
    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    # model define
    model = Model(inputs=input1, outputs=softmax)

    opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v4(nb_classes=4, Chans=64, Samples=128,
              dropoutRate=0.5, kernLength=64, F1=8,
              D=2, F2=16, norm_rate=0.25, dropout_type='Dropout'):
    """
    v2:
      Conv2D替换DepthwiseConv2D, SeparableConv2D
    v3:
      Block2 添加1层Conv2D
    v4:
      Block2 添加1层Conv2D
    """

    if dropout_type == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropout_type == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    x = Conv2D(F1, (1, kernLength), padding='same',
               input_shape=(Chans, Samples, 1),
               use_bias=False)(input1)
    x = BatchNormalization()(x)   # (N, H, W, C)

    """block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)"""

    x = Conv2D(F1*D, (Chans, 1), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = dropoutType(dropoutRate)(x)                                            # @v4

    # Block2
    ##################################################################
    """block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)"""
    x = Conv2D(F2, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same')(x)                  # +@v3
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same')(x)                  # +@v4
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 8))(x)
    #x = dropoutType(dropoutRate)(x)                                            # @v4

    # Dense
    x = Flatten(name='flatten')(x)
    """x = Dense(256, name='dense0', kernel_constraint=max_norm(norm_rate))(x)    # @v4
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    flatten = dropoutType(dropoutRate)(x)"""

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # model definition
    model = Model(inputs=input1, outputs=softmax)

    opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v5(nb_classes=4, Chans=64, Samples=128,
              dropoutRate=0.5, kernLength=64, F1=8,
              D=2, F2=32, norm_rate=0.25, l2_weight=0.001, dropoutType='Dropout'):
    """
    v2:
      Conv2D替换DepthwiseConv2D, SeparableConv2D
    v3:
      Block2 添加1层Conv2D
    v4:
      Block2 添加1层Conv2D
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    # (N, H, W, C)
    input1 = Input(shape=(Chans, Samples, 1))

    # Block1
    ##################################################################
    x = Conv2D(F1, (1, kernLength), padding='same',
               input_shape=(Chans, Samples, 1),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x = Conv2D(F1*D, (Chans, 1), use_bias=False, kernel_regularizer=regularizers.l2(l2_weight))(x)  # m@v2, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = dropoutType(dropoutRate)(x)                                            # @v4

    # Block2
    ##################################################################
    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)   # +@v3, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)   # +@v4, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 8))(x)
    #x = dropoutType(dropoutRate)(x)                                            # @v4

    # Dense
    x = Flatten(name='flatten')(x)
    """x = Dense(256, name='dense0', kernel_constraint=max_norm(norm_rate))(x)    # @v4
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""
    #x = dropoutType(dropoutRate)(x)

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # model definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v5')

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v6(nb_classes=4, Chans=64, Samples=128, strides=(1, 1),
              dropoutRate=0.5, kernLength=64, F1=8,
              D=2, F2=16, norm_rate=0.25, l2_weight=0.001, dropoutType='Dropout'):
    """
    v6:
      在第1层增加strides, 应对长的samples, 减少计算量和内存占用.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    # (N, H, W, C)
    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    x = Conv2D(F1, (1, kernLength), padding='same', strides=strides,
               input_shape=(Chans, Samples, 1),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    """block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)"""

    x = Conv2D(F1*D, (Chans, 1), use_bias=False, kernel_regularizer=regularizers.l2(l2_weight))(x)              # m@v2, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    x = dropoutType(dropoutRate)(x)

    # Block2
    """block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)"""
    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)   # +@v3, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)   # +@v4, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 8))(x)
    x = dropoutType(dropoutRate)(x)

    # Dense
    x = Flatten(name='flatten')(x)
    """x = Dense(256, name='dense0', kernel_constraint=max_norm(norm_rate))(x)    # @v4
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    flatten = dropoutType(dropoutRate)(x)"""

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # model definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v6')

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v7(nb_classes=4, Chans=64, Samples=128, strides=(1, 1),
              dropoutRate=0.5, kernLength=64, F1=16,
              D=2, F2=32, norm_rate=0.25, l2_weight=0.001, dropoutType='Dropout'):
    """
    v2:
      Conv2D替换DepthwiseConv2D, SeparableConv2D
    v3:
      Block2 添加1层Conv2D
    v4:
      Block2 添加1层Conv2D
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v7:
      修改F1=16, F2数值
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    # (N, H, W, C)
    input1 = Input(shape=(Chans, Samples, 1))

    # Block1
    ##################################################################
    x = Conv2D(F1, (1, kernLength), padding='same', strides=strides,
               input_shape=(Chans, Samples, 1),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x = Conv2D(F1, (Chans, 1), use_bias=False, kernel_constraint=max_norm(1.0))(x)  # m@v2, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    x = dropoutType(dropoutRate)(x)                                            # @v4

    # Block2
    ##################################################################
    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_constraint=max_norm(1.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_constraint=max_norm(1.0))(x)   # +@v3, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_constraint=max_norm(1.0))(x)   # +@v4, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(F2, (1, 16), use_bias=False, padding='same', kernel_constraint=max_norm(1.0))(x)   # +@v4, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 8))(x)
    x = dropoutType(dropoutRate)(x)                                            # @v4

    # Dense
    #x = Flatten(name='flatten')(x)
    #x = Dense(128, name='dense0')(x)    # @v4
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)
    #x = dropoutType(dropoutRate)(x)

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # model definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v7')

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v8(nb_classes=4, Chans=64, Samples=128, strides=(1, 1), image_channels=1,
              dropoutRate=0.5, kernLength=64, norm_rate=0.25, l2_weight=0.001, dropoutType='Dropout',
              f0=8, f1=16, f2=16):
    """
    v2:
      Conv2D替换DepthwiseConv2D, SeparableConv2D
    v3:
      Block2 添加1层Conv2D
    v4:
      Block2 添加1层Conv2D
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v7:
      修改F1=16, F2数值
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    # (N, H, W, C)
    input1 = Input(shape=(Chans, Samples, image_channels))

    # Block1
    ##################################################################
    x = Conv2D(8, (1, kernLength), padding='same', strides=strides,
               #input_shape=(Chans, Samples, 1),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (Chans, 1), use_bias=False)(x)  # m@v2, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    #x = Conv2D(16, (30, 1), use_bias=False, padding='same')(x)
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = dropoutType(dropoutRate)(x)

    # Block2
    ##################################################################
    x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)   # +@v3, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)   # +@v4, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    #x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)   # +@v4, m@v5
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = dropoutType(dropoutRate)(x)

    # Dense
    x = Flatten(name='flatten')(x)

    #x = Dense(64, name='dense0')(x)
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)
    #x = dropoutType(dropoutRate)(x)

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v8')

    # config
    opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v9(nb_classes=4, Chans=64, Samples=128, strides=(1, 1), image_channels=1,
              dropoutRate=0.5, kernLength=64, norm_rate=0.25, l2_weight=0.001, dropoutType='Dropout',
              f0=8, f1=16, f2=16):
    """
    v2:
      Conv2D替换DepthwiseConv2D, SeparableConv2D
    v3:
      Block2 添加1层Conv2D
    v4:
      Block2 添加1层Conv2D
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v7:
      修改F1=16, F2数值
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    # (N, H, W, C)
    input1 = Input(shape=(Chans, Samples, image_channels))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kernLength), padding='same', strides=strides, use_bias=False, name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (Chans, 1), use_bias=False, name='spatial')(x)  # m@v2, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    #x = Conv2D(16, (30, 1), use_bias=False, padding='same')(x)
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = dropoutType(dropoutRate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)   # +@v3, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)   # +@v4, m@v5
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = dropoutType(dropoutRate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = dropoutType(dropoutRate)(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)

    #x = Dense(64, name='dense0')(x)
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)
    #x = dropoutType(dropoutRate)(x)

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v9')

    # config
    opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


def EEGNet_v10(nb_classes=4, Chans=64, Samples=128, strides=(1, 1), image_channels=1,
              dropoutRate=0.5, kernLength=64, norm_rate=0.25, l2_weight=0.001, dropoutType='Dropout',
              f0=8, f1=16, f2=16):
    """
    v2:
      Conv2D替换DepthwiseConv2D, SeparableConv2D
    v3:
      Block2 添加1层Conv2D
    v4:
      Block2 添加1层Conv2D
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v7:
      修改F1=16, F2数值
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, Samples, image_channels))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kernLength), use_bias=False, padding='same', strides=strides)(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = dropoutType(dropoutRate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = dropoutType(dropoutRate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    #x = AveragePooling2D((1, 2))(x)
    x = GlobalAveragePooling2D()(x)
    x = dropoutType(dropoutRate)(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)
    x = dropoutType(dropoutRate)(x)

    x = Dense(128, name='dense1')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v10')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v11(nb_classes=4, Chans=64, Samples=128, strides=(1, 1), image_channels=1,
              kern_length=64, norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), use_bias=False, padding='same', strides=strides)(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = DropoutLayer(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = DropoutLayer(dropout_rate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    #x = Conv2D(32, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout_rate)(x)

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v11')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v12(nb_classes=4, kern_length=125,
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=125 @250Hz
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), use_bias=False, padding='same')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = DropoutLayer(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = DropoutLayer(dropout_rate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    #x = Conv2D(32, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    #x = BatchNormalization()(x)
    #x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout_rate)(x)

    # softmax
    x = Dense(nb_classes, name='dense', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v12')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v13(nb_classes=4, kern_length=25,
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)

    x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v13')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v15x(nb_classes=4, input_shape=(30, 501, 1), kernel_length=64,
               l2_weight=0.001, norm_rate=0.25, dropout_rate=0.5):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v15:
      输入shape调整为 (30, 501, 8~16)
    """

    input1 = Input(shape=input_shape)

    # Stage 0
    ##################################################################
    x = Conv2D(32, (1, kernel_length), use_bias=False, padding='same')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(64, (30, 1), use_bias=False, strides=(30, 1), kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = SpatialDropout2D(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    #x = SpatialDropout2D(dropout_rate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)
    #x = Dropout(dropout_rate)(x)

    # softmax
    x = Dense(nb_classes, name='dense')(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v15')

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v23(nb_classes=4, kern_length=25,
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v23:
      1xn -> 3x3
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(256, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    """x = Dense(64, name='fc2', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v23')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v24(nb_classes=4, kern_length=25,
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v24:
      1xn -> 4x4
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v24')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v25(nb_classes=4, kern_length=25,
               norm_rate=0.25, l2_weight=0.005, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v24:
      1xn -> 4x4
    v25:
      MaxPooling2D
    """
    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)
    #x = MaxPooling2D((2, 2))(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v25')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v30(nb_classes=4, kern_length=25,
               norm_rate=0.25, l2_weight=0.005, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v24:
      1xn -> 4x4
    v25:
      MaxPooling2D
    v30:
      loss='sparse_categorical_crossentropy', 省去了one-hot
    """
    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (4, 4), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v30')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v33(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)

    x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v33')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v34(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)

    #x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v34')

    # config
    opt = Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


from tensorflow.keras import backend as K
#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints


class GroupNormalization(Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group
    the mean and variance for normalization.
    Group Normalization's computation is independent
     of batch sizes, and its accuracy is stable in a wide range of batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical to
    Layer Normalization.

    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.

    # Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 name=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         trainable=True,
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        trainable=True,
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        #self.built = True
        super(GroupNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)      # !!!!
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        #                            keep_dims=True)
        mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)

        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        # finally we reshape the output back to the input shape
        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }

        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class GroupNormalization0(Layer):
    def __init__(self,
                 groups,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 name=None,
                 **kwargs):
        super(GroupNormalization0, self).__init__(name=name, **kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = 1.0e-5
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        shape = (self.groups,)
        broadcast_shape = [-1, self.groups, 1, 1, 1]

        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        super(GroupNormalization0, self).build(input_shape)  # 一定要在最后调用它

    def call_old(self, inputs, **kwargs):
        G = self.groups

        input_shape = K.int_shape(inputs)
        C = input_shape[-1]
        c_g = C // G

        if self.axis in {-1, 3}:
            inputs = K.permute_dimensions(inputs, (3, 0, 1, 2))

        outputs = []
        for g in range(G):
            inputs_n = inputs[g*c_g:(g+1)*c_g]
            gn_mean = K.mean(inputs_n, axis=[1, 2, 3], keepdims=True)
            gn_variance = K.var(inputs_n, axis=[1, 2, 3], keepdims=True)
            outputs_n = (inputs_n - gn_mean) / (K.sqrt(gn_variance + self.epsilon))
            outputs.append(outputs_n)

        outputs = K.concatenate(outputs, axis=0)
        if self.axis in {-1, 3}:
            outputs = K.permute_dimensions(outputs, (1, 2, 3, 0))

        """# transpose:[ba,h,w,c] -> [bs,c,h,w]
        if self.axis in {-1, 3}:
            inputs = K.permute_dimensions(inputs, (0, 3, 1, 2))

        input_shape = K.int_shape(inputs)

        # GN操作需要根据groups对通道分组        
        N, C, H, W = input_shape
        inputs = K.reshape(inputs, (N, G, C // G, H, W))
        #inputs.assign_sub()

        # 计算分组通道的均值和方差
        gn_mean = K.mean(inputs, axis=[2, 3, 4], keepdims=True)
        gn_variance = K.var(inputs, axis=[2, 3, 4], keepdims=True)

        # 当模型用于训练阶段时，使用分组通道实时计算均值/方差
        outputs = (inputs - gn_mean) / (K.sqrt(gn_variance + self.epsilon))
        outputs = K.reshape(outputs, [N, C, H, W])

        # transpose: [bs,c,h,w] -> [ba,h,w,c]
        if self.axis in {-1, 3}:
            outputs = K.permute_dimensions(outputs, (0, 2, 3, 1))

        # 根据模型状态不同选择不同的GN计算方法，train时选择outputs，test时选择gn_inference
        #return K.in_train_phase(outputs, gn_inference,training=training)"""
        return outputs

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)                                # (1, 1, 1, 1)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups      # (1, 1, 1, C//G)
        broadcast_shape.insert(1, self.groups)                                  # (1, G, 1, 1, C//G)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups           # C // G @self.axis
        group_axes.insert(1, self.groups)                                       # G @1

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        """mean, variance = KC.moments(inputs, group_reduction_axes[2:],
                                    keep_dims=True)"""
        mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)

        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        # finally we reshape the output back to the input shape
        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }

        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def EEGNet_v40(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    x = Flatten(name='flatten')(x)

    x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v40')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v41(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=4)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    ##################################################################
    x = Flatten(name='flatten')(x)

    x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v41')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v42(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=4)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    #x = BatchNormalization()(x)
    x = GroupNormalization(groups=4)(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    #x = BatchNormalization()(x)
    x = GroupNormalization(groups=4)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    #x = BatchNormalization()(x)
    x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    ##################################################################
    x = Flatten(name='flatten')(x)

    x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v42')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v43(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=4)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=4)(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=4)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Dense
    ##################################################################
    x = Flatten(name='flatten')(x)

    x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    #x = BatchNormalization()(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v43')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v51(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    v51:
      取消 GlobalAveragePooling2D 后的 fc
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=4)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    #x = GroupNormalization(groups=8)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)

    """x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v51')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v52(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    v51:
      取消 GlobalAveragePooling2D 后的 fc
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)

    """x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    #x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v52')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v53(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    v51:
      取消 GlobalAveragePooling2D 后的 fc
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)

    """x = Dropout(dropout_rate)(x)"""
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    #x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v53')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v54(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    v51:
      取消 GlobalAveragePooling2D 后的 fc
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=8)(x)     # v54
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=8)(x)     # v54
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)

    """x = Dropout(dropout_rate)(x)"""
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    #x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v54')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v55scnn(nb_classes=2, strides=(1, 1), l2_weight=0.001,):
    """
    v55scnn:
      标准的CNN架构, 用于对比EEGNet.
      BatchNormalization()
      Activation('elu')
      GlobalAveragePooling2D()
    """

    input1 = Input(shape=(30, 501, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(16, (5, 5), strides=strides, padding='same', name='temporal',
               kernel_regularizer=regularizers.l2(l2_weight),
               use_bias=False)(input1)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v55scnn')

    # config
    opt = Adam(lr=0.0001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v55(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    v51:
      取消 GlobalAveragePooling2D 后的 fc
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    """x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    x = Conv2D(8, (1, kern_length), strides=strides, padding='same', name='temporal',
               kernel_regularizer=regularizers.l2(l2_weight),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(0.5)(x)

    # Stage 1
    ##################################################################
    """x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=8)(x)     # v54
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=8)(x)     # v54
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)

    ##################################################################
    x = Conv2D(128, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    """x = Dropout(dropout_rate)(x)
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    #x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v55')

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v810(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    v810:
      修改EEGNet的第2层:
      30x1 -> 7x1
      'valid' -> 'same'
    """

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 501, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, padding='same', name='temporal',
               kernel_regularizer=regularizers.l2(l2_weight),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((7, 1), use_bias=False, name='spatial', padding='same',
                        depth_multiplier=2,
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 3))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v810')

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v820(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    v810:
      修改EEGNet的第2层:
      30x1 -> 7x1, 5x1
      'valid' -> 'same'
    v820:
      第1层输出通道数提升至16;
      第2层: dilation_rate=(2, 1)；
    """

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(16, (1, kern_length), strides=strides, padding='same', name='temporal',
               kernel_regularizer=regularizers.l2(l2_weight),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((5, 1), use_bias=False, name='spatial', padding='same',
                        depth_multiplier=2, dilation_rate=(2, 1),
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    """x = MaxPooling2D((2, 2))(x)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v820')

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v830(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    v810:
      修改EEGNet的第2层:
      30x1 -> 7x1, 5x1
      'valid' -> 'same'
    v820:
      第1层输出通道数提升至16;
    v830:
      第2层加入inception;
    """
    from tensorflow.keras.layers import Concatenate

    input1 = Input(shape=(None, None, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(16, (1, kern_length), strides=strides, padding='same', name='temporal',
               kernel_regularizer=regularizers.l2(l2_weight),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x1 = DepthwiseConv2D((5, 1), use_bias=False, name='spatial1', padding='same',
                         depth_multiplier=1, dilation_rate=(1, 1),
                         kernel_regularizer=regularizers.l2(l2_weight))(x)

    x2 = DepthwiseConv2D((5, 1), use_bias=False, name='spatial2', padding='same',
                         depth_multiplier=1, dilation_rate=(2, 1),
                         kernel_regularizer=regularizers.l2(l2_weight))(x)

    inception = [x1, x2]

    x = Concatenate(axis=-1)(inception)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    """x = MaxPooling2D((2, 2))(x)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    """x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    x = MaxPooling2D((1, 2))(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    """x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)"""

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v830')

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v60(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    v51:
      取消 GlobalAveragePooling2D 后的 fc
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)

    # Stage 3 (7~17)
    ##################################################################
    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)

    """x = Dropout(dropout_rate)(x)"""
    x = Dense(128, name='fc1', kernel_constraint=max_norm(2.0))(x)
    #x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v60')

    # config
    #opt = Adam(lr=0.001)
    opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v61(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    v51:
      取消 GlobalAveragePooling2D 后的 fc
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    #x = Dropout(dropout_rate)(x)

    # Stage 3 (7~17)
    ##################################################################
    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)

    """x = Dropout(dropout_rate)(x)"""
    x = Dense(256, name='fc1', kernel_constraint=max_norm(2.0))(x)
    #x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v61')

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v62(nb_classes=4, kern_length=125, strides=(1, 1),
               norm_rate=0.25, l2_weight=0.001, dropout_rate=0.5, dropout_type='Dropout'):
    """
    v5:
      添加L2正则项, 提高泛化能力.
      kernel_regularizer=regularizers.l2(l2_weight)
    v8:
      加入 full-connect 层
      ! kernel_constraint 要慎用 !
      def __init__(self, max_value=2, axis=0)
    v11:
      修改 Chans=None, 可以自适应输入的高度. 需要 GlobalAveragePooling2D() !
      训练和推理可以分别使用高度不同的输入.
    v12:
      kern_length=25 @250Hz
    v33:
      strides=(1, 4), 增加长度
      loss='sparse_categorical_crossentropy', 省去了one-hot
    v40:
      GroupNormalization
      加载模型注意:
      custom_objects = {"GroupNormalization": GroupNormalization}
      self_model = load_model(model_file, custom_objects=custom_objects)
    v51:
      取消 GlobalAveragePooling2D 后的 fc
    """

    if dropout_type == 'SpatialDropout2D':
        DropoutLayer = SpatialDropout2D
    elif dropout_type == 'Dropout':
        DropoutLayer = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(None, None, 1))

    # Stage 0 (1~6)
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, use_bias=False, padding='same', name='temporal')(input1)
    x = BatchNormalization()(x)

    x = Conv2D(16, (30, 1), use_bias=False, strides=(30, 1), name='spatial', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 1 (7~17)
    ##################################################################
    x = Conv2D(16, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 2 [18~27]
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=16)(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Stage 3 (7~17)
    ##################################################################
    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = GroupNormalization(groups=32)(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)

    """x = Dropout(dropout_rate)(x)"""
    x = Dense(256, name='fc1', kernel_constraint=max_norm(2.0))(x)
    #x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name='EEGNet_v62')

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


##############################################################################################

def EEGNet_v90a(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Normal
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 501, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, kern_length), strides=strides, padding='same', name='temporal1',
               kernel_regularizer=regularizers.l2(l2_weight),
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((7, 1), use_bias=False, name='spatial', padding='same',
                        depth_multiplier=2,
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 3))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v90b(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Inception
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 501, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((7, 1), use_bias=False, name='spatial', padding='same',
                        depth_multiplier=2,
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 3))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v90c(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Inception
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((7, 1), use_bias=False, name='spatial', padding='same',
                        depth_multiplier=2,
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 3))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v90c2(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Inception
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((7, 1), use_bias=False, name='spatial', padding='same',
                        depth_multiplier=2,
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 3))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v90d(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Inception
    """
    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v90e(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Inception
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                #kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                #kernel_regularizer=regularizers.l2(l2_weight),
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v96(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Inception
        None + depthwise_regularizer=regularizers.l2(l2_weight)
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_regularizer=regularizers.l2(l2_weight)
                        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v96b(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Inception
        depthwise_constraint=max_norm(1.0, axis=[0, 1, 2])
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0, axis=[0, 1, 2]),
                        #depthwise_regularizer=regularizers.l2(l2_weight)
                        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v96c(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(2.0))(x)      # == max_norm(2.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v96m(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(60, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((60, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(2.0))(x)      # == max_norm(2.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v96n(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def slice1(a):
        return a[:30]

    def slice2(a):
        return a[30:]

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(60, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    x3 = Slice(0, 30)(x)
    x4 = Slice(30, 60)(x)

    x3 = DepthwiseConv2D((30, 1), use_bias=False, name='spatial1',
                         depth_multiplier=2,
                         depthwise_constraint=max_norm(2.0))(x3)      # == max_norm(2.0， axis=0)

    x4 = DepthwiseConv2D((30, 1), use_bias=False, name='spatial2',
                         depth_multiplier=2,
                         depthwise_constraint=max_norm(2.0))(x4)      # == max_norm(2.0， axis=0)

    inception2 = [x3, x4]
    x = Concatenate(axis=-3)(inception2)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v96k(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    epochs的平均值与epoch合并
        30 -> 30+30
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def slice1(a):
        return a[:30]

    def slice2(a):
        return a[30:]

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30+30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    # x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
    #             use_bias=False)(input1)
    #
    # inception = [x1, x2]
    # x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x3 = Slice(0, 30)(x)
    x4 = Slice(30, 60)(x)

    # 参数共享
    depth_wise_conv_2d = DepthwiseConv2D((30, 1), use_bias=False, name='spatial1',
                                         depth_multiplier=2, depthwise_constraint=max_norm(2.0))     # == max_norm(2.0， axis=0)

    x3 = depth_wise_conv_2d(x3)
    x4 = depth_wise_conv_2d(x4)

    inception2 = [x3, x4]
    x = Concatenate(axis=-3)(inception2)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model


def EEGNet_v96s(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    epochs的平均值与epoch合并
        30 -> 30+30
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def slice1(a):
        return a[:30]

    def slice2(a):
        return a[30:]

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30+30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    # x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
    #             use_bias=False)(input1)
    #
    # inception = [x1, x2]
    # x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x3 = Slice(0, 30)(x)
    x4 = Slice(30, 60)(x)

    # 参数共享
    depth_wise_conv_2d = DepthwiseConv2D((30, 1), use_bias=False, name='spatial1',
                                         depth_multiplier=2, depthwise_constraint=max_norm(1.0, axis=0))     # == max_norm(1.0， axis=0)

    x3 = depth_wise_conv_2d(x3)
    x4 = depth_wise_conv_2d(x4)

    # inception2 = [x3, x4]
    # x = Concatenate(axis=-3)(inception2)
    x = x3 + x4

    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 3
    ##################################################################
    x = Conv2D(256, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()
    return model


def EEGNet_v98b(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    #input1 = Input(shape=(None, None, 1))
    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v98c(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


class Slice0(Layer):
    """
    切片层
    # Arguments

    # Input shape

    # Output shape

    """

    def __init__(self,
                 start,
                 end,
                 name=None,
                 **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.start = start
        self.end = end

    def build(self, input_shape):
        super(Slice, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        # broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        outputs = inputs[:, self.start:self.end, :, :]

        return outputs

    def get_config(self):
        config = {
            'start': self.start,
            'end': self.end
        }

        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0] // 2, ) + input_shape[1:]


def EEGNet_v110(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 16), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v110b(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        x Inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x25
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    # x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
    #             use_bias=False)(input1)
    #
    # inception = [x1, x2]
    # x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v110c(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        x Inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    # x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
    #             use_bias=False)(input1)
    #
    # inception = [x1, x2]
    # x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    """x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 3))(x)     # (2, 2)"""
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v110d(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    # x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
    #             use_bias=False)(input1)
    #
    # inception = [x1, x2]
    # x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=4,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # # Stage 1
    # ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)     # (2, 2)
    # #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 4
    ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v110e(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    # x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
    #             use_bias=False)(input1)
    #
    # inception = [x1, x2]
    # x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=4,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # # Stage 1
    # ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)     # (2, 2)
    # #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v110f(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    # x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
    #             use_bias=False)(input1)
    #
    # inception = [x1, x2]
    # x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (2, 2)
    # #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v110g(nb_classes=4, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
                use_bias=False)(input1)
    # x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal2',
    #             use_bias=False)(input1)
    #
    # inception = [x1, x2]
    # x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x1)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (2, 2)
    # #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v110h(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal1',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (2, 2)
    # #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v110k(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        inception
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal_1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal_2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # # Stage 1
    # ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = AveragePooling2D((1, 2))(x)     # (2, 2)
    # # #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v110m(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        inception(125, 250)
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x1 = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal_1',
                use_bias=False)(input1)
    x2 = Conv2D(8, (1, 25), strides=strides, padding='same', name='temporal_2',
                use_bias=False)(input1)

    inception = [x1, x2]
    x = Concatenate(axis=-1)(inception)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # # Stage 1
    # ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = AveragePooling2D((1, 2))(x)     # (2, 2)
    # # #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v110n(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (2, 2)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v110s(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (2, 2)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v110t(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (2, 2)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 2))(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(512, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(512, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v110w(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v110y(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 25), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v110z(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x9
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v120(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=UnitNorm() 单位向量

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x9
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=UnitNorm())(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 9), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v120b(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=UnitNorm() 单位向量

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=UnitNorm())(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v120c(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(16, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    # # Stage 1
    # ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


class SliceStep(Layer):
    """
    切片层
    # Arguments

    # Input shape

    # Output shape

    """

    def __init__(self,
                 start,
                 step,
                 name=None,
                 **kwargs):
        super(SliceStep, self).__init__(**kwargs)
        self.start = start
        self.step = step

    def build(self, input_shape):
        super(SliceStep, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        # broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        outputs = inputs[:, :, :, self.start::self.step]

        return outputs

    def get_config(self):
        config = {
            'start': self.start,
            'step': self.step
        }

        base_config = super(SliceStep, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] // self.step, )


def EEGNet_v120d(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        DepthwiseConv2D
        对每一个epoch的每一个channel单独训练一个temporal filter

        depthwise_constraint=max_norm()
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    input2 = Permute((3, 2, 1))(input1)    # (N, 1, 251, 30)

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter" for every channels in every epochs
    x = DepthwiseConv2D((1, 125), strides=strides, padding='same', name='temporal',
                        depth_multiplier=8,
                        use_bias=False)(input2)     # (N, 1, 251, 30*8)
    x = BatchNormalization()(x)

    x_list = []
    for n in range(8):
        xn = SliceStep(n, 8)(x)
        x_list.append(xn)
    x = Concatenate(axis=1)(x_list)     # (N, 8, 251, 30)
    x = Permute((3, 2, 1))(x)           # (N, 30, 251, 8)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v120f(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v120g(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v120h(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.3)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v120k(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = conv_block(filters=32, kernel_size=(1, 15), n_conv=2, padding='same', pooling=AveragePooling2D((1, 2)))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v120m(nb_classes=2, channels=251, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v123(nb_classes=2, samples=501, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v123a(nb_classes=2, samples=501, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v123b(nb_classes=2, samples=501, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((input_dim, 1, 1)),
                         Conv2D(input_dim//4, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v123c(nb_classes=2, samples=501, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    SE
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//4, (1, 1), use_bias=False,
                                #kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                #kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    se1 = SE(32, 1)(x)
    x = Multiply()([x, se1])

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v123d(nb_classes=2, samples=501, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    SE @evray blocks
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//4, (1, 1), use_bias=False,
                                #kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                #kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    se1 = SE(32, 1)(x)
    x = Multiply()([x, se1])

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    se2 = SE(64, 2)(x)
    x = Multiply()([x, se2])

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.2)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    se3 = SE(128, 3)(x)
    x = Multiply()([x, se3])

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    se4 = SE(256, 4)(x)
    x = Multiply()([x, se4])

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


class SliceSelect(Layer):
    """
    切片层
    # Arguments

    # Input shape

    # Output shape

    """

    def __init__(self,
                 index,
                 name=None,
                 **kwargs):
        super(SliceSelect, self).__init__(**kwargs)
        self.index = index

    def build(self, input_shape):
        super(SliceSelect, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # [1, 1, 1, 1]
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # [1, 1, 1, C//G]
        # broadcast_shape.insert(1, self.groups)                              # [1, G, 1, 1, C//G]
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        outputs = inputs[:, self.index:self.index+1, :, :]

        return outputs

    def get_config(self):
        config = {
            'index': self.index
        }

        base_config = super(SliceSelect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1) + input_shape[2:]


def EEGNet_v125(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like

    7x7, 3x3
    SpatialDropout2D
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 FFT-like
    ##################################################################
    conv_1x1 = Conv2D(251, (1, 1), use_bias=False)
    x_list = []
    for n in range(30):
        xn = SliceSelect(n)(input1)             # (N, 1, 251, 1)
        xn = Permute((1, 3, 2))(xn)             # (N, 1, 1, 251)
        xn = conv_1x1(xn)                       # (N, 1, 1, 251)
        x_list.append(xn)
    x = Concatenate(axis=1)(x_list)             # (N, 30, 1, 251)
    x = Permute((1, 3, 2))(x)                   # (N, 30, 251, 1)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v125b(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 FFT-like
    ##################################################################
    conv_1x1 = Conv2D(251, (1, 1), use_bias=False)
    x_list = []
    for n in range(30):
        xn = SliceSelect(n)(input1)             # (N, 1, 251, 1)
        xn = Permute((1, 3, 2))(xn)             # (N, 1, 1, 251)
        xn = conv_1x1(xn)                       # (N, 1, 1, 251)
        x_list.append(xn)
    x = Concatenate(axis=1)(x_list)             # (N, 30, 1, 251)
    x = Permute((1, 3, 2))(x)                   # (N, 30, 251, 1)
    x = BatchNormalization()(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v125c(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        251->126
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 FFT-like
    ##################################################################
    conv_1x1 = Conv2D(126, (1, 1), use_bias=False)
    x_list = []
    for n in range(30):
        xn = SliceSelect(n)(input1)             # (N, 1, 251, 1)
        xn = Permute((1, 3, 2))(xn)             # (N, 1, 1, 251)
        xn = conv_1x1(xn)                       # (N, 1, 1, 126)
        x_list.append(xn)
    x = Concatenate(axis=1)(x_list)             # (N, 30, 1, 126)
    x = Permute((1, 3, 2))(x)                   # (N, 30, 126, 1)
    x = BatchNormalization()(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v125d(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        251->126
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 FFT-like
    ##################################################################
    conv_1x1 = Conv2D(126, (1, 1), use_bias=False)
    x_list = []
    for n in range(30):
        xn = SliceSelect(n)(input1)             # (N, 1, 251, 1)
        xn = Permute((1, 3, 2))(xn)             # (N, 1, 1, 251)
        xn = conv_1x1(xn)                       # (N, 1, 1, 126)
        x_list.append(xn)
    x = Concatenate(axis=1)(x_list)             # (N, 30, 1, 126)
    x = Permute((1, 3, 2))(x)                   # (N, 30, 126, 1)
    x = BatchNormalization(axis=-3)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v126(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        251->126
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 FFT-like
    ##################################################################
    x = BatchNormalization(axis=1)(input1)          # 对EEG channels实施归一化(规则化), 而不是针对CNN的channels(feature maps)xxx

    x = Permute((1, 3, 2))(x)                       # (N, 30, 251, 1) -> (N, 30, 1, 251)
    x = Conv2D(126, (1, 1), use_bias=False, kernel_constraint=UnitNorm(axis=[0, 1, 2]))(x)      # (N, 30, 1, 126)
    x = Permute((1, 3, 2))(x)                       # (N, 30, 126, 1)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.2)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v126a(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        251->126
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 FFT-like
    ##################################################################
    x = BatchNormalization(axis=1)(input1)          # 对EEG channels实施归一化(规则化), 而不是针对CNN的channels(feature maps)

    x = Permute((1, 3, 2))(x)                       # (N, 30, 251, 1) -> (N, 30, 1, 251)
    x = Conv2D(251, (1, 1), use_bias=False, kernel_constraint=UnitNorm(axis=[0, 1, 2]))(x)      # (N, 30, 1, 126)
    x = Permute((1, 3, 2))(x)                       # (N, 30, 126, 1)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v126b(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        251->125
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 FFT-like
    ##################################################################
    x = Permute((1, 3, 2))(input1)                  # (N, 30, 1, 251)
    x = Conv2D(126, (1, 1), use_bias=False)(x)      # (N, 30, 1, 126)

    x = Permute((1, 3, 2))(x)                       # (N, 32, 126, 1)
    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v126c(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        251->125
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 DFT-like
    ##################################################################
    x = Permute((1, 3, 2))(input1)                  # (N, 30, 1, 251)
    x = Conv2D(126, (1, 1), use_bias=False)(x)      # (N, 30, 1, 126)

    x = Permute((1, 3, 2))(x)                       # (N, 32, 126, 1)
    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, stride=(1, 2), padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


class Sqrt(Layer):
    def __init__(self, **kwargs):
        super(Sqrt, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        super(Sqrt, self).build(input_shape)

    def call(self, inputs):
        return K.sqrt(inputs)

    def get_config(self):
        config = super(Sqrt, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


def EEGNet_v126d(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        251->125
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 DFT-like
    ##################################################################
    x = Permute((1, 3, 2))(input1)             # (N, 30, 1, 251)
    x1 = Conv2D(126, (1, 1), use_bias=False, kernel_constraint=UnitNorm(axis=[0, 1, 2]))(x)      # (N, 30, 1, 126)
    x1 = Permute((1, 3, 2))(x1)                # (N, 32, 126, 1)

    x2 = Conv2D(126, (1, 1), use_bias=False, kernel_constraint=UnitNorm(axis=[0, 1, 2]))(x)      # (N, 30, 1, 126)
    x2 = Permute((1, 3, 2))(x2)                # (N, 32, 126, 1)

    x = Multiply()([x1, x1]) + Multiply()([x2, x2])
    x = Sqrt()(x)

    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v126f(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        FFT-like
        251->125
        BatchNormalization()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 DFT-like
    ##################################################################
    x = Permute((1, 3, 2))(input1)                  # (N, 30, 1, 251)
    x = Conv2D(126, (1, 1), use_bias=False)(x)      # (N, 30, 1, 126)

    x = Permute((1, 3, 2))(x)                       # (N, 32, 126, 1)
    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (7, 7), use_bias=False, strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


class ZeroPadding2D(Layer):
    def __init__(self, real_imag=True, **kwargs):
        super(ZeroPadding2D, self).__init__(**kwargs)
        self.real_imag = real_imag

    def build(self, input_shape):
        super(ZeroPadding2D, self).build(input_shape)

    def call(self, x):
        if self.real_imag:
            res = K.concatenate([x, K.zeros_like(x)], axis=-1)
        else:
            res = K.concatenate([K.zeros_like(x), x], axis=-1)
        return res

    def get_config(self):
        config = {
            'real_imag': self.real_imag
        }

        base_config = super(ZeroPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = output_shape[-1] * 2
        output_shape = tuple(output_shape)
        return output_shape


class OrthoRegularizer(Regularizer):
    def __init__(self, gamma=0.001):
        self.gamma = gamma

    def __call__(self, x):
        x_shape = K.int_shape(x)
        hadamard_product = x[:, :, 0:x_shape[2]//2, :] * x[:, :, -x_shape[2]//2:, :]
        regularization = self.gamma * K.sum(K.square(K.sum(hadamard_product, axis=2)))

        return regularization

    def get_config(self):
        # config = {
        #     'real_imag': self.real_imag
        # }
        # base_config = super(ZeroPadding2D, self).get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return {'gamma': float(self.gamma)}


def EEGNet_v127(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        DFT-like
        251->125
        BatchNormalization()
        OrthoRegularizer()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 DFT-like
    ##################################################################
    conv1x1 = Conv2D(126, (1, 1), input_shape=(30, 1, 502), use_bias=False, kernel_regularizer=OrthoRegularizer(0.001))    # (N, 30, 1, 126)

    x = Permute((1, 3, 2))(input1)              # (N, 30, 1, 251)
    x1 = ZeroPadding2D(real_imag=True)(x)       # (N, 30, 1, 502)
    x2 = ZeroPadding2D(real_imag=False)(x)

    x1 = conv1x1(x1)                            # (N, 30, 1, 126)
    x1 = Permute((1, 3, 2))(x1)                 # (N, 32, 126, 1)

    x2 = conv1x1(x2)                            # (N, 30, 1, 126)
    x2 = Permute((1, 3, 2))(x2)                 # (N, 32, 126, 1)

    x = Multiply()([x1, x1]) + Multiply()([x2, x2])
    x = Sqrt()(x)

    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v127a(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        DFT-like
        251->125
        BatchNormalization()
        OrthoRegularizer()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 DFT-like
    ##################################################################
    conv1x1 = Conv2D(126, (1, 1), input_shape=(30, 1, 502), use_bias=False, kernel_regularizer=OrthoRegularizer(0.001))    # (N, 30, 1, 126)

    x = Permute((1, 3, 2))(input1)              # (N, 30, 1, 251)

    # real
    x1 = ZeroPadding2D(real_imag=True)(x)       # (N, 30, 1, 502)
    x1 = conv1x1(x1)                            # (N, 30, 1, 126)
    x1 = Permute((1, 3, 2))(x1)                 # (N, 32, 126, 1)

    # imaginary
    x2 = ZeroPadding2D(real_imag=False)(x)
    x2 = conv1x1(x2)                            # (N, 30, 1, 126)
    x2 = Permute((1, 3, 2))(x2)                 # (N, 32, 126, 1)

    # abs
    x = Multiply()([x1, x1]) + Multiply()([x2, x2])
    x = Sqrt()(x)

    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v127b(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        DFT-like
        251->125
        BatchNormalization()
        kernel_regularizer=OrthoRegularizer()
        kernel_constraint=UnitNorm()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 DFT-like
    ##################################################################
    conv1x1 = Conv2D(126, (1, 1), input_shape=(30, 1, 502), use_bias=False,
                     kernel_constraint=UnitNorm(),
                     kernel_regularizer=OrthoRegularizer(0.001))    # (N, 30, 1, 126)

    x = Permute((1, 3, 2))(input1)              # (N, 30, 1, 251)

    # real
    x1 = ZeroPadding2D(real_imag=True)(x)       # (N, 30, 1, 502)
    x1 = conv1x1(x1)                            # (N, 30, 1, 126)
    x1 = Permute((1, 3, 2))(x1)                 # (N, 32, 126, 1)

    # imaginary
    x2 = ZeroPadding2D(real_imag=False)(x)
    x2 = conv1x1(x2)                            # (N, 30, 1, 126)
    x2 = Permute((1, 3, 2))(x2)                 # (N, 32, 126, 1)

    # abs
    x = Multiply()([x1, x1]) + Multiply()([x2, x2])
    x = Sqrt()(x)

    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v127c(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        DFT-like
        251->125
        BatchNormalization()
        kernel_regularizer=OrthoRegularizer()
        kernel_constraint=UnitNorm()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 DFT-like
    ##################################################################
    conv1x1 = Conv2D(126, (1, 1), input_shape=(30, 1, 502), use_bias=False,
                     kernel_constraint=UnitNorm(),
                     kernel_regularizer=OrthoRegularizer(0.01))    # (N, 30, 1, 126)

    x = Permute((1, 3, 2))(input1)              # (N, 30, 1, 251)

    # real
    x1 = ZeroPadding2D(real_imag=True)(x)       # (N, 30, 1, 502)
    x1 = conv1x1(x1)                            # (N, 30, 1, 126)
    x1 = Permute((1, 3, 2))(x1)                 # (N, 32, 126, 1)

    # imaginary
    x2 = ZeroPadding2D(real_imag=False)(x)
    x2 = conv1x1(x2)                            # (N, 30, 1, 126)
    x2 = Permute((1, 3, 2))(x2)                 # (N, 32, 126, 1)

    # abs
    x = Multiply()([x1, x1]) + Multiply()([x2, x2])
    x = Sqrt()(x)

    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v127d(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        DFT-like
        251->125
        BatchNormalization()
        kernel_regularizer=OrthoRegularizer()
        kernel_constraint=UnitNorm()

    7x7, 3x3
    SpatialDropout2D() [0.4, 0.4]
    """

    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, 251, 1))

    # Stage 0 DFT-like
    ##################################################################
    conv1x1 = Conv2D(126, (1, 1), input_shape=(30, 1, 502), use_bias=False,
                     # kernel_constraint=UnitNorm(),
                     kernel_regularizer=OrthoRegularizer(0.01))    # (N, 30, 1, 126)

    x = Permute((1, 3, 2))(input1)              # (N, 30, 1, 251)

    # real
    x1 = ZeroPadding2D(real_imag=True)(x)       # (N, 30, 1, 502)
    x1 = conv1x1(x1)                            # (N, 30, 1, 126)
    x1 = Permute((1, 3, 2))(x1)                 # (N, 32, 126, 1)

    # imaginary
    x2 = ZeroPadding2D(real_imag=False)(x)
    x2 = conv1x1(x2)                            # (N, 30, 1, 126)
    x2 = Permute((1, 3, 2))(x2)                 # (N, 32, 126, 1)

    # abs
    x = Multiply()([x1, x1]) + Multiply()([x2, x2])
    x = Sqrt()(x)

    x = BatchNormalization(axis=-1)(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (7, 7), use_bias=False, strides=(1, 2), padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(0.4)(x)

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (3, 3), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)     # 0.5

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130a(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130as(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # # 奇怪的卷积
    # x = Conv2D(16, (30, 1), use_bias=False, padding='same')(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130b(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130c(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    x = Permute((1, 3, 2))(input1)                  # (N, 30, 1, 251)
    x = Conv2D(251, (1, 1), use_bias=False)(x)      # (N, 30, 1, 251)
    x = Permute((1, 3, 2))(x)                       # (N, 32, 251, 1)
    x = BatchNormalization(axis=-1)(x)

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(x)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130d(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    x = Permute((1, 3, 2))(input1)                  # (N, 30, 1, 251)
    x = Conv2D(251, (1, 1), use_bias=False, kernel_constraint=max_norm(2.0, axis=[0, 1, 2]))(x)      # (N, 30, 1, 251)

    x = Permute((1, 3, 2))(x)                       # (N, 32, 251, 1)
    x = BatchNormalization(axis=-1)(x)

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(x)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130e(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    x = Permute((1, 3, 2))(input1)                  # (N, 30, 1, 251)
    x = Conv2D(251, (1, 1), use_bias=False, kernel_constraint=max_norm(2.0, axis=[0, 1, 2]))(x)      # (N, 30, 1, 251)

    x = Permute((1, 3, 2))(x)                       # (N, 32, 251, 1)
    x = BatchNormalization(axis=-1)(x)

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(x)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130f(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130k(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    x = Permute((1, 3, 2))(input1)                  # (N, 30, 1, 251)
    x = Conv2D(125, (1, 1), use_bias=False)(x)      # (N, 30, 1, 251)
    x = Permute((1, 3, 2))(x)                       # (N, 32, 251, 1)
    x = BatchNormalization(axis=-1)(x)

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 25), padding='same', name='temporal',
               use_bias=False)(x)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v130m(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    x = Permute((1, 3, 2))(input1)                  # (N, 30, 1, 251)
    x = Conv2D(125, (1, 1), use_bias=False)(x)      # (N, 30, 1, 251)
    x = Permute((1, 3, 2))(x)                       # (N, 32, 251, 1)
    x = BatchNormalization(axis=-1)(x)

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(x)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v135(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0, r=4):
        """
        通道注意力机制. channels last!
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    se1 = SE(32, 1)(x)
    x = Multiply()([x, se1])

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v135b(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same'):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        return model_

    def SE(input_dim, name_sn=0):
        """
        通道注意力机制. channels last!
        :param input_dim:
        :param name_sn:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//4, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    cnn_kernel_size = (1, 15)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = conv_block(filters=32, kernel_size=(1, 15), n_conv=2, padding='same')(x)

    se1 = SE(32, 1)(x)
    x = Multiply()([x, se1])

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = conv_block(filters=64, kernel_size=(1, 15), n_conv=2, padding='same')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = conv_block(filters=128, kernel_size=(1, 15), n_conv=2, padding='same')(x)
    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = conv_block(filters=256, kernel_size=(1, 15), n_conv=1, padding='same')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v135c(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0, r=2):
        """
        通道注意力机制. channels last!
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         # Activation('sigmoid')],
                         Activation('softmax')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    se1 = SE(32, 1)(x)
    x = Multiply()([x, se1])

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v136(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(8, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D [0.4, 0.4]
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0, r=2):
        """
        通道注意力机制. channels last!
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         # Activation('sigmoid')],
                         Activation('softmax')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################

    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v137(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(8, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D [0.4, 0.4]
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0, r=2):
        """
        通道注意力机制. channels last!
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         # Activation('sigmoid')],
                         Activation('softmax')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################

    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v137b(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D [0.4, 0.4]
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0, r=2):
        """
        通道注意力机制. channels last!
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         # Activation('sigmoid')],
                         Activation('softmax')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################

    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(16, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v137c(nb_classes=2, samples=251, l2_weight=0.001):
    """
    Stage 0
        Conv2D(8, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D [0.4, 0.4]
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def SE(input_dim, name_sn=0, r=2):
        """
        通道注意力机制. channels last!
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                # kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         # Activation('sigmoid')],
                         Activation('softmax')],
                        name='SE' + str(name_sn))
        return sq

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################

    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v205(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.6)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.6)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v205b(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def dw_conv_block(kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        model_.add(DepthwiseConv2D(kernel_size, use_bias=False,
                                   depth_multiplier=2, kernel_regularizer=regularizers.l2(l2_weight)))
        model_.add(BatchNormalization())
        model_.add(Activation('elu'))

        for _ in range(n_conv-1):
            model_.add(DepthwiseConv2D(kernel_size, use_bias=False,
                                       depth_multiplier=1, kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def separable_conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False,
                                       kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    #x = MaxPooling2D((1, 2))(x)
    x = dw_conv_block((1, 15), n_conv=1, pooling=MaxPooling2D((1, 2)))(x)

    # Stage 2
    ##################################################################

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = MaxPooling2D((1, 2))(x)

    x = dw_conv_block((1, 15), n_conv=1, pooling=MaxPooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.6)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v205c(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def dw_conv_block(kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        model_.add(DepthwiseConv2D(kernel_size, use_bias=False,
                                   depth_multiplier=2, kernel_regularizer=regularizers.l2(l2_weight)))
        model_.add(BatchNormalization())
        model_.add(Activation('elu'))

        for _ in range(n_conv-1):
            model_.add(DepthwiseConv2D(kernel_size, use_bias=False,
                                       depth_multiplier=1, kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def separable_conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False,
                                       kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = MaxPooling2D((1, 2))(x)

    x = separable_conv_block(32, (1, 15), n_conv=1, pooling=MaxPooling2D((1, 2)))(x)

    # Stage 2
    ##################################################################

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = MaxPooling2D((1, 2))(x)

    x = separable_conv_block(32, (1, 15), n_conv=1, pooling=MaxPooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################

    x = separable_conv_block(128, (1, 15), n_conv=1, pooling=MaxPooling2D((1, 2)))(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.6)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = separable_conv_block(256, (1, 15), n_conv=1)(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v203(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v203b(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v203c(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(48, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(96, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v203d(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(48, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.6)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)
    x = SpatialDropout2D(0.6)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v204(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def separable_conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False,
                                       kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = AveragePooling2D((1, 2))(x)

    x = separable_conv_block(32, (1, 15), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(48, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = separable_conv_block(48, (1, 15), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = AveragePooling2D((1, 2))(x)

    x = separable_conv_block(48, (1, 15), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)        # SpatialDropout2D

    # Stage 3
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)

    x = separable_conv_block(64, (1, 15), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)        # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v204b(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def separable_conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False,
                                       kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    kernel = 15

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = AveragePooling2D((1, 2))(x)

    x = separable_conv_block(32, (1, kernel), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(48, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = AveragePooling2D((1, 2))(x)

    x = separable_conv_block(48, (1, kernel), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)        # SpatialDropout2D

    # Stage 3
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)

    x = separable_conv_block(64, (1, kernel), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)        # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v204c(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def separable_conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False,
                                       kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    kernel = 15

    # Stage 1
    ##################################################################
    x = separable_conv_block(32, (1, kernel), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)

    # Stage 2
    ##################################################################
    x = separable_conv_block(48, (1, kernel), n_conv=2, pooling=AveragePooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)        # SpatialDropout2D

    # Stage 3
    ##################################################################
    x = separable_conv_block(64, (1, kernel), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)        # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v204f(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    def separable_conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False,
                                       kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = AveragePooling2D((1, 2))(x)

    x = separable_conv_block(32, (1, 15), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(48, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = separable_conv_block(48, (1, 15), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)
    #
    # # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # # x = AveragePooling2D((1, 2))(x)
    #
    # x = separable_conv_block(48, (1, 15), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)        # SpatialDropout2D

    # Stage 3
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)

    x = separable_conv_block(64, (1, 15), n_conv=1, pooling=AveragePooling2D((1, 2)))(x)
    x = SpatialDropout2D(0.6)(x)        # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v401(nb_classes=2, samples=251, l2_weight=0.001):
    """
    SpatialDropout2D(0.4)

    Stage 0
        Conv2D(16, (1, 125))
        depthwise_constraint=max_norm()

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    1x15
    SpatialDropout2D
    samples=501
    """

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    def conv_block(filters, kernel_size, n_conv=1, padding='same', pooling=None):
        model_ = Sequential()
        for _ in range(n_conv):
            model_.add(Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                              kernel_regularizer=regularizers.l2(l2_weight)))
            model_.add(BatchNormalization())
            model_.add(Activation('elu'))

        if pooling is not None:
            model_.add(pooling)

        return model_

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    #  ~Unit sample response, "adaptive filter"
    x = Conv2D(8, (1, 125), padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # ~Spatial filter in Xdawn
    x = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                        depth_multiplier=2,
                        depthwise_constraint=max_norm(1.0))(x)      # == max_norm(1.0， axis=0) UnitNorm
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 4))(x)     # (1, 2)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)
    x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)
    # x = SpatialDropout2D(0.4)(x)            # SpatialDropout2D

    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(256, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # Stage Flatten
    ##################################################################
    x = GlobalAveragePooling2D()(x)
    x = Flatten(name='flatten')(x)
    x = Dropout(0.6)(x)                     # Dropout(0.5)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    return model


def EEGNet_v104f(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(60, 251, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x1 = Slice(0, 30)(x)
    x2 = Slice(30, 60)(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))      # == max_norm(1.0， axis=0)
    x3 = dw_conv(x1)
    x3 = BatchNormalization()(x3)
    x4 = dw_conv(x2)
    x4 = BatchNormalization()(x4)

    inception = [x3, x4]
    x = Concatenate(axis=-3)(inception)

    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)     # (1, 2)

    kern_w = 15

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (2, 2)
    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((2, 2))(x)     # (1, 2)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v414(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(60, 251, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x1 = Slice(0, 30)(x)
    x2 = Slice(30, 60)(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))      # == max_norm(1.0， axis=0)
    x3 = dw_conv(x1)
    x3 = BatchNormalization()(x3)
    x4 = dw_conv(x2)
    x4 = BatchNormalization()(x4)

    inception = [x3, x4]
    x = Concatenate(axis=-1)(inception)

    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)     # (1, 2)

    kern_w = 15

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)

    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v415(nb_classes=2, kern_length=125, strides=(1, 1), l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(60, 251, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    x1 = Slice(0, 30)(x)
    x2 = Slice(30, 60)(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))      # == max_norm(1.0， axis=0)

    dw_conv_av = DepthwiseConv2D((30, 1), use_bias=False, name='spatial2',
                                 depth_multiplier=2,
                                 depthwise_constraint=max_norm(1.0))      # == max_norm(1.0， axis=0)
    x3 = dw_conv(x1)
    x3 = BatchNormalization()(x3)
    x4 = dw_conv(x2)
    x4 = BatchNormalization()(x4)

    inception = [x3, x4]
    x = Concatenate(axis=-1)(inception)
    x = Activation('elu')(x)

    # x = Conv2D(16, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)     # (1, 2)

    kern_w = 15

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)

    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v420(nb_classes=2, kern_length=125, strides=(1, 1), samples = None, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # x1 = Slice(0, 30)(x)
    # x2 = Slice(30, 60)(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))      # == max_norm(1.0， axis=0)

    # dw_conv_av = DepthwiseConv2D((30, 1), use_bias=False, name='spatial2',
    #                              depth_multiplier=2,
    #                              depthwise_constraint=max_norm(1.0))      # == max_norm(1.0， axis=0)
    # x3 = dw_conv(x1)
    # x3 = BatchNormalization()(x3)
    # x4 = dw_conv(x2)
    # x4 = BatchNormalization()(x4)
    #
    # inception = [x3, x4]
    # x = Concatenate(axis=-1)(inception)

    x = dw_conv(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(16, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)     # (1, 2)

    kern_w = 15

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)

    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v430(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=None, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, channels))

    # Stage -1
    ##################################################################
    x2 = Multiply()([input1, input1])
    se1 = SE(channels, r=2)(x2)

    x = Multiply()([input1, se1])

    x = Conv2D(1, (1, 1), strides=strides, padding='same', name='norm',
               use_bias=False)(input1)
    x = BatchNormalization()(x)

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal',
               use_bias=False)(x)
    x = BatchNormalization()(x)

    # x1 = Slice(0, 30)(x)
    # x2 = Slice(30, 60)(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))      # == max_norm(1.0， axis=0)

    # dw_conv_av = DepthwiseConv2D((30, 1), use_bias=False, name='spatial2',
    #                              depth_multiplier=2,
    #                              depthwise_constraint=max_norm(1.0))      # == max_norm(1.0， axis=0)
    # x3 = dw_conv(x1)
    # x3 = BatchNormalization()(x3)
    # x4 = dw_conv(x2)
    # x4 = BatchNormalization()(x4)
    #
    # inception = [x3, x4]
    # x = Concatenate(axis=-1)(inception)

    x = dw_conv(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(16, (1, 1), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)     # (1, 2)

    kern_w = 15

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)

    #x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)

    # x = MaxPooling2D((1, 4))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


class Slice(Layer):
    """
    切片层. 按照指定的axis, 切片指定的部位.
    # Arguments
    # Input shape
    # Output shape
    """
    def __init__(self,
                 start,
                 end,
                 axis=-1,
                 **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.start = start
        self.end = end
        self.axis = axis

    def build(self, input_shape):
        super(Slice, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        # broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        if self.axis == 1:
            outputs = inputs[:, self.start:self.end, :, :]
        elif self.axis == 2:
            outputs = inputs[:, :, self.start:self.end, :]
        else:
            outputs = inputs[:, :, :, self.start:self.end]

        return outputs

    def get_config(self):
        config = {
            'start': self.start,
            'end': self.end,
            'axis': self.axis
        }

        base_config = super(Slice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.end - self.start
        return tuple(output_shape)


def EEGNet_v440(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=None, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, channels))

    kern_w = 15

    # Stage 0
    ##################################################################
    shared_filter = ts_filter()
    xs = []
    for n in range(channels):
        xn = Slice(n, n+1, axis=-1)(input1)
        xn = shared_filter(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)
    # x = Maximum()(xs)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)

    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)

    # x = MaxPooling2D((1, 4))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v440b(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=None, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, channels))

    kern_w = 15

    # Stage 0
    ##################################################################
    shared_filter = ts_filter()
    xs = []
    for n in range(channels):
        xn = Slice(n, n+1, axis=-1)(input1)
        xn = shared_filter(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)
    # x = Maximum()(xs)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)

    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)
    #
    # x = MaxPooling2D((1, 2))(x)
    #
    # # Stage 4
    # ##################################################################
    # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    #
    # # x = Conv2D(256, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # # x = BatchNormalization()(x)
    # # x = Activation('elu')(x)

    # x = MaxPooling2D((1, 4))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    #opt = SGD(lr=0.001, momentum=0.9)
    #model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v442(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v442c(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v442d(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (2, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(64, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = MaxPooling2D((4, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v442f(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (2, 2)
    # x = SpatialDropout2D(0.4)(x)
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = MaxPooling2D((4, 2))(x)     # (1, 2)
    # x = SpatialDropout2D(0.4)(x)
    x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v442h(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((4, 2))(x)     # (2, 2)
    # x = SpatialDropout2D(0.4)(x)
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = MaxPooling2D((4, 2))(x)     # (1, 2)
    # x = SpatialDropout2D(0.4)(x)
    x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v450(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((1, 2))(x)     # (2, 2)
    # x = MaxPooling2D((channels, 1))(x)    # try
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = MaxPooling2D((channels, 1))(x)      # try
    x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v450b(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = AveragePooling2D((4, 2))(x)     # (4, 2)
    # x = MaxPooling2D((channels, 1))(x)    # try
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # # x = MaxPooling2D((channels, 1))(x)      # try
    # x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v450c(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (4, 2)
    # x = MaxPooling2D((channels, 1))(x)    # try
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # # x = MaxPooling2D((channels, 1))(x)      # try
    # x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v450f(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    # x = AveragePooling2D((1, 4))(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (4, 2)
    # x = MaxPooling2D((channels, 1))(x)    # try
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # # x = MaxPooling2D((channels, 1))(x)      # try
    # x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v450h(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xn = BatchNormalization()(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    # x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    # x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (4, 2)
    # x = MaxPooling2D((channels, 1))(x)    # try
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # # x = MaxPooling2D((channels, 1))(x)      # try
    # x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v450k(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    kern_w = 15

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    bn = BatchNormalization()

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xn = bn(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    # x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    # x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (4, 2)
    # x = MaxPooling2D((channels, 1))(x)    # try
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # # x = MaxPooling2D((channels, 1))(x)      # try
    # x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v450m(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from EEGNet_v104f
    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = Conv2D(16, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((4, 2))(x)     # (4, 2)
    x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    x = Conv2D(32, (1, 5), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # # x = MaxPooling2D((channels, 1))(x)      # try
    # x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452mgpu(nb_classes=2, kern_length=125, strides=(1, 1), samples=251, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average
    from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    if channels == 1:
        x = dw_conv(x)
    else:
        xs = []
        for n in range(0, 30*channels, 30):
            xn = Slice(n, n+30, axis=1)(x)
            xn = dw_conv(xn)
            xs.append(xn)
        x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    mgpu_model = multi_gpu_model(model, gpus=2)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    mgpu_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return mgpu_model


def EEGNet_v452(nb_classes=2, kern_length=125, strides=(1, 1), samples=251, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    if channels == 1:
        x = dw_conv(x)
    else:
        xs = []
        for n in range(0, 30*channels, 30):
            xn = Slice(n, n+30, axis=1)(x)
            xn = dw_conv(xn)
            xs.append(xn)
        x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452_II(nb_classes=2, kern_length=125, strides=(1, 1), samples=251, channels=4, l2_weight=0.001):
    """
    II
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    # x = input1

    # 时域滤波器, 共享参数
    fir_conv = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)

    # 空间(电极)滤波器, 共享参数
    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    if channels == 1:
        # x = fir_conv(x)
        # x = BatchNormalization()(x)
        x = dw_conv(x)
    else:
        xs = []
        for n in range(0, 30*channels, 30):
            xn = Slice(n, n+30, axis=1)(x)

            # xn = fir_conv(xn)                   # II
            # xn = BatchNormalization()(xn)       # II
            xn = dw_conv(xn)
            xn = BatchNormalization()(xn)       # II

            xs.append(xn)
        x = Concatenate(axis=1)(xs)             # (N, channels, 251, 16)

    # x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452_III(nb_classes=2, kern_length=125, strides=(1, 1), samples=251, channels=4, l2_weight=0.001):
    """
    III
    x = SeparableConv2D(64, (1, 7)

    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    # x = input1

    # 时域滤波器, 共享参数
    fir_conv = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)

    # 空间(电极)滤波器, 共享参数
    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    if channels == 1:
        x = dw_conv(x)
    else:
        xs = []
        for n in range(0, 30*channels, 30):
            xn = Slice(n, n+30, axis=1)(x)
            xn = dw_conv(xn)
            xs.append(xn)
        x = Concatenate(axis=1)(xs)             # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2 III
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 7), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452avg(nb_classes=2, kern_length=125, strides=(1, 1), samples=251, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    if channels == 1:
        x = dw_conv(x)
    else:
        xs = []
        for n in range(0, 30*channels, 30):
            xn = Slice(n, n+30, axis=1)(x)
            xn = dw_conv(xn)
            xs.append(xn)
        x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)
    # x = MaxPooling2D((1, 2))(x)
    # x = AveragePooling2D((channels, 1))(x)

    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452b(nb_classes=2, kern_length=125, strides=(1, 1), samples=251, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    if channels == 1:
        x = dw_conv(x)
    else:
        xs = []
        for n in range(0, 30*channels, 30):
            xn = Slice(n, n+30, axis=1)(x)
            xn = dw_conv(xn)
            xs.append(xn)
        x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = SeparableConv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452c(nb_classes=2, kern_length=125, strides=(1, 1), samples=251, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    if channels == 1:
        x = dw_conv(x)
    else:
        xs = []
        for n in range(0, 30*channels, 30):
            xn = Slice(n, n+30, axis=1)(x)
            xn = dw_conv(xn)
            xs.append(xn)
        x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    # x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(1.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452e(nb_classes=2, kern_length=125, strides=(1, 1), samples=251, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    if channels == 1:
        x = dw_conv(x)
    else:
        xs = []
        for n in range(0, 30*channels, 30):
            xn = Slice(n, n+30, axis=1)(x)
            xn = dw_conv(xn)
            xs.append(xn)
        x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model



def EEGNet_v453(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)
    # x = MaxPooling2D((channels, 1))(x)
    # x = MaxPooling2D((1, 2))(x)

    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    x = Conv2D(128, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def EEGNet_v470(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 4))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v470b(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)     # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 4))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v502(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30*channels, 251, 1))

    # Stage -1
    x = BatchNormalization()(input1)

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    xs = []
    for n in range(0, 30*channels, 30):
        xn = Slice(n, n+30, axis=1)(x)
        xn = dw_conv(xn)
        xs.append(xn)
    x = Concatenate(axis=1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v560(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, 251, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    conv_2d = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        # xn = conv_2d(xn)
        xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v560b(nb_classes=2, kern_length=125, strides=(1, 1), samples=None, channels=4, l2_weight=0.001):
    """
    共享 conv_2d(spatial filter)
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, 251, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=strides, padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
                              depth_multiplier=2,
                              depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    # conv_2d = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)
    shared_conv_2d = Conv2D(4, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        xn = shared_conv_2d(xn)
        # xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v560c(nb_classes=2, samples=251, channels=4, l2_weight=0.001):
    """
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, samples, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        xn = shared_conv_2d(xn)
        # xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v560d(nb_classes=2, samples=251, channels=4, l2_weight=0.001):
    """
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, samples, 1))

    x = input1

    # Stage -1
    x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        xn = shared_conv_2d(xn)
        # xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v560e(nb_classes=2, samples=251, channels=4, l2_weight=0.001):
    """
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, samples, 1))

    x = input1

    # Stage -1
    x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        xn = shared_conv_2d(xn)
        # xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = MaxPooling2D((1, 2))(x)             # v560e
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v563(nb_classes=2, samples=251, channels=4, l2_weight=0.001):
    """
    from EEGNet_v560c
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, samples, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(4, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        xn = shared_conv_2d(xn)
        # xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v570(nb_classes=2, samples=251, channels=4, l2_weight=0.001):
    """
    from EEGNet_v560c
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, samples, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(4, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        # xn = shared_conv_2d(xn)
        xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # x = GlobalAveragePooling2D()(x)
    x = GlobalMaxPooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v571(nb_classes=2, samples=251, channels=4, l2_weight=0.001):
    """
    from EEGNet_v560c
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, samples, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(4, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        # xn = shared_conv_2d(xn)
        xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    x = Concatenate()([x1, x2])

    # Stage Flatten
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v580(nb_classes=2, samples=501, channels=4, l2_weight=0.001):
    """
    501: valid
    from EEGNet_v560c
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, None, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='valid', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(4, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        # xn = shared_conv_2d(xn)
        xn = Conv2D(1, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 5), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 5), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    x = Concatenate()([x1, x2])

    # Stage Flatten
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v580b(nb_classes=2, samples=501, channels=4, l2_weight=0.001):
    """
    501: valid
    from EEGNet_v560c
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, None, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='valid', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(4, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        # xn = shared_conv_2d(xn)
        xn = Conv2D(1, (30, 1), strides=(30, 1), padding='valid',  use_bias=False,
                    kernel_regularizer=regularizers.l2(l2_weight)
                    )(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 5), use_bias=True, padding='same',
                        depthwise_regularizer=regularizers.l2(l2_weight),
                        pointwise_regularizer=regularizers.l2(l2_weight),
                        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 5), use_bias=True, padding='same',
                        depthwise_regularizer=regularizers.l2(l2_weight),
                        pointwise_regularizer=regularizers.l2(l2_weight),
                        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    x = Concatenate()([x1, x2])

    # Stage Flatten
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', use_bias=True, kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v580c(nb_classes=2, samples=501, channels=4, l2_weight=0.001):
    """
    501: valid
    from EEGNet_v560c
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, None, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='valid', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(4, (30, 1), strides=(30, 1), padding='valid',  use_bias=False)

    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        # xn = shared_conv_2d(xn)
        xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False,
                    kernel_regularizer=regularizers.l2(l2_weight)
                    )(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 5), use_bias=True, padding='same',
                        depthwise_regularizer=regularizers.l2(l2_weight),
                        pointwise_regularizer=regularizers.l2(l2_weight),
                        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 5), use_bias=True, padding='same',
                        depthwise_regularizer=regularizers.l2(l2_weight),
                        pointwise_regularizer=regularizers.l2(l2_weight),
                        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    x = Concatenate()([x1, x2])

    # Stage Flatten
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', use_bias=True, kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v580d(nb_classes=2, samples=501, channels=4, l2_weight=0.001):
    """
    501: valid
    from EEGNet_v560c
    共享 conv_2d(spatial filter) 替代 DepthwiseConv2D
    from v452
    第一层添加 BatchNormalization()
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(None, None, 1))

    x = input1

    # Stage -1
    # x = BatchNormalization()(x)

    # Stage 0
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='valid', name='temporal', use_bias=False)(x)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    shared_conv_2d = Conv2D(4, (30, 1), strides=(30, 1), padding='valid',  use_bias=False,
                            kernel_regularizer=regularizers.l2(l2_weight))
    xs = []
    for n in range(8):
        xn = Slice(n, n+1, axis=-1)(x)
        xn = shared_conv_2d(xn)
        # xn = Conv2D(2, (30, 1), strides=(30, 1), padding='valid',  use_bias=False,
        #             kernel_regularizer=regularizers.l2(l2_weight)
        #             )(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, channels, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 5), use_bias=True, padding='same',
                        depthwise_regularizer=regularizers.l2(l2_weight),
                        pointwise_regularizer=regularizers.l2(l2_weight),
                        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((channels, 2))(x)      # (4, 2)
    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 5), use_bias=True, padding='same',
                        depthwise_regularizer=regularizers.l2(l2_weight),
                        pointwise_regularizer=regularizers.l2(l2_weight),
                        )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)             # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    x1 = GlobalAveragePooling2D()(x)
    x2 = GlobalMaxPooling2D()(x)
    x = Concatenate()([x1, x2])

    # Stage Flatten
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    x = Dense(nb_classes, name='fc_softmax', use_bias=True, kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def gru(input_shape=(7, 9), l2_lambda=0.001):
    """
    :param input_shape:
    :param l2_lambda:
    :return:
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, GRU, Activation, Dropout, BatchNormalization, Reshape, DepthwiseConv2D
    from tensorflow.keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Permute
    from tensorflow.keras.layers import TimeDistributed
    from tensorflow.keras import regularizers
    from tensorflow.keras.constraints import max_norm

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=input_shape)
    x = input1

    # x = TimeDistributed(Dense(32, kernel_regularizer=regularizers.l2(l2_lambda)))(x)
    # x = TimeDistributed(BatchNormalization())(x)
    # x = TimeDistributed(Activation('elu'))(x)

    # x = GRU(units=64,
    #         # dropout=0.6,
    #         # recurrent_dropout=0.2,
    #         # kernel_constraint=max_norm(1.0),
    #         # recurrent_constraint=max_norm(1.0),
    #         # kernel_regularizer=regularizers.l2(l2_lambda),
    #         # recurrent_regularizer=regularizers.l2(l2_lambda),
    #         return_sequences=True)(x)

    x = Bidirectional(GRU(units=64,
                          # dropout=0.4,
                          # recurrent_dropout=0.2,
                          # kernel_constraint=max_norm(1.0),
                          # recurrent_constraint=max_norm(1.0),
                          # kernel_regularizer=regularizers.l2(l2_lambda),
                          # recurrent_regularizer=regularizers.l2(l2_lambda),
                          return_sequences=True))(x)   # False: 只有最后一个单元输出

    x = Bidirectional(GRU(units=64,
                          # dropout=0.4,
                          # recurrent_dropout=0.2,
                          # kernel_constraint=max_norm(1.0),
                          # recurrent_constraint=max_norm(1.0),
                          # kernel_regularizer=regularizers.l2(l2_lambda),
                          # recurrent_regularizer=regularizers.l2(l2_lambda),
                          return_sequences=True))(x)   # False: 只有最后一个单元输出

    # x = TimeDistributed(Dense(128))(x)
    # x = TimeDistributed(BatchNormalization())(x)
    # x = TimeDistributed(Activation('elu'))(x)
    #
    x = GlobalMaxPooling1D()(x)         # return_sequences=True

    # # 加了效果不好
    # x = Dense(128, kernel_regularizer=regularizers.l2(l2_lambda))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Dropout(0.6)(x)                 # noise_shape=[1, 16]

    x = Dense(64, kernel_regularizer=regularizers.l2(l2_lambda))(x)#64
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.6)(x)

    # output1 = Dense(1, name='output1', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    output1 = Dense(1, name='output1')(x)#, use_bias=False

    model = Model(inputs=input1, outputs=output1)

    opt = Adam(lr=0.0002, clipnorm=0.01)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt, metrics=['mse'])
    # model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=['mse'])

    model.summary()

    return model


class SliceEpochs(Layer):
    """
    切片层. 按照指定的axis, 切片指定的部位.
    # Arguments
    # Input shape
    # Output shape
    """
    def __init__(self,
                 n_chs=30,
                 axis=1,
                 **kwargs):
        super(SliceEpochs, self).__init__(**kwargs)
        self.n_chs = n_chs
        self.axis = axis

    def build(self, input_shape):
        super(SliceEpochs, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        # broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        input_shape = K.int_shape(inputs)
        # print(input_shape)

        outputs = []
        for offset in range(0, input_shape[self.axis], self.n_chs):
            outputs.append(inputs[:, offset: offset+self.n_chs, :, :])

        return tuple(outputs)

    def get_config(self):
        config = {
            'n_chs': self.n_chs,
            'axis': self.axis
        }

        base_config = super(SliceEpochs, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.n_chs
        return tuple(output_shape)


class LayerSlice(Layer):
    """
    切片层. 按照指定的axis, 切片指定的部位.
    # Arguments
    # Input shape
    # Output shape
    """
    def __init__(self,
                 slice_step=30,
                 axis=1,
                 **kwargs):
        super(LayerSlice, self).__init__(**kwargs)
        self.slice_step = slice_step
        self.axis = axis

    def build(self, input_shape):
        super(LayerSlice, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        # broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        input_shape = K.int_shape(inputs)

        if self.axis < 0:
            self.axis = len(input_shape) + self.axis

        outputs = []
        for offset in range(0, input_shape[self.axis], self.slice_step):
            if self.axis == 1:
                outputs.append(inputs[:, offset: offset+self.slice_step, :, :])
            elif self.axis == 2:
                outputs.append(inputs[:, :, offset: offset+self.slice_step, :])
            else:
                outputs.append(inputs[:, :, :, offset: offset+self.slice_step])

        return tuple(outputs)

    def get_config(self):
        config = {
            'slice_step': self.slice_step,
            'axis': self.axis
        }

        base_config = super(LayerSlice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = self.slice_step
        return tuple(output_shape)


class SliceChannels(Layer):
    """
    切片层. 按照指定的axis, 切片指定的部位.
    # Arguments
    # Input shapeChannels
    # Output shape
    """
    def __init__(self,
                 axis=-1,
                 **kwargs):
        super(SliceChannels, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        super(SliceChannels, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        # broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        input_shape = K.int_shape(inputs)
        # print(input_shape)

        outputs = []
        for offset in range(0, input_shape[self.axis], 1):
            outputs.append(inputs[:, :, :, offset: offset+1])

        return tuple(outputs)

    def get_config(self):
        config = {
            'axis': self.axis
        }

        base_config = super(SliceChannels, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = 1
        return tuple(output_shape)


class MaxPoolingEEGChannel(Layer):
    """

    # Arguments
    # Input shape
    # Output shape
    """
    def __init__(self,
                 axis=1,
                 **kwargs):
        super(MaxPoolingEEGChannel, self).__init__(**kwargs)
        self.axis = axis

    def build(self, input_shape):
        super(MaxPoolingEEGChannel, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        # broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        outputs = K.max(inputs, axis=self.axis, keepdims=True)
        return outputs

    def get_config(self):
        config = {
            'axis': self.axis
        }

        base_config = super(MaxPoolingEEGChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[self.axis] = 1
        return tuple(output_shape)


class SPPNet(Layer):
    """

    # Arguments
    # Input shape
    # Output shape
    """
    def __init__(self,
                 output_size=(5, 5),
                 **kwargs):
        super(SPPNet, self).__init__(**kwargs)
        self.output_size = output_size

    def build(self, input_shape):
        super(SPPNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # input_shape = K.int_shape(inputs)
        # tensor_input_shape = K.shape(inputs)

        # # Prepare broadcasting shape.
        # reduction_axes = list(range(len(input_shape)))
        # del reduction_axes[self.axis]
        # broadcast_shape = [1] * len(input_shape)                            # (1, 1, 1, 1)
        # broadcast_shape[self.axis] = input_shape[self.axis] // self.groups  # (1, 1, 1, C//G)
        # broadcast_shape.insert(1, self.groups)                              # (1, G, 1, 1, C//G)
        #
        # reshape_group_shape = K.shape(inputs)
        # group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        # group_axes[self.axis] = input_shape[self.axis] // self.groups
        # group_axes.insert(1, self.groups)
        #
        # # reshape inputs to new group shape
        # group_shape = [group_axes[0], self.groups] + group_axes[2:]
        # group_shape = K.stack(group_shape)      # !!!!
        # inputs = K.reshape(inputs, group_shape)
        #
        # group_reduction_axes = list(range(len(group_axes)))
        # #mean, variance = KC.moments(inputs, group_reduction_axes[2:],
        # #                            keep_dims=True)
        # mean = K.mean(inputs, group_reduction_axes[2:], keepdims=True)
        # variance = K.var(inputs, group_reduction_axes[2:], keepdims=True)
        # inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        #
        # # prepare broadcast shape
        # inputs = K.reshape(inputs, group_shape)

        # tensor_input_shape = K.shape(inputs)
        # input_shape = K.int_shape(inputs)
        # input_shape = inputs.get_shape()
        input1 = inputs[0]
        input_shape = inputs[1]

        # input1 = input1[:, :input_shape[0], :input_shape[1], :]

        strides = (input_shape[0] // self.output_size[0], input_shape[1] // self.output_size[1])
        pool_size = (input_shape[0] - self.output_size[0]*strides[0],
                     input_shape[1] - self.output_size[1]*strides[1])

        outputs = K.pool2d(input1,
                           pool_size=pool_size,
                           strides=strides,
                           padding='valid',
                           data_format=None,
                           pool_mode='avg')
        return outputs

    def get_config(self):
        config = {
            'output_size': self.output_size
        }

        base_config = super(SPPNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        # output_shape[self.axis] = 1
        return tuple(output_shape)


def EEGNet_v452adp(nb_classes=2, samples=251, l2_weight=0.001):
    """
    自适应合成Example的epochs数量
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    def SE(input_dim, r=2, name_sn=0):
        """
        通道注意力机制.
        :param input_dim:
        :param name_sn:
        :param r:
        :return:
        """
        sq = Sequential([GlobalAveragePooling2D(),
                         Reshape((1, 1, input_dim)),
                         Conv2D(input_dim//r, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='squeeze' + str(name_sn)),
                         Activation('relu'),
                         Conv2D(input_dim, (1, 1), use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_weight),
                                name='excitation' + str(name_sn)),
                         Activation('sigmoid')],
                        name='SE' + str(name_sn))
        return sq

    def ts_filter(name_sn=0):
        sq = Sequential([Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False),
                         BatchNormalization(),
                         DepthwiseConv2D((30, 1), use_bias=False, name='spatial', depth_multiplier=2,
                                         depthwise_constraint=max_norm(1.0)),      # == max_norm(1.0， axis=0)
                         BatchNormalization(),
                         Activation('elu'),
                         MaxPooling2D((1, 4)),
                         Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight)),
                         BatchNormalization(),
                         Activation('elu')
                         ],
                        name='ts_filter' + str(name_sn))
        return sq

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    # input1 = Input(shape=(30*channels, samples, 1))
    input1 = Input(shape=(None, samples, 1))

    # Stage 0
    ##################################################################
    x = Conv2D(8, (1, 125), strides=(1, 1), padding='same', name='temporal', use_bias=False)(input1)
    x = BatchNormalization()(x)

    # dw_conv = DepthwiseConv2D((30, 1), use_bias=False, name='spatial',
    #                           depth_multiplier=2,
    #                           depthwise_constraint=max_norm(1.0))     # == max_norm(1.0， axis=0)

    groups = LayerSlice(slice_step=1, axis=-1)(x)

    xs = []
    for xn in groups:
        xn = Conv2D(2, (30, 1), strides=(30, 1), padding='same', use_bias=False, kernel_constraint=max_norm(1.0, axis=0))(xn)
        xs.append(xn)
    x = Concatenate(axis=-1)(xs)     # (N, None, 251, 16)

    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = MaxPooling2D((1, 4))(x)

    # Stage 1
    ##################################################################
    # x = Conv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(32, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = MaxPooling2D((channels, 2))(x)
    x = MaxPooling2D((1, 2))(x)
    x = MaxPoolingEEGChannel(axis=1)(x)

    x = SpatialDropout2D(0.4)(x)
    # x = Dropout(0.5)(x)

    # Stage 2
    ##################################################################
    # x = Conv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = SeparableConv2D(64, (1, 15), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = MaxPooling2D((1, 2))(x)     # (1, 2)
    x = SpatialDropout2D(0.4)(x)

    # Stage 3
    ##################################################################
    # x = Conv2D(128, (1, kern_w), use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)
    # x = MaxPooling2D((1, 2))(x)

    x = GlobalAveragePooling2D()(x)

    # Stage Flatten
    ##################################################################
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)

    # softmax
    ##################################################################
    x = Dense(nb_classes, name='fc_softmax', kernel_constraint=max_norm(0.5))(x)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def test_ssp_net(nb_classes=2, samples=251, l2_weight=0.001):
    """
    自适应合成Example的epochs数量
    SeparableConv2D 替代 Conv2D

    Stage 0
        depthwise_constraint=max_norm(1.0)
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)
    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(30, 251, 1))
    input2 = Input(shape=(2, ), dtype='int32')

    x = SPPNet((5, 6))(input1)

    output1 = x

    # definition
    model = Model(inputs=[input1, input2], outputs=output1, name=model_func_name)

    # # config
    # opt = Adam(lr=0.001)
    # # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=opt, metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452_FC(nb_classes=2, input_dim=256, l2_weight=0.0001):
    """

    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(input_dim))

    x = Dense(1024, name='fc_1', kernel_regularizer=regularizers.l2(l2_weight))(input1)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dense(1024, name='fc_2', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dense(1024, name='fc_22', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Dense(input_dim, name='fc_3', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Dense(nb_classes, name='fc_softmax')(x)#, kernel_constraint=max_norm(0.5)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452_FC2(nb_classes=2, input_dim=256, l2_weight=0.0001):
    """

    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(input_dim))

    x = input1

    # x = BatchNormalization()(x)

    x = Dense(2048, name='fc_1', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(2048, name='fc_3', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Dense(2048, name='fc_3', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = Dense(input_dim, name='fc_2', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(2048, name='fc_4', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    # x = Dense(input_dim, name='fc_5', kernel_regularizer=regularizers.l2(l2_weight))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = Dense(nb_classes, name='fc_softmax')(x)#, kernel_constraint=max_norm(0.5)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452_FC_Testing(nb_classes=2, input_dim=512, l2_weight=0.001):
    """

    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(input_dim))

    x = input1

    # x = Dense(1024, name='fc_1', kernel_regularizer=regularizers.l2(l2_weight))(input1)
    # x = BatchNormalization()(x)
    # x = Activation('elu')(x)

    x = Dense(1024, name='fc_2', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dropout(0.4)(x)

    x = Dense(input_dim, name='fc_22', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dropout(0.4)(x)

    x = Dense(1024, name='fc_3', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    # # x = Activation('elu')(x)

    x = Dense(nb_classes, name='fc_softmax')(x)#, kernel_constraint=max_norm(0.5)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452_FC_New(nb_classes=2, input_dim=512, l2_weight=0.0001):
    """

    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(input_dim, ))

    x = input1

    x = Dense(1024, name='fc_1', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dropout(0.4)(x)

    x = Dense(1024, name='fc_2', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Dropout(0.4)(x)

    x = Dense(input_dim, name='fc_22', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # x = Dropout(0.4)(x)

    x = Dense(1024, name='fc_3', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)

    x = Dense(nb_classes, name='fc_softmax')(x)#, kernel_constraint=max_norm(0.5)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


def EEGNet_v452_FCS(nb_classes=2, input_dim=512, l2_weight=0.001):
    """

    """
    from tensorflow.keras.layers import Concatenate, Add, Maximum, Average

    # 自动获取函数名
    model_func_name = sys._getframe().f_code.co_name

    input1 = Input(shape=(input_dim))

    x = input1

    x = Dense(1024, name='fc_2', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dropout(0.4)(x)

    x = Dense(input_dim, name='fc_22', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    x = Dense(1024, name='fc_3', kernel_regularizer=regularizers.l2(l2_weight))(x)
    x = BatchNormalization()(x)

    x = Dense(nb_classes, name='fc_softmax')(x)#, kernel_constraint=max_norm(0.5)
    softmax = Activation('softmax', name='softmax')(x)

    # definition
    model = Model(inputs=input1, outputs=softmax, name=model_func_name)

    # config
    opt = Adam(lr=0.001)
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=opt,
                  metrics=['accuracy'])

    # model.summary()

    return model


if __name__ == '__main__':
    """
    测试模型结构
    """
    from tensorflow.keras.utils import plot_model

    tt_model = EEGNet_v452_FC(nb_classes=2, input_dim=256)
    tt_model.summary()

    # plot_model(tt_model, to_file=tt_model.name + '.png',
    #            rankdir='TD',
    #            show_shapes=True, dpi=300)


