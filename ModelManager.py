import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras import backend as K
K.set_image_data_format('channels_last')
from UtilityManager import saveDatasetInPklFormat, loadDatasetFromPklFormat
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Add, Multiply, Dense, Flatten, Lambda, Conv2D, MaxPool2D, UpSampling2D, Input, LeakyReLU, Dropout, BatchNormalization, concatenate, Reshape, Softmax, ReLU, Conv2DTranspose
from keras.models import Model, model_from_json, load_model
from keras.activations import linear, selu, elu
from keras.losses import mean_squared_error, binary_crossentropy
from keras.utils.vis_utils import plot_model

import numpy as np
import ConfigManager as config
import LogManager

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def build_vgg(in_shape, InLayer):
    vgg = VGG16(weights="imagenet", input_shape=in_shape, include_top=False)
    if InLayer is not None:
        outputs = vgg.layers[InLayer].output
    else:
        outputs = vgg.output

    return Model(vgg.input, outputs)


def renameLayer(baseModel, InLayerExt='', trainable=False):
    count = 0
    for layer in baseModel.layers:
        layer.name = f'{InLayerExt}_{count}'
        layer.trainable = trainable
        count += 1


def freezeLayer(baseModel):
    for layer in baseModel.layers:
        layer.trainable = False


def save_model_to_json(InModel, InFileName):
    model_json = InModel.to_json()
    with open(f"{InFileName}.json", "w") as json_file:
        json_file.write(model_json)


def psnrInDB_loss(yTrue, yPred):
    im1 = tf.image.convert_image_dtype(yTrue, tf.float32)
    im2 = tf.image.convert_image_dtype(yPred, tf.float32)

    # mse = tf.math.reduce_mean(tf.math.squared_difference(im1, im2), [-3, -2, -1]) + 1e-8
    mse = tf.math.reduce_mean(tf.math.squared_difference(im1, im2),
                              keepdims=True) + 1e-8
    psnr2 = tf.math.subtract(20 * tf.math.log(1.0) / tf.math.log(10.0),
                             np.float32(10 / np.log(10)) * tf.math.log(mse),
                             name='psnr')

    psnr2 = (10 - psnr2) / 10
    return tf.math.sqrt(tf.math.square(psnr2))


def ssimError(yTrue, yPred):
    img1 = tf.image.convert_image_dtype(yTrue, tf.float32)
    img2 = tf.image.convert_image_dtype(yPred, tf.float32)

    # convert -1 to 1 --> 0 to 1 range
    ssimInRange = tf.image.ssim(img1, img2, max_val=1.0)
    # revert the order as 1 is better and 0 is worse in ssim error
    return (1 - ssimInRange)


def combo_loss(yTrue, yPred):
    mse = MaskMeanSquareError(yTrue, yPred)
    ssim = ssimError(yTrue, yPred)
    return (mse * 0.999) + (ssim * 0.0001)


def MaskMeanSquareError(yTrue, yPred):
    return K.sqrt(K.sqrt(K.mean(K.square(yTrue - yPred), keepdims=True)))


def predict_classes(InMode, x, batch_size=32, verbose=1):
    proba = InMode.predict(x, batch_size=batch_size, verbose=verbose)
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')


def loadWeights(InModel, InWeightPath, IsLog=True):
    if IsLog:
        LogManager.displayLog(f'Loading {InWeightPath} ..', 'cyan')
    # if self.multiGPU > 1:
    #     InModel.layers[MODEL_INDEX].load_weights(os.path.join(InWeightPath),
    #                                              by_name=True,
    #                                              skip_mismatch=True)
    # else:
    InModel.load_weights(os.path.join(InWeightPath),
                         by_name=True,
                         skip_mismatch=True)
    return InModel


def project_point_on_sphere(args):
    from tensorflow.keras.layers import concatenate, Add, Multiply

    x, y, z = args

    radius = 1
    P = K.abs(K.sqrt(K.square(x) + K.square(y) + K.square(z)))
    Q = (radius / P) * concatenate([x, y, z])

    return Q


def point_sphere(args):
    from tensorflow.keras.layers import Add, Multiply, concatenate
    import tensorflow.keras.backend as K

    longitute, latitude = args

    r = K.constant([1])

    # r = Add()([radius, altitude])
    x = Multiply()([Multiply()([r, K.cos(latitude)]), K.cos(longitute)])
    y = Multiply()([Multiply()([r, K.cos(latitude)]), K.sin(longitute)])
    z = Multiply()([r, K.sin(latitude)])

    score = concatenate([x, y, z])

    return score


def residual_block_1(x,
                     InFilterSize,
                     InActivation=LeakyReLU,
                     Iskernel_regularizer=False):
    x = BatchNormalization(momentum=0.8)(conv_layer(
        x,
        1,
        1,
        InActivation=InActivation,
        Iskernel_regularizer=Iskernel_regularizer))
    x1 = BatchNormalization(momentum=0.8)(conv_layer(
        x, 8, 3, InActivation=InActivation))
    x2 = BatchNormalization(momentum=0.8)(conv_layer(
        x,
        16,
        5,
        InActivation=InActivation,
        Iskernel_regularizer=Iskernel_regularizer))
    x3 = BatchNormalization(momentum=0.8)(conv_layer(
        x,
        32,
        7,
        InActivation=InActivation,
        Iskernel_regularizer=Iskernel_regularizer))
    f = concatenate([x1, x2, x3])
    f = BatchNormalization(momentum=0.8)(conv_layer(
        f,
        32,
        3,
        InActivation=InActivation,
        Iskernel_regularizer=Iskernel_regularizer))
    score = conv_layer(Add()([x, f]),
                       InFilterSize,
                       3,
                       InActivation=InActivation)
    return score


def save_model_pkl(InPath, InModel):
    saveDatasetInPklFormat(InPath, InModel)


def load_model_pkl(InPath):
    return loadDatasetFromPklFormat(InPath)


# def get_autoencoder_with_class():
#     model_name = 'autoencoder_with_class'
#     print(f'Model: {model_name}')
#     InNetSize = 1
#     InDropout = 0.5
#     InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
#     imageInput = Input((InImageHeight, InImageWidth, 3))

#     def conv_layer(InInput,
#                    InFilter,
#                    InKernelSize=3,
#                    InDilationRate=(1, 1),
#                    InActivation=LeakyReLU):
#         score = []
#         x = Conv2D(InFilter,
#                    kernel_size=InKernelSize,
#                    padding='same',
#                    init='he_normal',
#                    dilation_rate=InDilationRate)(InInput)
#         x = InActivation()(x)
#         score.append(x)

#         return score[-1]

#     def up(x):
#         return UpSampling2D(interpolation='bilinear')(x)

#     def down(x):
#         return MaxPool2D()(x)

#     def conv_block(InInput,
#                    InFilter=[],
#                    InDilationRate=(1, 1),
#                    IsDropout=InDropout,
#                    InNetSize=InNetSize,
#                    IsBatchNorm=True):

#         outBlock = []

#         x = conv_layer(InInput,
#                        InFilter=InFilter[0] // InNetSize,
#                        InDilationRate=InDilationRate)
#         outBlock.append(x)
#         for i in range(len(InFilter) - 1):
#             if InFilter[i + 1] == 'M':
#                 x = down(outBlock[-1])
#             elif InFilter[i + 1] == 'U':
#                 x = up(outBlock[-1])
#             else:
#                 x = conv_layer(outBlock[-1],
#                                InFilter=InFilter[i + 1] // InNetSize,
#                                InDilationRate=InDilationRate)
#             outBlock.append(x)
#         if IsDropout:
#             outBlock.append(Dropout(IsDropout)(outBlock[-1]))

#         # if IsBatchNorm:
#         #     outBlock.append((BatchNormalization()(outBlock[-1])))
#         return outBlock[-1]

#     def self_attention(InInput,
#                        InFilter,
#                        IsDropout=InDropout,
#                        InDilationRate=(1, 1),
#                        InNetSize=InNetSize):
#         outBlock = []
#         s_branch = []

#         x = conv_layer(InInput,
#                        InKernelSize=1,
#                        InFilter=InFilter // InNetSize,
#                        InDilationRate=(1, 1))
#         s_branch.append(
#             conv_layer(x,
#                        InFilter=InFilter // InNetSize,
#                        InDilationRate=InDilationRate))
#         s_branch.append(
#             conv_layer(x,
#                        InFilter=InFilter // InNetSize,
#                        InDilationRate=InDilationRate))
#         s_branch.append(
#             conv_layer(x,
#                        InFilter=InFilter // InNetSize,
#                        InDilationRate=InDilationRate))

#         x1 = Multiply()([s_branch[0], s_branch[1]])
#         x1 = Softmax()(x1)
#         x1 = Multiply()([x1, s_branch[2]])
#         x1 = Add()([x, x1])

#         outBlock.append(x1)

#         if IsDropout:
#             outBlock.append(Dropout(IsDropout)(outBlock[-1]))
#         return outBlock[-1]

#     # def attention_blocK(InInput, InFilter, InNetSize=InNetSize):
#     #     s_at_1 = self_attention(InInput, InFilter=InFilter, InDilationRate=(1, 1), InNetSize=InNetSize)
#     #     s_at_2 = self_attention(InInput, InFilter=InFilter, InDilationRate=(3, 3), InNetSize=InNetSize)
#     #     s_at_3 = self_attention(InInput, InFilter=InFilter, InDilationRate=(5, 5), InNetSize=InNetSize)
#     #     att_agg = concatenate([s_at_1, s_at_2, s_at_3], name='self_attention_block')
#     #
#     #     return att_agg

#     # Encoder
#     x = conv_block(InInput=imageInput, InFilter=[32, 'M', 64, 'M', 128])
#     sx = self_attention(x, InFilter=128)
#     # Decoder
#     x = conv_block(InInput=concatenate([sx, x]),
#                    InFilter=[128, 'U', 64, 'U', 32])
#     x = Conv2D(3, 3, padding='same')(x)
#     x = LeakyReLU(name='preview')(x)

#     x = Flatten()(x)
#     x = Dense(300, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(300, activation='relu')(x)
#     x = Dropout(0.6)(x)
#     x_out = Dense(43, activation='softmax', name='class')(x)

#     model = Model(inputs=[imageInput], outputs=[x, x_out])

#     adm = Adam(lr=config.LR,
#                decay=config.LR /
#                (int(config.TOTAL_EPOCHS / config.MIN_EPOCHS) * 0.5))
#     model.compile(loss=[MaskMeanSquareError, 'categorical_crossentropy'],
#                   loss_weights=[1, 1],
#                   optimizer=adm,
#                   metrics=['accuracy'])
#     return model


# ===
def conv_layer(InInput,
               InFilter,
               InKernelSize=3,
               InDilationRate=(1, 1),
               InActivation=LeakyReLU,
               InStride=(1, 1),
               Iskernel_regularizer=False):
    score = []
    if Iskernel_regularizer:
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   strides=InStride,
                   dilation_rate=InDilationRate,
                   kernel_regularizer='l2')(InInput)
    else:
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   strides=InStride,
                   dilation_rate=InDilationRate)(InInput)
    x = InActivation()(x)

    score.append(x)

    return score[-1]


def conv_transpose_layer(InInput,
                         InFilter,
                         InKernelSize=3,
                         InStride=(2, 2),
                         InActivation=LeakyReLU):
    score = []
    x = Conv2DTranspose(InFilter,
                        kernel_size=InKernelSize,
                        padding='same',
                        init='he_normal',
                        strides=InStride)(InInput)

    x = InActivation()(x)

    score.append(x)

    return score[-1]


def up(x, Type=(128, 'c'), InActivation=LeakyReLU):
    feature, ConvType = Type
    if ConvType == 'c':
        return conv_transpose_layer(x,
                                    feature,
                                    InStride=(2, 2),
                                    InActivation=InActivation)
    else:
        return UpSampling2D()(x)


def down(x, Type=(128, 'c'), InActivation=LeakyReLU):
    feature, ConvType = Type
    if ConvType == 'c':
        return conv_layer(x,
                          feature,
                          InStride=(2, 2),
                          InActivation=InActivation)
    else:
        return MaxPool2D()(x)


def conv_block(InInput,
               InFilter=[],
               InKernelSize=3,
               InDilationRate=(1, 1),
               IsDropout=0.5,
               InNetSize=1,
               IsBatchNorm=True,
               InActivation=LeakyReLU,
               InBatchNorm=False,
               Iskernel_regularizer=False,
               IsConvTranspose=True):

    outBlock = []

    x = conv_layer(InInput,
                   InFilter=InFilter[0],
                   InDilationRate=InDilationRate,
                   InActivation=InActivation)
    outBlock.append(x)
    for i in range(len(InFilter) - 1):
        if InFilter[i + 1] == 'M':
            if IsConvTranspose:
                x = down(outBlock[-1], InActivation=InActivation)
            else:
                x = down(outBlock[-1], (128, 'd'))
        elif InFilter[i + 1] == 'U':
            if IsConvTranspose:
                x = up(outBlock[-1], InActivation=InActivation)
            else:
                x = up(outBlock[-1], (128, 'u'))
        else:
            x = conv_layer(outBlock[-1],
                           InFilter=InFilter[i + 1] // InNetSize,
                           InDilationRate=InDilationRate,
                           InKernelSize=InKernelSize,
                           InActivation=InActivation)
            if InBatchNorm:
                x = BatchNormalization(momentum=0.8)(x)
        outBlock.append(x)
    if IsDropout:
        outBlock.append(Dropout(IsDropout)(outBlock[-1]))

    return outBlock[-1]


def dense_block(InInput,
                InFilter=[],
                IsDropout=0.5,
                InBatchNorm=True,
                InActivation=LeakyReLU):

    outBlock = []

    x = Dense(
        InFilter[0],
        init='he_normal',
    )
    x = InActivation()(x)
    outBlock.append(x)
    for i in range(len(InFilter) - 1):
        if InFilter[i + 1] == 'M':
            x = down(outBlock[-1])
        elif InFilter[i + 1] == 'U':
            x = up(outBlock[-1])
        else:
            x = Dense(
                outBlock[-1],
                InFilter=InFilter[i + 1],
                InActivation=InActivation,
                init='he_normal',
            )
            if InBatchNorm:
                x = BatchNormalization(momentum=0.8)(x)
        outBlock.append(x)
    if IsDropout:
        outBlock.append(Dropout(IsDropout)(outBlock[-1]))

    return outBlock[-1]


def conv_tree_block(x,
                    filters,
                    depth=2,
                    InKernel=3,
                    InActivation=ReLU,
                    InDropout=0.3,
                    IsDownUp='D'):
    def _branch(InX, InFilters, InDepth=1):

        if InDepth == 0:
            ll = _single_branch(InX, InFilters, InKernel=3)
            lr = _single_branch(InX, InFilters, InKernel=3)
            return ll, lr
        else:
            if InDepth % 2 == 0 and IsDownUp == 'D':
                InX = _single_branch(InX, InFilters, InStrides=(2, 2))
            elif InDepth % 2 == 0 and IsDownUp == 'U':
                InX = conv_transpose_layer(InX,
                                           InFilters,
                                           InActivation=InActivation)
            else:
                InX = _single_branch(InX, InFilters, InKernel=3)

            a, b = _branch(_single_branch(InX, InFilters), InFilters,
                           InDepth - 1)
            c, d = _branch(_single_branch(InX, InFilters), InFilters,
                           InDepth - 1)
            out_1, out_2 = Concatenate()([a, d]), Concatenate()([b, c])
            return out_1, out_2

    def _single_branch(InX, InFilters, InKernel=InKernel, InStrides=(1, 1)):
        sx = Conv2D(InFilters,
                    kernel_size=InKernel,
                    padding='same',
                    init='he_normal',
                    strides=InStrides,
                    dilation_rate=(1, 1))(InX)
        sx = InActivation()(sx)
        sx = BatchNormalization(momentum=0.8)(sx)
        return sx

    b_l, b_r = _branch(x, filters, depth)
    if IsDownUp == "D":
        x = down(x, InActivation=InActivation)
    else:
        x = up(x, InActivation=InActivation)
    score = Concatenate()([b_l, b_r, x])

    return score


def dense_tree_block(x,
                     filters,
                     depth=2,
                     InKernel=3,
                     InActivation=ReLU,
                     InDropout=0.3,
                     IsDownUp='D'):
    def _branch(InX, InFilters, InDepth=1):

        if InDepth == 0:
            ll = _single_branch(InX, InFilters, InKernel=3)
            lr = _single_branch(InX, InFilters, InKernel=5)
            ll = Dropout(InDropout)(ll)
            lr = Dropout(InDropout)(lr)
            return ll, lr
        else:
            if InDepth % 2 == 0 and IsDownUp == 'D':
                InX = down(InX, (1, 'M'))
            elif InDepth % 2 == 0 and IsDownUp == 'U':
                InX = up(InX, (1, 'U'))

            else:
                InX = _single_branch(InX, InFilters, InKernel=1)

            a, b = _branch(_single_branch(InX, InFilters), InFilters,
                           InDepth - 1)
            c, d = _branch(_single_branch(InX, InFilters), InFilters,
                           InDepth - 1)
            out_1, out_2 = Add()([a, d]), Add()([b, c])
            return out_1, out_2

    def _single_branch(InX, InFilters, InKernel=InKernel, InStrides=(1, 1)):
        sx = Dense(InFilters)(InX)
        sx = InActivation()(sx)
        sx = BatchNormalization(momentum=0.8)(sx)
        return sx

    b_l, b_r = _branch(x, filters, depth)
    score = Concatenate()([b_l, b_r])

    return score


# ===


def get_traffic_signs_recogniser(IsLog=False):
    if IsLog:
        model_name = 'traffic_sign_classifier'
        print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.5
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    def up(x):
        return UpSampling2D(interpolation='bilinear')(x)

    def down(x):
        return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
            outBlock.append(x)
        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))

        return outBlock[-1]

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    # Encoder
    x = conv_block(InInput=imageInput, InFilter=[32, 'M', 64, 'M', 128])
    sx = self_attention(x, InFilter=128)
    # Decoder
    x = conv_block(InInput=concatenate([sx, x]),
                   InFilter=[128, 'U', 64, 'U', 32])
    x = Conv2D(3, 3, padding='same')(x)
    x = LeakyReLU(name='preview')(x)
    x = Flatten()(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(InDropout)(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(InDropout)(x)
    x_out = Dense(43, activation='softmax', name='classification')(x)

    model = Model(inputs=[imageInput],
                  outputs=[x_out],
                  name='traffic_sign_classifier')

    adm = Adam(lr=config.LR,
               decay=config.LR /
               (int(config.TOTAL_EPOCHS / config.MIN_EPOCHS) * 0.5))
    model.compile(
        loss=['categorical_crossentropy'],
        # metrics=['accuracy'],
        optimizer=adm)

    return model


def get_traffic_signs_recognition_v1(InName):
    # if IsLog:
    #     model_name = 'traffic_sign_classifier'
    #     print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.5
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU,
                   InStride=(1, 1)):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   strides=InStride,
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    # def up(x):
    #     return UpSampling2D()(x)

    # def down(x):
    #     return MaxPool2D()(x)

    def up(x, Type=(128, 'c'), InActivation=LeakyReLU):
        feature, ConvType = Type
        if ConvType == 'c':
            return conv_transpose_layer(x,
                                        feature,
                                        InStride=(2, 2),
                                        InActivation=InActivation)
        else:
            return UpSampling2D()(x)

    def down(x, Type=(128, 'c'), InActivation=LeakyReLU):
        feature, ConvType = Type
        if ConvType == 'c':
            return conv_layer(x,
                              feature,
                              InStride=(2, 2),
                              InActivation=InActivation)
        else:
            return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
            outBlock.append(x)
        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))

        return outBlock[-1]

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    # Encoder
    x = conv_block(InInput=imageInput, InFilter=[32, 'M', 64, 'M', 128])
    sx = self_attention(x, InFilter=128)
    # Decoder
    x = conv_block(InInput=concatenate([sx, x]),
                   InFilter=[128, 'U', 64, 'U', 32])
    x = Conv2D(3, 3, padding='same')(x)
    x = LeakyReLU(name='preview')(x)
    x = Flatten()(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(InDropout)(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(InDropout)(x)
    x_out = Dense(43, activation='softmax', name='classification')(x)

    model = Model(inputs=[imageInput], outputs=[x_out], name=InName)

    return model


def get_autoencoder(IsCompileModel=True):
    model_name = 'autoencoder'
    # print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.4
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    def up(x):
        return UpSampling2D(interpolation='bilinear')(x)
        # return UpSampling2D()(x)

    def down(x):
        return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
            outBlock.append(x)
        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        #
        # if IsBatchNorm:
        #     outBlock.append((BatchNormalization()(outBlock[-1])))
        return outBlock[-1]

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    # def attention_blocK(InInput, InFilter, InNetSize=InNetSize):
    #     s_at_1 = self_attention(InInput, InFilter=InFilter, InDilationRate=(1, 1), InNetSize=InNetSize)
    #     s_at_2 = self_attention(InInput, InFilter=InFilter, InDilationRate=(3, 3), InNetSize=InNetSize)
    #     s_at_3 = self_attention(InInput, InFilter=InFilter, InDilationRate=(5, 5), InNetSize=InNetSize)
    #     att_agg = concatenate([s_at_1, s_at_2, s_at_3], name='self_attention_block')
    #
    #     return att_agg

    # Encoder
    x = conv_block(InInput=imageInput, InFilter=[32, 'M', 64, 'M', 128])
    sx = self_attention(x, InFilter=128, InDilationRate=(1, 1))
    # Decoder
    x = conv_block(InInput=sx, InFilter=[128, 'U', 64, 'U', 32])
    X = conv_layer(x, 3)

    model = Model(inputs=[imageInput], outputs=[x])
    if IsCompileModel:
        adm = Adam(lr=1e-6)
        model.compile(loss=[MaskMeanSquareError], optimizer=adm)

    return model


def get_autoencoder_V2(IsCompileModel=False, InLatenDim=64, InChannel=3):
    model_name = 'autoencoder'
    # print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.4
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, InChannel))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(2, 2),
                   InActivation=LeakyReLU,
                   InStride=(1, 1)):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   strides=InStride,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation(alpha=0.2)(x)
        score.append(x)

        return score[-1]

    def up(x):
        # return UpSampling2D(interpolation='bilinear')(x)
        return UpSampling2D()(x)

    def down(x):
        return MaxPool2D()(x)

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InKernelSize=1,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
                if IsBatchNorm:
                    x = BatchNormalization(momentum=0.8)(x)
            outBlock.append(x)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))

        return outBlock[-1]

    latentDim = InLatenDim
    x = conv_block(InInput=imageInput,
                   InFilter=[64, 'M', 96, 'M', 128],
                   IsBatchNorm=True)
    # sx = self_attention(x, 32)
    # f = Concatenate()([x, sx])
    # save the shape of the x i.e w,h,c
    volumeSize = K.int_shape(x)
    f = Flatten()(x)
    x = Dense(latentDim, name='latentSpace')(f)
    # decoder
    x = Dense(np.prod(volumeSize[1:]))(x)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
    x = conv_block(InInput=x,
                   InFilter=[128, 'U', 96, 'U', 64],
                   IsBatchNorm=True)
    x = conv_layer(x, 3)

    model = Model(inputs=[imageInput], outputs=[x], name='autoencoder')

    if IsCompileModel:
        adm = Adam(lr=config.LR)
        model.compile(loss=['mse'], optimizer=adm)

    return model


def get_autoencoder_gray(IsCompileModel=False, InLatenDim=64, InChannel=3):
    model_name = 'autoencoder'
    # print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.4
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, InChannel))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(2, 2),
                   InActivation=LeakyReLU,
                   InStride=(1, 1)):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   strides=InStride,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation(alpha=0.2)(x)
        score.append(x)

        return score[-1]

    def up(x):
        # return UpSampling2D(interpolation='bilinear')(x)
        return UpSampling2D()(x)

    def down(x):
        return MaxPool2D()(x)

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InKernelSize=1,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
                if IsBatchNorm:
                    x = BatchNormalization(momentum=0.8)(x)
            outBlock.append(x)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))

        return outBlock[-1]

    latentDim = InLatenDim
    x = conv_block(InInput=imageInput,
                   InFilter=[64, 'M', 96],
                   IsBatchNorm=True)
    # x = self_attention(x, 128)
    # save the shape of the x i.e w,h,c
    volumeSize = K.int_shape(x)
    f = Flatten()(x)
    x = Dense(latentDim, name='latentSpace')(f)
    # decoder
    x = Dense(np.prod(volumeSize[1:]))(x)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
    x = conv_block(InInput=x, InFilter=[96, 'U', 64], IsBatchNorm=True)
    x = conv_layer(x, 1)

    model = Model(inputs=[imageInput], outputs=[x], name='autoencoder')

    if IsCompileModel:
        adm = Adam(lr=config.LR)
        model.compile(loss=['mse'], optimizer=adm)

    return model


def get_AutoencoderGAN():
    encoder = get_autoencoder_V2(False, 64)

    classifier = Sequential([
        Conv2D(64,
               5,
               5,
               subsample=(2, 2),
               input_shape=(48, 48, 3),
               border_mode='same',
               activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Conv2D(96,
               3,
               3,
               subsample=(2, 2),
               border_mode='same',
               activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Flatten(),
        Dense(43, activation='sigmoid')
    ])

    adm = Adam(lr=config.LR)
    # adm = Adam(lr=config.LR,
    #            decay=config.LR / (int(config.TOTAL_EPOCHS) * 0.5))

    classifier.compile(loss='binary_crossentropy', optimizer=adm)
    classifier.trainable = False

    ganInput = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = encoder(ganInput)
    ganOutput = classifier(x)
    gan = Model(input=ganInput, output=[x, ganOutput])
    gan.compile(loss=['mse', 'binary_crossentropy'],
                loss_weights=[0.999, 0.001],
                optimizer=adm)

    return encoder, classifier, gan


def get_MicronNet():
    model_name = 'MicronNet'
    print(f'Model: {model_name}')

    InDropout = 0.2
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=ReLU,
                   IsBatchNorm=True):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        if IsBatchNorm:
            x = BatchNormalization()(x)
        score.append(x)

        return score[-1]

    def down(x):
        return MaxPool2D()(x)

    x = BatchNormalization()(imageInput)
    x = conv_layer(x, 1, 1)
    x = conv_layer(x, 29, 5)
    x = conv_layer(down(x), 59, 3)
    x = conv_layer(down(x), 74, 3)

    x = Flatten()(down(x))
    x = Dense(300, activation='relu')(x)
    x = Dropout(InDropout)(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(0.5)(x)
    x_out = Dense(43, activation='softmax')(x)

    model = Model(inputs=[imageInput], outputs=[x_out])

    adm = Adam(lr=config.LR,
               decay=config.LR /
               (int(config.TOTAL_EPOCHS / config.MIN_EPOCHS) * 0.5))
    model.compile(loss='categorical_crossentropy',
                  optimizer=adm,
                  metrics=['accuracy'])
    return model


def get_micronet_v1():
    l2_reg_rate = 1e-5
    eps = 1e-6
    InDropout = 0.5
    input_ = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name='data')
    # 1-part
    x = Conv2D(filters=1,
               kernel_size=(1, 1),
               padding='same',
               kernel_regularizer=l2(l2_reg_rate))(input_)
    x = BatchNormalization(epsilon=eps)(x)
    x = ReLU()(x)
    # 2-part
    x = Conv2D(filters=29,
               kernel_size=(5, 5),
               kernel_regularizer=l2(l2_reg_rate))(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Dropout(InDropout)(x)
    # 3-part
    x = Conv2D(filters=59,
               kernel_size=(3, 3),
               padding='same',
               kernel_regularizer=l2(l2_reg_rate))(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Dropout(InDropout)(x)
    # 4-part
    x = Conv2D(filters=74,
               kernel_size=(3, 3),
               padding='same',
               kernel_regularizer=l2(l2_reg_rate))(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=3, strides=2)(x)
    x = Dropout(InDropout)(x)
    # 5-part
    x = Flatten()(x)
    x = Dense(300, kernel_regularizer=l2(l2_reg_rate))(x)
    x = BatchNormalization(epsilon=eps)(x)
    x = ReLU()(x)
    x = Dropout(InDropout)(x)
    x = Dense(300)(x)
    x = ReLU()(x)
    x = Dense(config.NUM_CLASSES)(x)
    x = Softmax()(x)

    model = Model(inputs=input_, outputs=x)

    adm = Adam(lr=config.LR,
               decay=config.LR /
               (int(config.TOTAL_EPOCHS / config.MIN_EPOCHS) * 0.5))
    model.compile(loss='categorical_crossentropy',
                  optimizer=adm,
                  metrics=['accuracy'])
    return model


def get_DCGANPlus():
    encoder = Sequential([
        Dense(128 * 6 * 6, input_dim=100, activation=LeakyReLU(0.2)),
        BatchNormalization(),
        Reshape((6, 6, 128)),
        UpSampling2D(),
        Conv2D(64, 5, 5, border_mode='same', activation=LeakyReLU(0.2)),
        BatchNormalization(),
        UpSampling2D(),
        Conv2D(64, 5, 5, border_mode='same', activation=LeakyReLU(0.2)),
        UpSampling2D(),
        Conv2D(3, 5, 5, border_mode='same', activation='tanh')
    ])

    classifier = Sequential([
        Conv2D(64,
               5,
               5,
               subsample=(2, 2),
               input_shape=(48, 48, 3),
               border_mode='same',
               activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Conv2D(96,
               3,
               3,
               subsample=(2, 2),
               border_mode='same',
               activation=LeakyReLU(0.2)),
        Dropout(0.3),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    adm = Adam(lr=config.LR)
    encoder.compile(loss='binary_crossentropy', optimizer=adm)
    classifier.compile(loss='binary_crossentropy', optimizer=adm)

    classifier.trainable = False

    ganInput = Input(shape=(100, ))
    # getting the output of the encoder
    # and then feeding it to the classifier
    # new model = D(G(input))
    x = encoder(ganInput)
    ganOutput = classifier(x)
    gan = Model(input=ganInput, output=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adm)
    # gan.summary()
    return encoder, classifier, gan


def get_anomalyClassifier(IsDisplaySummary=False):

    model_name = 'anomalyClassifier'
    # print(f'Model: {model_name}')

    InDropout = 0.2
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=ReLU,
                   IsBatchNorm=True):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        if IsBatchNorm:
            x = BatchNormalization()(x)
        score.append(x)

        return score[-1]

    def down(x):
        return MaxPool2D()(x)

    x = conv_layer(imageInput, 64, 3, IsBatchNorm=True)
    x = Dropout(0.4)(x)
    x = conv_layer(x, 96, 3, IsBatchNorm=True)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x_out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[imageInput], outputs=[x_out])

    if IsDisplaySummary:
        model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=config.LR),
                  metrics=['accuracy'])
    return model


def get_anomalyClassifier_v2(IsDisplaySummary=False):

    model_name = 'anomalyClassifier_V2'
    # print(f'Model: {model_name}')

    InDropout = 0.4
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=ReLU,
                   IsBatchNorm=True):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        if IsBatchNorm:
            x = BatchNormalization()(x)
        score.append(x)

        return score[-1]

    def down(x):
        return MaxPool2D()(x)

    x = conv_layer(imageInput, 64, 3, IsBatchNorm=True)
    x = Dropout(InDropout)(x)
    x = conv_layer(x, 128, 3, IsBatchNorm=True)
    x = Dropout(InDropout)(x)
    x = Flatten()(x)
    x_out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[imageInput], outputs=[x_out])

    if IsDisplaySummary:
        model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=config.LR),
                  metrics=['accuracy'])
    return model


def get_gan():
    # model_name = 'GAN'
    # print(f'Model: {model_name}')

    InNetSize = 1
    InDropout = 0.3

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    def up(x):
        return UpSampling2D(interpolation='bilinear')(x)

    def down(x):
        return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
            outBlock.append(x)
        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        #
        # if IsBatchNorm:
        #     outBlock.append((BatchNormalization()(outBlock[-1])))
        return outBlock[-1]

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    nDimInput = Input((100, ))
    x = Dense(128 * 6 * 6, input_dim=100, activation=LeakyReLU(0.2))(nDimInput)
    x = BatchNormalization()(x)
    x = Reshape((6, 6, 128))(x)
    x = up(x)
    x = conv_block(x, [64])
    x = conv_block(InInput=x, InFilter=[64, 'U'], IsDropout=(0.1))
    x = self_attention(x, 32)
    x = up(x)
    x = conv_block(InInput=x, InFilter=[3])

    gen = Model(inputs=[nDimInput], outputs=[x])
    # gen.summary()

    inImage = Input((48, 48, 3))
    x = conv_block(InInput=inImage, InFilter=[64, 96])
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    dis = Model(inputs=[inImage], outputs=[x])
    # dis.summary()

    adm = Adam(lr=config.LR)
    gen.compile(loss='binary_crossentropy', optimizer=adm)
    dis.compile(loss='binary_crossentropy', optimizer=adm)

    dis.trainable = False

    ganInput = Input(shape=(100, ))
    # getting the output of the encoder
    # and then feeding it to the classifier
    # new model = D(G(input))
    x = gen(ganInput)
    ganOutput = dis(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adm)
    gan.summary()
    return gen, dis, gan


def load_model_from_json(InJsonFile):
    json_file = open(InJsonFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return model_from_json(loaded_model_json)


def get_sr_resnet_model(input_channel_num=3, feature_dim=64, resunit_num=16):
    def _residual_block(inputs):
        x = Conv2D(feature_dim, (3, 3),
                   padding="same",
                   kernel_initializer="he_normal")(inputs)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(feature_dim, (3, 3),
                   padding="same",
                   kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        m = Add()([x, inputs])

        return m

    inputs = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, input_channel_num))
    x = Conv2D(feature_dim, (3, 3),
               padding="same",
               kernel_initializer="he_normal")(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x0 = x

    for i in range(resunit_num):
        x = _residual_block(x)

    x = Conv2D(feature_dim, (3, 3),
               padding="same",
               kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Add()([x, x0])
    x = Conv2D(input_channel_num, (3, 3),
               padding="same",
               kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)

    return model


def get_context_generator(IsCompileModel=False):
    model_name = 'get_context_generator'
    # print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.4
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    def up(x):
        return UpSampling2D(interpolation='bilinear')(x)
        # return UpSampling2D()(x)

    def down(x):
        return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
            outBlock.append(x)

        if IsBatchNorm:
            outBlock.append(BatchNormalization()(outBlock[-1]))
        else:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    def attention_blocK(InInput, InFilter, InNetSize=InNetSize):
        s_at_1 = self_attention(InInput,
                                InFilter=InFilter,
                                InDilationRate=(1, 1),
                                InNetSize=InNetSize)
        s_at_2 = self_attention(InInput,
                                InFilter=InFilter,
                                InDilationRate=(3, 3),
                                InNetSize=InNetSize)
        s_at_3 = self_attention(InInput,
                                InFilter=InFilter,
                                InDilationRate=(5, 5),
                                InNetSize=InNetSize)
        att_agg = concatenate([s_at_1, s_at_2, s_at_3],
                              name='self_attention_block')

        return att_agg

    # Encoder
    x0 = conv_block(InInput=imageInput, InFilter=[64, 64, 64])
    sx = self_attention(x0, InFilter=64)
    x = conv_block(InInput=concatenate([sx, x0]),
                   InFilter=[64, 64],
                   IsBatchNorm=True)
    x = Add()([x, x0])
    x = conv_layer(x, 3)

    model = Model(inputs=[imageInput], outputs=[x])

    if IsCompileModel:
        adm = Adam(lr=1e-6)
        model.compile(loss=[MaskMeanSquareError], optimizer=adm)

    return model


def Get_Context_encoder_gan():
    def build_generator():
        encoder = Sequential()
        # Encoder
        encoder.add(
            Conv2D(32,
                   kernel_size=3,
                   strides=2,
                   input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
                   padding="same"))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        encoder.add(LeakyReLU(alpha=0.2))
        encoder.add(Dropout(0.5))

        # Decoder
        encoder.add(UpSampling2D())
        encoder.add(Conv2D(128, kernel_size=3, padding="same"))
        encoder.add(Activation('relu'))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(UpSampling2D())
        encoder.add(Conv2D(128, kernel_size=3, padding="same"))
        encoder.add(Activation('relu'))
        encoder.add(UpSampling2D())
        encoder.add(Conv2D(64, kernel_size=3, padding="same"))
        encoder.add(Activation('relu'))
        encoder.add(UpSampling2D())
        encoder.add(Conv2D(32, kernel_size=3, padding="same"))
        encoder.add(Activation('relu'))
        encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Conv2D(3, kernel_size=3, padding="same"))
        encoder.add(Activation('tanh'))

        #model.summary()

        masked_img = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
        gen_missing = encoder(masked_img)

        return Model(inputs=[masked_img], outputs=[gen_missing])

    def build_discriminator(InMissingImgShape=(8, 8, 3)):

        classifier = Sequential()

        classifier.add(
            Conv2D(64,
                   kernel_size=3,
                   strides=2,
                   input_shape=InMissingImgShape,
                   padding="same"))
        classifier.add(LeakyReLU(alpha=0.2))
        classifier.add(BatchNormalization(momentum=0.8))
        classifier.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        classifier.add(LeakyReLU(alpha=0.2))
        classifier.add(BatchNormalization(momentum=0.8))
        classifier.add(Conv2D(256, kernel_size=3, padding="same"))
        classifier.add(LeakyReLU(alpha=0.2))
        classifier.add(BatchNormalization(momentum=0.8))
        classifier.add(Flatten())
        classifier.add(Dense(43, activation='sigmoid'))

        #model.summary()

        img = Input(shape=InMissingImgShape)
        score = classifier(img)

        return Model(inputs=img, outputs=score)

    adam = Adam(config.LR)
    encoder = build_generator()
    #encoder.compile(loss='binary_crossentropy', optimizer=adam)
    classifier = build_discriminator(InMissingImgShape=(config.IMG_SIZE,
                                                        config.IMG_SIZE, 3))
    classifier.compile(loss='binary_crossentropy', optimizer=adam)

    classifier.trainable = False

    ganInput = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    generatedImage = encoder(ganInput)
    ganOutput = classifier(generatedImage)
    gan = Model(inputs=[ganInput], outputs=[generatedImage, ganOutput])

    gan.compile(loss=['mse', 'binary_crossentropy'],
                optimizer=adam,
                loss_weights=[0.999, 0.001])
    # gan.summary()
    return encoder, classifier, gan


def get_Context_encoder_gan_plus():
    # def build_generator():
    #     encoder = Sequential()
    #     # Encoder
    #     encoder.add(
    #         Conv2D(32,
    #                kernel_size=3,
    #                strides=2,
    #                input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    #                padding="same"))
    #     encoder.add(LeakyReLU(alpha=0.2))
    #     encoder.add(BatchNormalization(momentum=0.8))
    #     encoder.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #     encoder.add(LeakyReLU(alpha=0.2))
    #     encoder.add(BatchNormalization(momentum=0.8))
    #     encoder.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    #     encoder.add(LeakyReLU(alpha=0.2))
    #     encoder.add(BatchNormalization(momentum=0.8))
    #     encoder.add(Conv2D(512, kernel_size=1, strides=2, padding="same"))
    #     encoder.add(LeakyReLU(alpha=0.2))
    #     encoder.add(Dropout(0.5))

    #     # Decoder
    #     encoder.add(UpSampling2D())
    #     encoder.add(Conv2D(128, kernel_size=3, padding="same"))
    #     encoder.add(Activation('relu'))
    #     encoder.add(BatchNormalization(momentum=0.8))
    #     encoder.add(UpSampling2D())
    #     encoder.add(Conv2D(64, kernel_size=3, padding="same"))
    #     encoder.add(Activation('relu'))
    #     encoder.add(UpSampling2D())
    #     encoder.add(Conv2D(64, kernel_size=3, padding="same"))
    #     encoder.add(Activation('relu'))
    #     encoder.add(UpSampling2D())
    #     encoder.add(Conv2D(64, kernel_size=3, padding="same"))
    #     encoder.add(Activation('relu'))
    #     encoder.add(BatchNormalization(momentum=0.8))
    #     encoder.add(Conv2D(3, kernel_size=3, padding="same"))
    #     encoder.add(Activation('tanh'))

    #     #model.summary()

    #     masked_img = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    #     gen_missing = encoder(masked_img)

    #     return Model(inputs=[masked_img], outputs=[gen_missing])

    def build_discriminator(InMissingImgShape=(8, 8, 3), InOut=43):

        classifier = Sequential()

        classifier.add(
            Conv2D(64,
                   kernel_size=3,
                   strides=2,
                   input_shape=InMissingImgShape,
                   padding="same"))
        classifier.add(LeakyReLU(alpha=0.2))
        classifier.add(BatchNormalization(momentum=0.8))
        classifier.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        classifier.add(LeakyReLU(alpha=0.2))
        classifier.add(BatchNormalization(momentum=0.8))
        classifier.add(Conv2D(256, kernel_size=3, padding="same"))
        classifier.add(LeakyReLU(alpha=0.2))
        classifier.add(BatchNormalization(momentum=0.8))
        classifier.add(Flatten())
        classifier.add(Dense(InOut, activation='sigmoid'))

        #model.summary()

        img = Input(shape=InMissingImgShape)
        score = classifier(img)

        return Model(inputs=img, outputs=score)

    adam = Adam(config.LR)
    encoder = get_context_generator()
    #encoder = build_generator()
    #encoder.compile(loss='binary_crossentropy', optimizer=adam)
    classifier = build_discriminator(InMissingImgShape=(config.IMG_SIZE,
                                                        config.IMG_SIZE, 3))
    classifier.compile(loss='binary_crossentropy',
                       optimizer=adam,
                       metrics=['accuracy'])

    classifier.trainable = False

    ganInput = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    generatedImage = encoder(ganInput)
    ganOutput = classifier(generatedImage)
    gan = Model(inputs=[ganInput], outputs=[generatedImage, ganOutput])

    gan.compile(loss=[MaskMeanSquareError, 'binary_crossentropy'],
                optimizer=adam,
                loss_weights=[0.999, 0.001])
    # gan.summary()
    return encoder, classifier, gan


def get_reconstruction_gan(IsCompileModel=False):
    model_name = 'get_context_generator'
    # print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.3
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    def up(x):
        return UpSampling2D(interpolation='bilinear')(x)
        # return UpSampling2D()(x)

    def down(x):
        return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=False):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
                if IsBatchNorm:
                    x = BatchNormalization(momentum=0.8)(x)

            outBlock.append(x)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    def get_generator():
        latentDim = 96
        x = conv_block(InInput=imageInput,
                       InFilter=[64, 'M', 96, 'M', 128],
                       IsBatchNorm=True)

        volumeSize = K.int_shape(x)
        f = Flatten()(x)
        x = Dense(latentDim, name='latent_Space')(f)
        # decoder
        x = Dense(np.prod(volumeSize[1:]))(x)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        sx = self_attention(x, 96)
        x = conv_block(InInput=concatenate([x, sx]),
                       InFilter=[128, 'U', 96, 'U', 64],
                       IsBatchNorm=True)
        x = conv_layer(x, 3)

        return Model(inputs=[imageInput], outputs=[x], name='autoencoder')

    def get_discriminator():
        x = conv_block(imageInput, InFilter=[64], IsBatchNorm=True)
        x = Dropout(0.4)(x)
        x = conv_block(x, InFilter=[96], IsBatchNorm=True)
        x = Dropout(0.4)(x)
        x = Flatten()(x)
        x_out = Dense(1, activation='sigmoid')(x)

        return Model(inputs=[imageInput], outputs=[x_out])

    adam = Adam(lr=config.LR,
                decay=config.LR / (int(config.TOTAL_EPOCHS) * 0.5))

    encoder = get_generator()
    classifier = get_discriminator()

    encoder.compile(loss='mse', optimizer=adam)
    classifier.compile(loss='binary_crossentropy', optimizer=adam)

    classifier.trainable = False

    ganInput = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    generatedImage = encoder(ganInput)
    ganOutput = classifier(generatedImage)
    gan = Model(inputs=[ganInput], outputs=[generatedImage, ganOutput])

    gan.compile(loss=[combo_loss, 'binary_crossentropy'],
                optimizer=adam,
                loss_weights=[0.999, 0.001])

    return encoder, classifier, gan


def build_discriminator(InName):

    image_input = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = conv_block(image_input, [64, 'M', 128, 'M', 256],
                   IsDropout=0.5,
                   Iskernel_regularizer=True)
    x = Flatten()(x)

    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=image_input, outputs=x, name=InName)


def get_sphere_projection(IsLog=False, InLatenSpace=64):
    if IsLog:
        model_name = 'anomaly_detection'
        print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.2
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)

        x = InActivation()(x)

        score.append(x)

        return score[-1]

    def up(x):
        return UpSampling2D(interpolation='bilinear')(x)

    def down(x):
        return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True,
                   InActivation=LeakyReLU,
                   InBatchNorm=False):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0],
                       InDilationRate=InDilationRate,
                       InActivation=InActivation)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate,
                               InActivation=InActivation)
                if InBatchNorm:
                    x = BatchNormalization(momentum=0.8)(x)
            outBlock.append(x)
        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))

        return outBlock[-1]

    def residual_block(x, InFilterSize):
        x = BatchNormalization(momentum=0.8)(conv_layer(x, 1, 1))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(x, 8, 3))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(x, 16, 5))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(x, 32, 7))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(f, 32, 3))
        score = conv_layer(Add()([x, f]), InFilterSize, 3)
        score = Dropout(InDropout)(score)
        return score

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(
            shape=(batch,
                   dim))  # by default, random_normal has mean=0 and std=1.0
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def point_sphere(args):
        from tensorflow.keras.layers import Add, Multiply, concatenate
        import tensorflow.keras.backend as K

        longitute, latitude = args

        r = K.constant([1])

        # r = Add()([radius, altitude])
        x = Multiply()([Multiply()([r, K.cos(latitude)]), K.cos(longitute)])
        y = Multiply()([Multiply()([r, K.cos(latitude)]), K.sin(longitute)])
        z = Multiply()([r, K.sin(latitude)])

        score = concatenate([x, y, z])

        return score

    def project_point_on_sphere(args):
        from tensorflow.keras.layers import concatenate, Add, Multiply

        x, y, z = args
        radius = K.constant([1])
        # sphere_center = K.constant([0, 0, 0])
        l_x = K.sqrt(K.square(x))
        l_y = K.sqrt(K.square(y))
        l_z = K.sqrt(K.square(z))

        p = concatenate([x, y, z])
        point = K.abs(Add()([Add()([l_x, l_y]), l_z]))

        Q = Multiply()([1 / point, p])

        return Q

    latentSpace = 64
    filters = 128
    inception_range = 12
    x = conv_block(imageInput, [filters, 'M', filters, 'M'], IsBatchNorm=True)
    x = conv_block(x, [filters], IsBatchNorm=True)
    sx = x
    for _ in range(inception_range):
        sx = residual_block(sx, filters)
    x = conv_block(Concatenate()([x, sx]), [filters], IsBatchNorm=True)

    volumeSize = K.int_shape(x)
    f = Flatten()(x)

    x_ = Dense(1, activation='tanh', name='x')(f)
    y_ = Dense(1, activation='tanh', name='y')(f)
    # z_ = Dense(1, activation='tanh', name='z')(f)

    l_out = Lambda(point_sphere, output_shape=(3, ))([x_, y_])

    x_out = Dense(latentSpace, name='latent_Space')(concatenate([l_out, f]))

    encoder = Model(imageInput, [l_out, x_out], name='encoder')

    # == Decoder == #
    decoder_in = Input(shape=(latentSpace, ))
    x = Dense(np.prod(volumeSize[1:]))(decoder_in)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
    x = conv_block(x, [filters], IsBatchNorm=True)
    sx = x
    for _ in range(inception_range):
        sx = residual_block(sx, filters)
    x = conv_block(Add()([x, sx]), [filters, 'U', filters, 'U'],
                   IsBatchNorm=True)
    # x_out = conv_layer(x, 3)
    x_out = Conv2D(3, padding='same', init='he_normal', activation='tanh')(x)
    decoder = Model(decoder_in, x_out, name='decoder')

    # encoder.summary()
    # decoder.summary()
    output = decoder(encoder(imageInput)[1])
    model = Model(imageInput,
                  output,
                  name=f'anomaly_detection_vae_{config.ANOMALY_DETECTION_VAE}')
    return model, encoder, decoder


def get_sphere_projection_v1(IsLog=False, InLatenSpace=64):

    if IsLog:
        model_name = 'anomaly_detection'
        print(f'Model: {model_name}')
    # InNetSize = 1
    InDropout = 0.3
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def residual_block_0(x, InFilterSize):
        x = BatchNormalization(momentum=0.8)(conv_layer(x, 1, 1))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(x, 8))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(x, 16))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(x, 32))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(f, 32))
        score = conv_layer(Add()([x, f]), InFilterSize, 3)
        score = Dropout(InDropout)(score)
        return score

    def residual_block_1(x, InFilterSize, InActivation=LeakyReLU):
        x = BatchNormalization(momentum=0.8)(conv_layer(
            x, 1, 1, InActivation=InActivation))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 8, 3, InActivation=InActivation))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 16, 5, InActivation=InActivation))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 32, 7, InActivation=InActivation))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(
            f, 32, 3, InActivation=InActivation))
        score = conv_layer(Add()([x, f]),
                           InFilterSize,
                           3,
                           InActivation=InActivation)
        score = Dropout(InDropout)(score)
        return score

    # def residual_block_v1(x, InFilterSize):
    #     x2 = BatchNormalization(momentum=0.8)(conv_layer(x, InFilterSize))
    #     x2 = BatchNormalization(momentum=0.8)(conv_layer(x2, InFilterSize))
    #     score = conv_layer(Add()([x, x2]), InFilterSize, 3)
    #     score = Dropout(InDropout)(score)
    #     return score

    # def post_residual_block(x, x1, InFilterSize):
    #     x = BatchNormalization(momentum=0.8)(conv_layer(x, InFilterSize))
    #     score = Add()([x, x1])
    #     score = Dropout(InDropout)(score)
    #     return score

    def point_sphere(args):
        from tensorflow.keras.layers import Add, Multiply, concatenate
        import tensorflow.keras.backend as K

        longitute, latitude = args

        r = K.constant([1])

        # r = Add()([radius, altitude])
        x = Multiply()([Multiply()([r, K.cos(latitude)]), K.cos(longitute)])
        y = Multiply()([Multiply()([r, K.cos(latitude)]), K.sin(longitute)])
        z = Multiply()([r, K.sin(latitude)])

        score = concatenate([x, y, z])

        return score

    def project_point_on_sphere(args):
        from tensorflow.keras.layers import concatenate, Add, Multiply

        x, y, z = args
        radius = 1
        P = K.abs(K.sqrt(K.square(x) + K.square(y) + K.square(z)))
        Q = (radius / P) * concatenate([x, y, z])

        return Q

    latentSpace = 128
    filters = 96
    inception_range = 6
    x = conv_block(imageInput, [64, 'M', 'M', 256], IsDropout=InDropout)
    sx = x
    for _ in range(inception_range):
        sx = residual_block_0(sx, filters)

    x = conv_block(Concatenate()([x, sx]), [filters], IsDropout=InDropout)

    volumeSize = K.int_shape(x)
    f = Flatten()(x)

    x_ = Dense(1, activation='tanh', name='x')(f)
    y_ = Dense(1, activation='tanh', name='y')(f)
    z_ = Dense(1, activation='tanh', name='z')(f)

    l_out = Lambda(project_point_on_sphere, output_shape=(3, ))([x_, y_, z_])

    x_out = Dense(latentSpace, name='latent_Space')(concatenate([l_out, f]))

    encoder = Model(imageInput, [l_out, x_out], name='encoder')

    # == Decoder == #
    decoder_in = Input(shape=(latentSpace, ))
    x = Dense(np.prod(volumeSize[1:]))(decoder_in)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    x = conv_block(x, [256, 'U', 'U'],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)
    sx = x
    for _ in range(inception_range):
        sx = residual_block_1(sx, filters, InActivation=ReLU)

    x = conv_block(concatenate([sx, x]), [64],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)

    x_out = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(decoder_in, x_out, name='decoder')
    decoder.summary()

    output = decoder(encoder(imageInput)[1])
    model = Model(imageInput,
                  output,
                  name=f'anomaly_detection_vae_{config.ANOMALY_DETECTION_VAE}')

    return model


def get_anomaly_detection(IsLog=False, InLatenSpace=64):
    if IsLog:
        model_name = 'anomaly_detection'
        print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.1
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    def up(x):
        return UpSampling2D(interpolation='bilinear')(x)

    def down(x):
        return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True,
                   InActivation=LeakyReLU,
                   InBatchNorm=False):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0],
                       InDilationRate=InDilationRate,
                       InActivation=InActivation)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate,
                               InActivation=InActivation)
                if InBatchNorm:
                    x = BatchNormalization(momentum=0.8)(x)
            outBlock.append(x)
        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))

        return outBlock[-1]

    def self_attention(InInput,
                       InFilter,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    def inception_(x, InFilterSize):
        keys = conv_layer(x, InFilterSize, 1)
        querys = conv_layer(x, InFilterSize, 3)
        values = conv_layer(x, InFilterSize, 3)
        score = Multiply()([querys, keys])
        score = Softmax()(score)
        outputs = Add()([score, x])
        outputs = BatchNormalization(momentum=0.8)(outputs)
        outputs = Dropout(InDropout)(outputs)
        return outputs

    def residual_block(x, InFilterSize):
        x = BatchNormalization(momentum=0.8)(conv_layer(x, 1, 1))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(x, 8, 3))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(x, 16, 3))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(x, 32, 3))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(f, 32, 3))
        score = conv_layer(concatenate([x, f]), InFilterSize, 3)
        score = Dropout(InDropout)(score)
        return score

    latentSpace = InLatenSpace
    filters = 64
    x = conv_block(imageInput, [filters, 'M', filters], IsBatchNorm=True)
    x = conv_block(x, [filters], IsBatchNorm=True)
    for _ in range(4):
        x = residual_block(x, filters)
    x = conv_block(x, [filters], IsBatchNorm=True)

    volumeSize = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(latentSpace, name='latentSpace_1')(x)
    x = Dense(np.prod(volumeSize[1:]))(x)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    x = conv_block(x, [filters], IsBatchNorm=True)
    # for _ in range(8):
    #     sx = residual_block(x, filters)
    x = conv_block(x, [filters, 'U', filters], IsBatchNorm=True)
    x_out = conv_layer(x, 3)

    model = Model(inputs=[imageInput], outputs=[x_out])

    # adm = Adam(lr=config.LR,
    # decay=config.LR / (int(config.TOTAL_EPOCHS) * 0.5))
    model.compile(loss=['mse'], optimizer=Adam(lr=config.LR))

    return model


def get_anomaly_detection_gan():
    encoder = get_sphere_projection_v1()
    encoder.summary()
    classifier = build_discriminator()
    classifier.summary()

    # encoder.compile(loss='mse', optimizer=Adam(config.LR))
    # classifier.compile(loss='categorical_crossentropy',
    #                    optimizer=Adam(config.LR))

    classifier.trainable = False

    ganInput = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    generatedImage = encoder(ganInput)
    ganOutput = classifier(generatedImage)
    gan = Model(inputs=[ganInput],
                outputs=[generatedImage, ganOutput],
                name='anomaly_gan')

    # gan.compile(loss=['mse', 'binary_crossentropy'],
    #             optimizer=Adam(config.LR),
    #             loss_weights=[0.999, 0.0001])

    return encoder, classifier, gan


def get_anomaly_detection_vae(IsLog=False, InLatenSpace=64):
    if IsLog:
        model_name = 'anomaly_detection'
        print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.1
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    def up(x):
        return UpSampling2D(interpolation='bilinear')(x)

    def down(x):
        return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True,
                   InActivation=LeakyReLU,
                   InBatchNorm=False):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0],
                       InDilationRate=InDilationRate,
                       InActivation=InActivation)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate,
                               InActivation=InActivation)
                if InBatchNorm:
                    x = BatchNormalization(momentum=0.8)(x)
            outBlock.append(x)
        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))

        return outBlock[-1]

    def residual_block(x, InFilterSize):
        x = BatchNormalization(momentum=0.8)(conv_layer(x, 1, 1))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(x, 8, 3))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(x, 16, 3))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(x, 32, 3))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(f, 32, 3))
        score = conv_layer(concatenate([x, f]), InFilterSize, 3)
        score = Dropout(InDropout)(score)
        return score

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(
            shape=(batch,
                   dim))  # by default, random_normal has mean=0 and std=1.0
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def point_sphere(args):
        longitute, latitude, altitude = args

        radius = K.constant([1])
        r = Add()([radius, altitude])
        x = Multiply()(
            [Multiply()([r, tf.math.cos(longitute)]),
             tf.math.sin(latitude)])
        y = Multiply()(
            [Multiply()([r, tf.math.sin(longitute)]),
             tf.math.sin(latitude)])
        z = Multiply()([r, tf.math.cos(latitude)])

        score = concatenate([x, y, z])

        return score

    def dot_matrix(args):
        x = args
        worldMatrix = K.constant([[1.0, 0.0, 0.0], [0.0, 0.1, 0.0],
                                  [0.0, 0.0, 1.0]])

        dm = tf.matmul(worldMatrix, tf.transpose(x))
        dm = tf.transpose(dm)

        return dm

    latentSpace = InLatenSpace
    filters = 64
    x = conv_block(imageInput, [filters, 'M', filters], IsBatchNorm=True)
    x = conv_block(x, [filters], IsBatchNorm=True)
    for _ in range(4):
        x = residual_block(x, filters)
    x = conv_block(x, [filters], IsBatchNorm=True)

    volumeSize = K.int_shape(x)
    latent_in_space = 3
    x = Flatten()(x)

    # d_1 = Dense(1, name='x', activation='tanh')(x)
    # d_2 = Dense(1, name='y', activation='tanh')(x)
    # d_3 = Dense(1, name='z', activation='tanh')(x)

    # z = Lambda(point_sphere, output_shape=(latent_in_space, ),
    #            name='z_')([d_1, d_2, d_3])

    z_mean = Dense(latent_in_space, name='z_mean', activation='relu')(x)
    z_log_var = Dense(latent_in_space, name='z_log_var', activation='relu')(x)
    z = Lambda(sampling, output_shape=(64, ), name='z')([z_mean, z_log_var])

    encoder = Model(imageInput, [z_mean, z_log_var, z], name='encoder')
    # encoder.summary()
    # instantiate decoder model
    latent_inputs = Input(shape=(latent_in_space, ), name='z_sampling')
    x = Dense(np.prod(volumeSize[1:]), name='d_1')(latent_inputs)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]), name='d_2')(x)
    x = conv_block(x, [filters], IsBatchNorm=True, InActivation=ReLU)
    # for _ in range(4):
    #     x = residual_block(x, filters)
    x = conv_block(x, [filters, 'U', filters],
                   IsBatchNorm=True,
                   InActivation=ReLU)
    x_out = conv_layer(x, 3, InActivation=ReLU)
    decoder = Model(latent_inputs, x_out, name='decoder')
    # decoder.summary()

    outputs = decoder(encoder(imageInput)[2])
    model = Model(imageInput,
                  outputs,
                  name=f'anomaly_detection_vae_{config.ANOMALY_DETECTION_VAE}')

    return model, encoder, decoder


def get_traffic_to_meta_traffic(InName, InLatenSpace=64):
    # InNetSize = 1
    InDropout = 0.5
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def residual_block_0(x, InFilterSize):
        x = BatchNormalization(momentum=0.8)(conv_layer(x, 1, 1))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(x, 8))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(x, 16))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(x, 32))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(f, 32))
        score = conv_layer(Add()([x, f]), InFilterSize, 3)
        score = Dropout(InDropout)(score)
        return score

    def residual_block_1(x, InFilterSize, InActivation=LeakyReLU):
        x = BatchNormalization(momentum=0.8)(conv_layer(
            x, 1, 1, InActivation=InActivation))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 8, 3, InActivation=InActivation))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 16, 5, InActivation=InActivation))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 32, 7, InActivation=InActivation))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(
            f, 32, 3, InActivation=InActivation))
        score = conv_layer(Add()([x, f]),
                           InFilterSize,
                           3,
                           InActivation=InActivation)
        score = Dropout(InDropout)(score)
        return score

    def point_sphere(args):
        from tensorflow.keras.layers import Add, Multiply, concatenate
        import tensorflow.keras.backend as K

        longitute, latitude = args

        r = K.constant([1])

        # r = Add()([radius, altitude])
        x = Multiply()([Multiply()([r, K.cos(latitude)]), K.cos(longitute)])
        y = Multiply()([Multiply()([r, K.cos(latitude)]), K.sin(longitute)])
        z = Multiply()([r, K.sin(latitude)])

        score = concatenate([x, y, z])

        return score

    def project_point_on_sphere(args):
        from tensorflow.keras.layers import concatenate, Add, Multiply

        x, y, z = args
        radius = 1
        P = K.abs(K.sqrt(K.square(x) + K.square(y) + K.square(z)))
        Q = (radius / P) * concatenate([x, y, z])

        return Q

    latentSpace = 76
    filters = 96
    inception_range = 4
    x = conv_block(imageInput, [64, 'M', 256, 'M', 512, 'M'],
                   IsDropout=InDropout)
    # sx = x
    # for _ in range(inception_range):
    #     sx = residual_block_0(sx, filters)

    # x = conv_block(concatenate([x, sx]), [256], IsDropout=InDropout)

    f = Flatten()(x)
    # x_ = Dense(1, activation='tanh', name='x')(f)
    # y_ = Dense(1, activation='tanh', name='y')(f)
    # z_ = Dense(1, activation='tanh', name='z')(f)

    # l_out = Lambda(project_point_on_sphere, output_shape=(3, ))([x_, y_, z_])

    x_out = Dense(latentSpace, name='latent_Space')(f)

    encoder = Model(imageInput, x_out, name='encoder')
    encoder.summary()
    # # == Decoder == #
    decoder_in = Input(shape=(latentSpace, ))
    x = Dense(24 * 24 * 48, activation='relu')(decoder_in)
    x = Reshape((24, 24, 48))(x)

    x = conv_block(x, [256, 'U'],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)
    sx = x
    for _ in range(inception_range):
        sx = residual_block_1(sx, filters, InActivation=ReLU)

    x = conv_block(concatenate([sx, x]), [128],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)

    x_out = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(decoder_in, x_out, name='decoder')
    decoder.summary()

    output = decoder(encoder(imageInput))
    model = Model(imageInput, output, name=f'{InName}')

    return model


def get_traffic_to_meta_traffic_v2(IsLog=False, InLatenSpace=64):

    if IsLog:
        model_name = 'anomaly_detection'
        print(f'Model: {model_name}')
    # InNetSize = 1
    InDropout = 0.5
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def residual_block_0(x, InFilterSize):
        x = BatchNormalization(momentum=0.8)(conv_layer(x, 1, 1))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(x, 8))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(x, 16))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(x, 32))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(f, 32))
        score = conv_layer(Add()([x, f]), InFilterSize, 3)
        score = Dropout(InDropout)(score)
        return score

    def residual_block_1(x, InFilterSize, InActivation=LeakyReLU):
        x = BatchNormalization(momentum=0.8)(conv_layer(
            x, 1, 1, InActivation=InActivation))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 8, 3, InActivation=InActivation))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 16, 5, InActivation=InActivation))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 32, 7, InActivation=InActivation))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(
            f, 32, 3, InActivation=InActivation))
        score = conv_layer(Add()([x, f]),
                           InFilterSize,
                           3,
                           InActivation=InActivation)
        score = Dropout(InDropout)(score)
        return score

    def inception_block(x, filters, depth=2, InActivation=ReLU):
        def _branch(InX, InDepth=1):
            x1 = _single_branch(InX)
            x2 = _single_branch(InX)
            if InDepth > 0:
                depth -= 1
                return _branch(concatenate([x1, x2]), depth)

        def _single_branch(InX):
            sx = residual_block_1(InX, filters, InActivation=InActivation)
            return sx

        x = conv_layer(x, filters, InActivation=InActivation)
        x = _branch(x, 6)
        x = Dropout(InDropout)
        return x

    def point_sphere(args):
        from tensorflow.keras.layers import Add, Multiply, concatenate
        import tensorflow.keras.backend as K

        longitute, latitude = args

        r = K.constant([1])

        # r = Add()([radius, altitude])
        x = Multiply()([Multiply()([r, K.cos(latitude)]), K.cos(longitute)])
        y = Multiply()([Multiply()([r, K.cos(latitude)]), K.sin(longitute)])
        z = Multiply()([r, K.sin(latitude)])

        score = concatenate([x, y, z])

        return score

    def project_point_on_sphere(args):
        from tensorflow.keras.layers import concatenate, Add, Multiply

        x, y, z = args
        radius = 1
        P = K.abs(K.sqrt(K.square(x) + K.square(y) + K.square(z)))
        Q = (radius / P) * concatenate([x, y, z])

        return Q

    latentSpace = 43
    filters = 96
    inception_range = 12
    # x = conv_block(imageInput, [64, 'M', 'M', 256], IsDropout=InDropout)
    # sx = x
    # for _ in range(inception_range // 2):
    #     sx = residual_block_0(sx, filters)

    # x = conv_block(Concatenate()([x, sx]), [filters], IsDropout=InDropout)

    # volumeSize = K.int_shape(x)
    # f = Flatten()(x)
    # x = LeakyReLU(0.2)(Dense(256)(f))
    # x = Dropout(InDropout)(x)
    # x = LeakyReLU(0.2)(Dense(256)(x))
    # x = Dropout(InDropout)(x)
    # x_out = Dense(latentSpace, activation='softmax', name='classification')(x)
    # encoder = Model(imageInput, [x_out], name='encoder')
    encoder = get_traffic_signs_recogniser()
    # encoder.summary()
    # # == Decoder == #
    decoder_in = Input(shape=(latentSpace, ))
    shapes = 48
    features = 32
    x = Dense(features * shapes * shapes, activation='relu')(decoder_in)
    x = Reshape((shapes, shapes, features))(x)

    x = conv_block(x, [256, 128],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)
    sx = x
    for _ in range(inception_range):
        sx = residual_block_1(sx, 64, InActivation=ReLU)

    x = conv_block(concatenate([sx, x]), [128],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)

    x_out = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(decoder_in, x_out, name='decoder')
    # decoder.summary()

    output = decoder(encoder(imageInput))
    model = Model(imageInput,
                  output,
                  name=f'traffic_to_meta_traffic_{config.TO_META}')
    # model.summary()
    return model, encoder, decoder


def get_traffic_to_meta_traffic_v3(InName='traffic_to_meta_traffic',
                                   IsLog=False,
                                   InLatenSpace=64):

    if IsLog:
        model_name = InName
        print(f'Model: {model_name}')
    # InNetSize = 1
    InDropout = 0.5
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def residual_block_0(x, InFilterSize):
        x = BatchNormalization(momentum=0.8)(conv_layer(x, 1, 1))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(x, 8))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(x, 16))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(x, 32))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(f, 32))
        score = conv_layer(Add()([x, f]), InFilterSize, 3)
        score = Dropout(InDropout)(score)
        return score

    def residual_block_1(x, InFilterSize, InActivation=LeakyReLU):
        x = BatchNormalization(momentum=0.8)(conv_layer(
            x, 1, 1, InActivation=InActivation))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 8, 3, InActivation=InActivation))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 16, 5, InActivation=InActivation))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 32, 7, InActivation=InActivation))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(
            f, 32, 3, InActivation=InActivation))
        score = conv_layer(Add()([x, f]),
                           InFilterSize,
                           3,
                           InActivation=InActivation)
        score = Dropout(InDropout)(score)
        return score

    def point_sphere(args):
        from tensorflow.keras.layers import Add, Multiply, concatenate
        import tensorflow.keras.backend as K

        longitute, latitude = args

        r = K.constant([1])

        # r = Add()([radius, altitude])
        x = Multiply()([Multiply()([r, K.cos(latitude)]), K.cos(longitute)])
        y = Multiply()([Multiply()([r, K.cos(latitude)]), K.sin(longitute)])
        z = Multiply()([r, K.sin(latitude)])

        score = concatenate([x, y, z])

        return score

    def project_point_on_sphere(args):
        from tensorflow.keras.layers import concatenate, Add, Multiply

        x, y, z = args
        radius = 1
        P = K.abs(K.sqrt(K.square(x) + K.square(y) + K.square(z)))
        Q = (radius / P) * concatenate([x, y, z])

        return Q

    latentSpace = 128
    filters = 96
    inception_range = 6
    x = conv_block(imageInput, [64, 'M', 'M', 256], IsDropout=InDropout)
    sx = x
    for _ in range(inception_range):
        sx = residual_block_0(sx, filters)

    x = conv_block(Concatenate()([x, sx]), [filters], IsDropout=InDropout)

    volumeSize = K.int_shape(x)
    # f = Flatten()(x)

    # x_ = Dense(1, activation='tanh', name='x')(f)
    # y_ = Dense(1, activation='tanh', name='y')(f)
    # z_ = Dense(1, activation='tanh', name='z')(f)

    # l_out = Lambda(project_point_on_sphere, output_shape=(3, ))([x_, y_, z_])

    x_out = Dense(latentSpace, name='latent_Space')(f)

    encoder = Model(imageInput, x_out, name='encoder')
    # encoder.summary()
    # == Decoder == #
    decoder_in = Input(shape=(latentSpace, ))
    shapes = 24
    features = 32
    x = Dense(features * shapes * shapes, activation='relu')(decoder_in)
    x = Reshape((shapes, shapes, features))(x)

    x = conv_block(x, [256, 'U'],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)
    sx = x
    for _ in range(inception_range):
        sx = residual_block_1(sx, 96, InActivation=ReLU)

    x = conv_block(concatenate([sx, x]), [128],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)

    x_out = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(decoder_in, x_out, name='decoder')
    # decoder.summary()

    output = decoder(encoder(imageInput))
    model = Model(imageInput,
                  output,
                  name=f'traffic_to_meta_traffic_{config.TO_META}')

    return model


def get_traffic_to_meta_traffic_v4(InName):
    InDropout = 0.5
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def residual_block_0(x, InFilterSize):
        x = BatchNormalization(momentum=0.8)(conv_layer(x, 1, 1))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(x, 8))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(x, 16))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(x, 32))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(f, 32))
        score = conv_layer(Add()([x, f]), InFilterSize, 3)
        score = Dropout(InDropout)(score)
        return score

    def residual_block_1(x, InFilterSize, InActivation=LeakyReLU):
        x = BatchNormalization(momentum=0.8)(conv_layer(
            x, 1, 1, InActivation=InActivation))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 8, 3, InActivation=InActivation))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 16, 5, InActivation=InActivation))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 32, 7, InActivation=InActivation))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(
            f, 32, 3, InActivation=InActivation))
        score = conv_layer(Add()([x, f]),
                           InFilterSize,
                           3,
                           InActivation=InActivation)
        score = Dropout(InDropout)(score)
        return score

    latentSpace = 96
    filters = 96
    inception_range = 4
    x = conv_block(imageInput, [64, 'M', 'M', 256], IsDropout=InDropout)
    # sx = x
    # for _ in range(inception_range):
    #     sx = residual_block_0(sx, filters)

    x = conv_block(x, [filters], IsDropout=InDropout)

    f = Flatten()(x)

    x_out = Dense(latentSpace, name='latent_Space')(f)

    encoder = Model(imageInput, [x_out], name='encoder')
    # encoder.summary()
    # == Decoder == #
    decoder_in = Input(shape=(latentSpace, ))
    shapes = 12
    features = 32
    x = Dense(features * shapes * shapes, activation='relu')(decoder_in)
    x = Reshape((shapes, shapes, features))(x)

    x = conv_block(x, [256, 'U', 256, 'U'],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)
    sx = x
    for _ in range(inception_range):
        sx = residual_block_1(sx, 64, InActivation=ReLU)

    x = conv_block(concatenate([sx, x]), [128],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)

    x_out = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(decoder_in, x_out, name='decoder')
    # decoder.summary()

    output = decoder(encoder(imageInput))
    model = Model(imageInput, output, name=InName)

    return model


def get_traffic_to_meta_traffic_v6(InName):
    latentspace = 43
    encoder = get_traffic_signs_recognition_v1('encoder')
    encoder.summary()

    inDecoder = Input(shape=(latentspace, ))
    x = Dense(12 * 12 * 192, activation='relu')(inDecoder)
    x = Reshape((12, 12, 192))(x)

    x = conv_block(x, [512, 'U', 256],
                   InActivation=ReLU,
                   Iskernel_regularizer=True)
    sx = x
    for _ in range(4):
        sx = residual_block_1(sx, 64, ReLU, Iskernel_regularizer=True)
        sx = Dropout(0.2)(sx)
    x = conv_block(concatenate([x, sx]), [256, 'U', 64],
                   InActivation=ReLU,
                   Iskernel_regularizer=True)

    x = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(inDecoder, x, name='decoder')
    decoder.summary()
    inImage = Input((config.IMG_SIZE, config.IMG_SIZE, 3))
    out = decoder(encoder(inImage))

    return Model(inImage, out, name=InName)


def get_traffic_to_meta_traffic_v5(InName):
    latentspace = 128
    inImage = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = conv_block(inImage, [64, 'M', 256, 'M', 512, 'M'],
                   Iskernel_regularizer=True)

    x = Flatten()(x)
    x_out = Dense(latentspace)(x)

    encoder = Model(inImage, [x_out], name='encoder')
    encoder.summary()

    inDecoder = Input(shape=(latentspace, ))
    shape = 12
    x = Dense(shape * shape * 256, activation='relu')(inDecoder)
    x = Reshape((shape, shape, 256))(x)

    x = conv_block(x, [512, 'U', 256],
                   InActivation=ReLU,
                   Iskernel_regularizer=True)
    sx = x
    for _ in range(4):
        sx = residual_block_1(sx, 64, ReLU, Iskernel_regularizer=True)
        sx = Dropout(0.2)(sx)
    x = conv_block(concatenate([x, sx]), [256, 'U', 64],
                   InActivation=ReLU,
                   Iskernel_regularizer=True)

    x = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(inDecoder, x, name='decoder')
    decoder.summary()
    out = decoder(encoder(inImage))

    return Model(inImage, out, name=InName)


def get_anomaly_detection_vae_with_Tree():
    InDropout = 0.3

    latentSpace = 128
    imageInput = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = conv_block(imageInput, [64, 'M', 'M', 256], IsDropout=InDropout)
    sx = conv_tree_block(x, 96, 2, InActivation=LeakyReLU, InDropout=InDropout)

    x = conv_layer(concatenate([x, sx]), 96)
    volumeSize = K.int_shape(x)
    f = Flatten()(x)

    x_ = Dense(1, activation='tanh', name='x')(f)
    y_ = Dense(1, activation='tanh', name='y')(f)
    z_ = Dense(1, activation='tanh', name='z')(f)

    l_out = Lambda(project_point_on_sphere, output_shape=(3, ))([x_, y_, z_])
    x_out = Dense(latentSpace, name='latent_Space')(concatenate([l_out, f]))

    encoder = Model(imageInput, [l_out, x_out], name='encoder')
    # encoder.summary()
    # plot_model(encoder,
    #            to_file='model_plot.png',
    #            show_shapes=True,
    #            show_layer_names=True)
    # == Decoder == #
    decoder_in = Input(shape=(latentSpace, ))
    x = Dense(np.prod(volumeSize[1:]))(decoder_in)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
    x = conv_block(x, [256, 'U', 'U'],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)

    sx = conv_tree_block(x, 64, 1, InKernel=5, InDropout=InDropout)
    x = conv_block(concatenate([sx, x]), [128],
                   InKernelSize=5,
                   IsDropout=InDropout,
                   InActivation=ReLU)

    x_out = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(decoder_in, x_out, name='decoder')
    decoder.summary()

    output = decoder(encoder(imageInput)[1])
    model = Model(imageInput,
                  output,
                  name=f'traffic_to_meta_traffic_{config.TO_META}')

    model.summary()
    return model


def simple_autoencoder(InName):
    image_input = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    latenspace = 96

    x = Dense(64)(image_input)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = down(x, (1, 'M'))
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = down(x, (1, 'M'))
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = down(x, (1, 'M'))
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = down(x, (1, 'M'))
    x = Flatten()(x)
    x = Dense(latenspace)(x)

    encoder = Model(image_input, x, name='encoder')
    # # encoder.summary()

    decoder_in = Input((latenspace, ))
    x = Dense(128 * 24 * 24, activation='relu')(decoder_in)
    x = BatchNormalization()(x)
    x = Reshape((24, 24, 128))(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = up(x, (1, 'U'))
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(3, activation='tanh')(x)
    decoder = Model(decoder_in, x, name='decoder')
    decoder.summary()
    out = decoder(encoder(image_input))
    model = Model(image_input, out, name=InName)
    return model


def get_traffic_to_fourier():
    generator = simple_autoencoder()

    inImage = Input((48, 48, 3))
    x = Dense(64, activation='relu')(inImage)
    x = BatchNormalization()(x)
    x = down(x, (1, 'M'))
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='sigmoid')(x)
    dis = Model(inImage, x)

    score = (generator(inImage))
    d_out = dis(score)
    dis.trainable = False
    model = Model(inImage, [score, d_out], name='traffic_to_fourier_gan')
    return generator, dis, model


def get_anomaly_with_score():
    encoder = get_traffic_to_meta_traffic_v3()
    ano_classifier = build_discriminator()
    ano_classifier.trainable = False

    image = Input((48, 48, 3))
    reconstructImage = encoder(image)
    score = ano_classifier(reconstructImage)

    gan = Model(image, [reconstructImage, score])
    return encoder, ano_classifier, gan


def get_anomaly_with_classifier():
    in_image = Input((48, 48, 3))
    latenspace = 128
    filters = 16

    x = Dense(filters * 2)(in_image)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = down(x, (1, 'M'))
    x = Dense(filters * 3)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    f = Dense(latenspace, activation='relu')(x)
    encoder = Model(in_image, f, name='encoder')
    encoder.summary()

    classifer_input = Input((latenspace, ))
    x = Dense(300, activation='relu')(classifer_input)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)
    x = BatchNormalization()(x)
    x_out = Dense(1, activation='sigmoid')(x)
    classifier = Model(classifer_input, x_out, name='classification')
    classifier.summary()

    decoder_input = Input((latenspace, ))
    x = Dense(48 * 24 * 24, activation='relu')(decoder_input)
    x = BatchNormalization()(x)
    x = Reshape((24, 24, 48))(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = up(x, (1, 'U'))
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    decoder_out = Dense(3, activation='tanh')(x)
    decoder = Model(decoder_input, decoder_out, name='decoder')
    decoder.summary()

    out_image = decoder(encoder(in_image))
    out_score = classifier(encoder(in_image))
    model = Model(
        in_image, [out_image, out_score],
        name=f'anomaly_detection_score_{config.ANOMALY_DETECTION_WITH_SCORE}')
    return model


def test_tree(InName):
    latentSpace = 48
    in_image = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    # x = conv_block(in_image, [64], InActivation=LeakyReLU)
    sx = conv_tree_block(in_image, 32, 3, IsDownUp='D', InActivation=LeakyReLU)
    x = conv_block(sx, [96], InActivation=LeakyReLU)
    f = Flatten()(x)
    x = Dense(latentSpace)(f)
    encoder = Model(in_image, x, name='encoder')
    encoder.summary()

    inDecoder = Input(shape=(latentSpace, ))
    x = Dense(12 * 12 * 48, activation='relu')(inDecoder)
    x = Reshape((12, 12, 48))(x)
    x = conv_tree_block(x, 64, 3, IsDownUp='U')
    x = conv_block(x, [128, 'U'], InActivation=ReLU)
    x = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(inDecoder, x, name='decoder')
    decoder.summary()
    out = decoder(encoder(in_image))

    model = Model(in_image, out, name=InName)
    # model.summary()
    # plot_model(model, 'model_preview.png', True)
    return model


# def get_anomaly_autoencoder(InName):
#     latentspace = 128
#     inImage = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
#     x = conv_block(inImage, [64, 'M', 256, 'M', 512, 'M'], IsDropout=0.1)
#     x = Flatten()(x)
#     x_, y_, z_ = Dense(1, activation='tanh')(x), Dense(
#         1, activation='tanh')(x), Dense(1, activation='tanh')(x)

#     sphere = Lambda(project_point_on_sphere)([x_, y_, z_])
#     x_out = Dense(latentspace)(concatenate([sphere, x]))
#     encoder = Model(inImage, [sphere, x_out], name='encoder')
#     encoder.summary()

#     inDecoder = Input(shape=(latentspace, ))
#     x = Dense(12 * 12 * 128, activation='relu')(inDecoder)
#     x = Reshape((12, 12, 128))(x)
#     x = conv_block(x, [512, 'U', 256, 'U', 64],
#                    InActivation=ReLU,
#                    IsDropout=0.1)
#     x = Conv2D(3, 1, activation='tanh')(x)
#     decoder = Model(inDecoder, x, name='decoder')
#     decoder.summary()
#     out = decoder(encoder(inImage)[1])


#     return Model(inImage, out, name=InName)
def get_gan(InGenerator, InDiscriminator, InOptimiser):
    InGenerator.compile(loss=MaskMeanSquareError, optimizer=InOptimiser)
    InDiscriminator.compile(loss='binary_crossentropy', optimizer=InOptimiser)

    InDiscriminator.trainable = False

    ganInput = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    generatedImage = InGenerator(ganInput)
    ganOutput = InDiscriminator(generatedImage)
    gan = Model(inputs=[ganInput], outputs=[generatedImage, ganOutput])

    gan.compile(loss=[MaskMeanSquareError, 'binary_crossentropy'],
                optimizer=InOptimiser,
                loss_weights=[0.999, 0.001])

    return InGenerator, InDiscriminator, gan

    pass


def get_anomaly_autoencoder(InName):
    def residual_block_1(x,
                         InFilterSize,
                         InActivation=LeakyReLU,
                         Iskernel_regularizer=False):
        x = BatchNormalization(momentum=0.8)(conv_layer(
            x,
            1,
            1,
            InActivation=InActivation,
            Iskernel_regularizer=Iskernel_regularizer))
        x1 = BatchNormalization(momentum=0.8)(conv_layer(
            x, 8, 3, InActivation=InActivation))
        x2 = BatchNormalization(momentum=0.8)(conv_layer(
            x,
            16,
            5,
            InActivation=InActivation,
            Iskernel_regularizer=Iskernel_regularizer))
        x3 = BatchNormalization(momentum=0.8)(conv_layer(
            x,
            32,
            7,
            InActivation=InActivation,
            Iskernel_regularizer=Iskernel_regularizer))
        f = concatenate([x1, x2, x3])
        f = BatchNormalization(momentum=0.8)(conv_layer(
            f,
            32,
            3,
            InActivation=InActivation,
            Iskernel_regularizer=Iskernel_regularizer))
        score = conv_layer(Add()([x, f]),
                           InFilterSize,
                           3,
                           InActivation=InActivation)
        return score

    latentspace = 48
    inImage = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = conv_block(inImage, [64, 'M', 256, 'M', 512, 'M'],
                   Iskernel_regularizer=True)

    x = Flatten()(x)

    x_out = Dense(latentspace)(x)

    encoder = Model(inImage, [x_out], name='encoder')
    encoder.summary()

    inDecoder = Input(shape=(latentspace, ))
    x = Dense(12 * 12 * 386, activation='relu')(inDecoder)
    x = Reshape((12, 12, 386))(x)

    x = conv_block(x, [512, 'U', 256],
                   InActivation=ReLU,
                   Iskernel_regularizer=True)
    sx = x
    for _ in range(4):
        sx = residual_block_1(sx, 64, ReLU, Iskernel_regularizer=True)
        sx = Dropout(0.2)(sx)
    x = conv_block(concatenate([x, sx]), [256, 'U', 64],
                   InActivation=ReLU,
                   Iskernel_regularizer=True)

    x = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(inDecoder, x, name='decoder')
    decoder.summary()
    out = decoder(encoder(inImage))

    return Model(inImage, out, name=InName)


def get_anomaly_autoencoder_relu(InName):
    latentspace = 48
    inImage = Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    x = conv_block(inImage, [64, 'M', 256, 'M', 512, 'M'],
                   Iskernel_regularizer=True,
                   InActivation=ReLU)

    x = Flatten()(x)

    x_out = Dense(latentspace)(x)

    encoder = Model(inImage, [x_out], name='encoder')
    encoder.summary()

    inDecoder = Input(shape=(latentspace, ))
    x = Dense(12 * 12 * 386, activation='relu')(inDecoder)
    x = Reshape((12, 12, 386))(x)

    x = conv_block(x, [512, 'U', 256],
                   InActivation=ReLU,
                   Iskernel_regularizer=True,
                   IsConvTranspose=False)
    sx = x
    for _ in range(4):
        sx = residual_block_1(sx, 64, ReLU, Iskernel_regularizer=True)
        sx = Dropout(0.2)(sx)
    x = conv_block(concatenate([x, sx]), [256, 'U', 64],
                   InActivation=ReLU,
                   Iskernel_regularizer=True,
                   IsConvTranspose=False)

    x = Conv2D(3, 1, activation='tanh')(x)
    decoder = Model(inDecoder, x, name='decoder')
    decoder.summary()
    out = decoder(encoder(inImage))

    return Model(inImage, out, name=InName)


def run_mode(InFunction, InName):
    m = InFunction(InName)
    m.summary()
    # m.save(f'Models/{InName}.h5')
    model_json = m.to_json()
    with open(f"Models/{InName}.json", "w") as json_file:
        json_file.write(model_json)


def get_demo_traffic_signs_recognition_v1(InName):
    # if IsLog:
    #     model_name = 'traffic_sign_classifier'
    #     print(f'Model: {model_name}')
    InNetSize = 1
    InDropout = 0.5
    InImageHeight, InImageWidth = config.IMG_SIZE, config.IMG_SIZE
    imageInput = Input((InImageHeight, InImageWidth, 3))

    def conv_layer(InInput,
                   InFilter,
                   InKernelSize=3,
                   InDilationRate=(1, 1),
                   InActivation=LeakyReLU,
                   InStride=(1, 1)):
        score = []
        x = Conv2D(InFilter,
                   kernel_size=InKernelSize,
                   padding='same',
                   init='he_normal',
                   strides=InStride,
                   dilation_rate=InDilationRate)(InInput)
        x = InActivation()(x)
        score.append(x)

        return score[-1]

    # def up(x):
    #     return UpSampling2D()(x)

    # def down(x):
    #     return MaxPool2D()(x)

    def up(x, Type=(128, 'c'), InActivation=LeakyReLU):
        feature, ConvType = Type
        if ConvType == 'c':
            return conv_transpose_layer(x,
                                        feature,
                                        InStride=(2, 2),
                                        InActivation=InActivation)
        else:
            return UpSampling2D()(x)

    def down(x, Type=(128, 'c'), InActivation=LeakyReLU):
        feature, ConvType = Type
        if ConvType == 'c':
            return conv_layer(x,
                              feature,
                              InStride=(2, 2),
                              InActivation=InActivation)
        else:
            return MaxPool2D()(x)

    def conv_block(InInput,
                   InFilter=[],
                   InDilationRate=(1, 1),
                   IsDropout=InDropout,
                   InNetSize=InNetSize,
                   IsBatchNorm=True):

        outBlock = []

        x = conv_layer(InInput,
                       InFilter=InFilter[0] // InNetSize,
                       InDilationRate=InDilationRate)
        outBlock.append(x)
        for i in range(len(InFilter) - 1):
            if InFilter[i + 1] == 'M':
                x = down(outBlock[-1])
            elif InFilter[i + 1] == 'U':
                x = up(outBlock[-1])
            else:
                x = conv_layer(outBlock[-1],
                               InFilter=InFilter[i + 1] // InNetSize,
                               InDilationRate=InDilationRate)
            outBlock.append(x)
        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))

        return outBlock[-1]

    def self_attention(InInput,
                       InFilte,
                       IsDropout=InDropout,
                       InDilationRate=(1, 1),
                       InNetSize=InNetSize):
        outBlock = []
        s_branch = []

        x = conv_layer(InInput,
                       InKernelSize=1,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=(1, 1))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))
        s_branch.append(
            conv_layer(x,
                       InFilter=InFilter // InNetSize,
                       InDilationRate=InDilationRate))

        x1 = Multiply()([s_branch[0], s_branch[1]])
        x1 = Softmax()(x1)
        x1 = Multiply()([x1, s_branch[2]])
        x1 = Add()([x, x1])

        outBlock.append(x1)

        if IsDropout:
            outBlock.append(Dropout(IsDropout)(outBlock[-1]))
        return outBlock[-1]

    # Encoder

    x = conv_block(InInput=imageInput, InFilter=[32, 'M', 64, 'M', 128])
    x = Flatten()(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(InDropout)(x)
    x_out = Dense(43, activation='softmax', name='classification')(x)

    model = Model(inputs=[imageInput], outputs=[x_out], name=InName)

    return model


def self_attention(InInput,
                   InFilter,
                   IsDropout=0.2,
                   InDilationRate=(1, 1),
                   InNetSize=1,
                   InActivation=ReLU,
                   IsKernal_reg=True):
    outBlock = []
    s_branch = []

    x = conv_layer(InInput,
                   InKernelSize=1,
                   InFilter=InFilter // InNetSize,
                   InDilationRate=(1, 1),
                   InActivation=InActivation,
                   Iskernel_regularizer=IsKernal_reg)
    s_branch.append(
        conv_layer(x,
                   InFilter=InFilter // InNetSize,
                   InDilationRate=InDilationRate,
                   InActivation=InActivation,
                   Iskernel_regularizer=IsKernal_reg))
    s_branch.append(
        conv_layer(x,
                   InFilter=InFilter // InNetSize,
                   InDilationRate=InDilationRate,
                   InActivation=InActivation,
                   Iskernel_regularizer=IsKernal_reg))
    s_branch.append(
        conv_layer(x,
                   InFilter=InFilter // InNetSize,
                   InDilationRate=InDilationRate,
                   InActivation=InActivation,
                   Iskernel_regularizer=IsKernal_reg))

    x1 = Multiply()([s_branch[0], s_branch[1]])
    x1 = Softmax()(x1)
    x1 = Multiply()([x1, s_branch[2]])
    x1 = Add()([x, x1])

    outBlock.append(x1)

    if IsDropout:
        outBlock.append(Dropout(IsDropout)(outBlock[-1]))
    return outBlock[-1]


# Encoder


def get_denoise(InName):

    inImage = Input(shape=(None, None, 3))
    # encoder
    x = conv_block(inImage, [64, 'M', 256, 'M', 512, 'M', 512, 'M', 512],
                   Iskernel_regularizer=True,
                   InActivation=ReLU,
                   IsConvTranspose=True)

    # x2 = conv_block(x, [256, 'M', 512, 'M', 512],
    #                 Iskernel_regularizer=True,
    #                 InActivation=ReLU,
    #                 IsConvTranspose=True)

    # # ls = self_attention(x2, 64, 0.2, (2, 2))
    # x3 = conv_block(x2, [512, 256, 'U', 256],
    #                 InActivation=ReLU,
    #                 Iskernel_regularizer=True,
    #                 IsConvTranspose=True)

    x4 = conv_block(x, [256, 'U', 256, 'U', 128, 'U', 64],
                    InActivation=ReLU,
                    Iskernel_regularizer=True,
                    IsConvTranspose=True)

    # x5 = conv_block(x4, [64, 'U'],
    #                 InActivation=ReLU,
    #                 Iskernel_regularizer=True,
    #                 IsConvTranspose=True)

    x6 = Conv2D(3, 1, activation='tanh')(x4)

    return Model(inImage, x6, name=InName)


# run_mode(get_denoise, f'Denoise_Autoencoder')
