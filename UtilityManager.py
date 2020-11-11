import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import ConfigManager as config
import LogManager
import pickle as cPickle
import cv2
import numpy as np
from skimage import color, exposure, transform, io
from random import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# import matplotlib
# matplotlib.use('TkAgg')
# print(matplotlib.get_backend())
# plt.switch_backend('TkAgg')

import matplotlib.pyplot as plt
# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.callbacks import Callback
# import tensorflow.keras.backend.tensorflow_backend as K


def setGpuIdForCuda(InGPUID):
    os.environ['CUDA_VISIBLE_DEVICES'] = InGPUID


def disableWarning():
    warnings.filterwarnings("ignore")
    # 0 = all messagesare logged(defaultbehavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #import tensorflow as tf
    # old_v = tf.logging.get_verbosity()
    # tf.logging.set_verbosity(tf.logging.ERROR)


def lr_schedule(epoch):
    return config.LR * (0.1**int(epoch / 10))


def read_image(InPath):
    return io.imread(InPath)


def read_image_cv(InPath, InChannelOrder='BGR'):
    image = cv2.imread(InPath)
    if InChannelOrder == 'RGB':
        return image[:, :, ::-1]
    return cv2.imread(InPath)


def save_image_cv(InFileName, InImage):
    cv2.imwrite(InFileName, InImage)


# def loadDataFromPath(InPath, InTotalSample=None, InColor='blue'):
#     LogManager.displayLog(f'Loading <{InPath}> data..', InColor)
#     filesList = [
#         filename for filename in os.listdir(InPath)
#         if os.path.isfile(os.path.join(InPath, filename))
#     ]
#     filesList.sort()

#     Filetype = filesList[0].split('.')[-1]

#     if InTotalSample is None:
#         totalFile = len(filesList)
#     elif InTotalSample > len(filesList):
#         totalFile = len(filesList)
#     else:
#         totalFile = InTotalSample

#     tempList = FileLoader(totalFile, InPath, filesList, Filetype)

#     return tempList

# def FileLoader(totalFile, InPath, filesList, Filetype):
#     tempList = []
#     for i in range(totalFile):
#         if Filetype == 'ppm':
#             tempList.append(io.imread(os.path.join(InPath, filesList[i])))

#         if i % 100 == 0:
#             LogManager.displayLog(f'Loading {i}/{totalFile}', 'green')
#     return tempList


def saveDatasetInPklFormat(InPath_FileName, InDataset):
    # if not os.path.exists(InPath_FileName):
    LogManager.displayLog(f'Saving {InPath_FileName}', 'blue', 'bold')
    with open(InPath_FileName, 'wb') as fid:
        cPickle.dump(InDataset, fid, cPickle.HIGHEST_PROTOCOL)


def loadDatasetFromPklFormat(InPath, InLog=True):
    with open(InPath, 'rb') as fid:
        if InLog:
            LogManager.displayLog(f'Loading {InPath}', 'blue')
        outDataset = cPickle.load(fid, encoding='latin')
    return outDataset


# --learning rate functions -- #
def clr(epoch):
    base_lr = config.MIN_LR
    max_lr = config.MAX_LR
    step_size = config.STEP_SIZE
    mode = config.CLR_METHOD
    gamma = 1.
    # scale_fn = None
    scale_mode = 'cycle'

    # if scale_fn == None:
    if mode == 'triangular':
        scale_fn = lambda x: 1.
        scale_mode = 'cycle'
    elif mode == 'triangular2':
        scale_fn = lambda x: 1 / (2.**(x - 1))
        scale_mode = 'cycle'
    elif mode == 'exp_range':
        scale_fn = lambda x: gamma**(x)
        scale_mode = 'iterations'

    def circleLR():
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        if scale_mode == 'cycle':
            return base_lr + (max_lr - base_lr) * np.maximum(
                0, (1 - x)) * scale_fn(cycle)
        else:
            return base_lr + (max_lr - base_lr) * np.maximum(
                0, (1 - x)) * scale_fn(epoch)

    if epoch == 0:
        return base_lr
    else:
        return circleLR()


def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate,
    # and power of the polynomial
    power = 1.0

    # compute the new learning rate based on polynomial decay
    alpha = config.MIN_LR * (1 - (epoch / float(config.TOTAL_EPOCHS)))**power

    # return the new learning rate
    return alpha


class customLearningRateScheduler(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """
    def __init__(self, schedule, verbose=0):
        super(customLearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(self.epoch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(self.epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def saveWeights(InModel, InPath='Weights/', IsLogEnable=True):
    path = os.path.join(InPath)

    if IsLogEnable:
        LogManager.displayLog(f"Saving {path}", 'blue')

        # if self.multiGPU > 1:
        # get single GPU model weights
        #     singleGPUModel = self.model.layers[MODEL_INDEX]
        #     singleGPUModel.save(path + '.h5')
        # else:
    InModel.save_weights(path + '.h5')


def cycle_batch_size(epoch, InMaxBatchSize=64, InMinBatchSize=2):
    base_lr = config.MAX_BATCH_SIZE
    max_lr = config.MIN_BATCH_SIZE

    step_size = config.BATCH_CYCLE_STEPS
    mode = config.BATCH_CLR_METHOD
    gamma = 1.
    # scale_fn = None
    scale_mode = 'cycle'

    # if scale_fn == None:
    if mode == 'triangular':
        scale_fn = lambda x: 1.
        scale_mode = 'cycle'
    elif mode == 'triangular2':
        scale_fn = lambda x: 1 / (2.**(x - 1))
        scale_mode = 'cycle'
    elif mode == 'exp_range':
        scale_fn = lambda x: gamma**(x)
        scale_mode = 'iterations'

    def circleLR():
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        if scale_mode == 'cycle':
            return base_lr + (max_lr - base_lr) * np.maximum(
                0, (1 - x)) * scale_fn(cycle)
        else:
            return base_lr + (max_lr - base_lr) * np.maximum(
                0, (1 - x)) * scale_fn(epoch)

    if epoch == 0:
        return base_lr
    else:
        return int(circleLR())


def poly_decay_batch(epoch, power=1, totalEpoch=config.TOTAL_EPOCHS):
    # compute the new learning rate based on polynomial decay
    alpha = config.MAX_BATCH_SIZE * (1 - (epoch / float(totalEpoch)))**power

    # return the new batch size
    if alpha < config.MIN_BATCH_SIZE:
        alpha = config.MIN_BATCH_SIZE
    return int(alpha)


def loadWeights(InModel, InWeightPath, Log=True):
    if Log:
        LogManager.displayLog(f'Loading {InWeightPath} ..', 'cyan')
    # if self.multiGPU > 1:
    #     InModel.layers[MODEL_INDEX].load_weights(os.path.join(InWeightPath),
    #                                              by_name=True,
    #                                              skip_mismatch=True)
    # else:
    InModel.load_weights(os.path.join(InWeightPath),
                         by_name=True,
                         skip_mismatch=False)
    return InModel


def getListOfRandomNo(InRange, InNumberOfValue, InIsRepeatNo=False):
    return np.random.choice(range(InRange),
                            InNumberOfValue,
                            replace=InIsRepeatNo)


def showImage(InImage):
    plt.imshow(InImage)
    plt.show()


def augmentation(InImage):
    InImage = (InImage * 255).astype(np.uint8)
    # -- Image augmentation  -- #
    index = getListOfRandomNo(6, 6)
    noisyIndex = np.squeeze(getListOfRandomNo(6, 1))
    InImage = augment_method(index[noisyIndex], InImage)
    return (InImage / 255.).astype(np.float32)


def convertImageFloatToInt(InImage):
    InImage = (InImage * 255).astype(np.uint8)
    return InImage


def convertImageToFloat(InImage):
    return (InImage / 255)


def augment_method(noise_typ, image):
    # gauss
    if noise_typ == 0:
        return image
    if noise_typ == 1:
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 0.01
        return noisy
    # "salt-pepper"
    elif noise_typ == 2:
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [
            np.random.randint(0, i - 1, int(num_salt)) for i in image.shape
        ]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [
            np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape
        ]
        out[coords] = 0
        return out
    # "poisson"
    elif noise_typ == 3:
        vals = len(np.unique(image))
        vals = 2**np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    # "speckle"
    elif noise_typ == 4:
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * 0.01
        return noisy
    # light augmentation
    elif noise_typ == 5:
        out = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        out = np.array(out, dtype=np.float32)
        random_bright = .5 + np.random.uniform()
        out[:, :, 2] = out[:, :, 2] * random_bright
        out[:, :, 2][out[:, :, 2] > 255] = 255
        out = np.array(out, dtype=np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
        return out
    elif noise_typ == 6:
        # image = image.astype(np.uint8)
        out = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        out_image = np.zeros((out.shape[0], out.shape[1], 3))
        out_image[:, :, 0] = out
        out_image[:, :, 1] = out
        out_image[:, :, 2] = out

        return out_image
    elif noise_typ == 7:
        # image = image.astype(np.uint8)
        out = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        out = np.array(out, dtype=np.float32)
        random_bright = .5 + np.random.uniform()
        out[:, :, 2] = out[:, :, 2] * random_bright
        out[:, :, 2][out[:, :, 2] > 255] = 255
        out = np.array(out, dtype=np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_HSV2RGB)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)

        out_image = np.zeros((out.shape[0], out.shape[1], 3))
        out_image[:, :, 0] = out
        out_image[:, :, 1] = out
        out_image[:, :, 2] = out

        return out_image


def data_attack_with_noise(InImage):
    InImage = (InImage * 255).astype(np.uint8)
    # -- Image augmentation  -- #
    index = getListOfRandomNo(4, 4)
    noisyIndex = np.squeeze(getListOfRandomNo(4, 1))
    InImage = augment_method_no_light(index[noisyIndex], InImage)
    return (InImage / 255.).astype(np.float32)


def augment_method_no_light(noise_typ, image):
    # gauss
    if noise_typ == 0:
        return image
    if noise_typ == 1:
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 0.01
        return noisy
    # "salt-pepper"
    elif noise_typ == 2:
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [
            np.random.randint(0, i - 1, int(num_salt)) for i in image.shape
        ]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [
            np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape
        ]
        out[coords] = 0
        return out
    # "poisson"
    elif noise_typ == 3:
        vals = len(np.unique(image))
        vals = 2**np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    # "speckle"
    elif noise_typ == 4:
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * 0.01
        return noisy


def imageToArray(InImage, InArray):
    x_offset = y_offset = 0
    InArray[y_offset:y_offset + InImage.shape[0],
            x_offset:x_offset + InImage.shape[1]] = InImage
    return InArray


def resizeImageAndPreserveAspectRatio(image, scale=1, InMaxImageSize=None):
    h, w = image.shape[:2]

    if InMaxImageSize is None:
        InMaxImageSize = config.IMG_SIZE

    scale_percent = 100

    while True:
        width = int(w * scale_percent / 100)
        height = int(h * scale_percent / 100)
        if width < InMaxImageSize and height < InMaxImageSize:
            break
        scale_percent -= scale

    return cv2.resize(image, (width, height), cv2.INTER_AREA)


def flipDataset(InDataset, flipOption=0, IsImage=True):
    totalSample = len(InDataset)
    # horizontally
    for i in range(totalSample):
        image = InDataset[i]
        if flipOption == 0:
            image = cv2.flip(image, 0)
        elif flipOption == 1:
            image = cv2.flip(image, 1)
        elif flipOption == 2:
            image = cv2.flip(image, -1)

        if IsImage is False:
            InDataset[i] = np.expand_dims(image, 3)
        else:
            InDataset[i] = image
    return InDataset


def create_pkl_file(InPath, InFileName, InData):
    saveDatasetInPklFormat(f'{InPath}/{InFileName}.pkl', InData)


def create_lr_decay(decay_step_size, lr_decay_rate):
    def lr_decay(epoch, lr):
        if epoch % decay_step_size == 0:
            lr = lr * lr_decay_rate
        return lr

    return lr_decay


def get_confusion_matrix(y_true, y_pred, InFileName=None):
    def plot_confusion_matrix(y_true,
                              y_pred,
                              classes,
                              normalise=False,
                              title=None,
                              cmap=plt.cm.Blues):

        if not title:
            if normalise:
                title = 'Normalised confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        cm = confusion_matrix(y_true, y_pred)
        if normalise:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalised confusion matrix")
        else:
            pass
            # print('Confusion matrix, without normalization')

        fig, ax = plt.subplots(figsize=(15, 15))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes,
               yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(),
                 rotation=0,
                 ha="right",
                 rotation_mode="anchor")

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                if value == 0 or value == 1:
                    value = int(value)
                ax.text(j,
                        i,
                        round(value, 2),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    np.set_printoptions(precision=2)

    class_names = range(43)
    plot_confusion_matrix(y_true,
                          y_pred,
                          classes=class_names,
                          normalise=True,
                          title='Normalised confusion matrix')

    if InFileName:
        plt.savefig(InFileName)

    plt.tight_layout()
    # plt.show()
    plt.close()


def plot_chart(InX,
               InY,
               InXtitle=None,
               InYtitle=None,
               InChartTitle=None,
               saveFigName=None):
    plt.style.use('fivethirtyeight')
    # fig, ax = plt.subplots()
    # ax.scatter(InX, InY)
    plt.plot(InX, InY)
    if InXtitle is not None:
        plt.xlabel(InXtitle)
    if InYtitle is not None:
        plt.ylabel(InYtitle)
    if InChartTitle is not None:
        plt.title(InChartTitle)
    plt.tight_layout()
    if saveFigName is not None:
        plt.savefig(f'{saveFigName}.png')

    plt.show()


# def attack_augment_image(x_batch, InBatchSize=None):
#     list_of_attack = [
#         salt_pepper, gaussian_noise, create_box, poission, speckle
#     ]

#     # list_of_attack_name = [
#     #     'salt_pepper', 'gaussian_noise', 'create_box', 'poission', 'speckle'
#     # ]

#     for i in range(x_batch.shape[0]):
#         shuffle(list_of_attack)
#         attackIndex = np.random.randint(0, 4)
#         noise_intensity = np.random.randint(1, 5)
#         maxBoxSize = np.random.randint(10, 15)
#         box_size = np.random.randint(1, maxBoxSize)
#         attack = list_of_attack[attackIndex]

#         x_batch[i], _ = attack(x_batch[i], noise_intensity,
#                                int(maxBoxSize / box_size))
#     return x_batch


def shuffle_dataset(X, y):
    return shuffle(X, y)


def class_cat(InPred, InLabelNames):
    if InPred.shape[-1] > 1:
        InPred = InPred.argmax(axis=-1)
    else:
        InPred = (InPred > 0.5).astype('int32')
    name = InLabelNames[int(InPred)]
    return name


def preprocess_img(image,
                   IsHistrogarmFix=True,
                   IsRescaleImage=True,
                   IsCenteralCrop=True,
                   IsDemo=False,
                   resize=(config.IMG_SIZE, config.IMG_SIZE)):
    if IsHistrogarmFix:
        hsv = color.rgb2hsv(image)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        image = color.hsv2rgb(hsv)
    # central square crop
    if IsCenteralCrop:
        if IsDemo:
            min_side = min(image.shape[:-1])
            centre = image.shape[0] // 2, image.shape[1] // 2
            image = image[centre[0] - min_side // 4:centre[0] + min_side // 4,
                          centre[1] - min_side // 4:centre[1] +
                          min_side // 4, :]
        else:
            min_side = min(image.shape[:-1])
            centre = image.shape[0] // 2, image.shape[1] // 2
            image = image[centre[0] - min_side // 2:centre[0] + min_side // 2,
                          centre[1] - min_side // 2:centre[1] +
                          min_side // 2, :]

    # out_img = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
    # image = Utility.imageToArray(Utility.resizeImageAndPreserveAspectRatio(img), out_img)

    # rescale to standard size
    if IsRescaleImage:
        image = cv2.resize(image, resize, cv2.INTER_AREA)

        # Histogram normalization in v channel

    return image


def preprocess_img_for_image_gen(image,
                                 IsHistrogarmFix=True,
                                 IsRescaleImage=False,
                                 IsCenteralCrop=False):

    # Histogram normalization in v channel
    if IsHistrogarmFix:
        hsv = color.rgb2hsv(image)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        image = color.hsv2rgb(hsv)

    # central square crop
    if IsCenteralCrop:
        min_side = min(image.shape[:-1])
        centre = image.shape[0] // 2, image.shape[1] // 2
        image = image[centre[0] - min_side // 2:centre[0] + min_side // 2,
                      centre[1] - min_side // 2:centre[1] + min_side // 2, :]

    # out_img = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.float32)
    # image = Utility.imageToArray(Utility.resizeImageAndPreserveAspectRatio(img), out_img)

    # rescale to standard size
    if IsRescaleImage:
        image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE),
                           cv2.INTER_AREA)

    return image


def meanSquareError(image, pred):
    return np.mean((image - pred)**2)


import glob


def create_pipeline_out_folder(InPath):
    try:
        os.mkdir(InPath)
        anomaly = f'{InPath}/anomaly/'
        anomalyRec = f'{InPath}/anomaly/reconstruct/'
        anomalySig = f'{InPath}/anomaly/signs/'
        detection = f'{InPath}/detection/'
        non_det = f'{InPath}/non_detected/'
        recog = f'{InPath}/recog/'
        allFile = [anomaly, anomalySig, anomalyRec, detection, non_det, recog]
        for path in allFile:
            os.mkdir(path)
    except:
        LogManager.displayLog(f' Folder: {InPath} exits!')


def empty_output_folder():
    InPath = '/home/anish/Documents/CARAMEL/Project/traffic_signs_v2/outputs/pipeline_output'
    anomaly = f'{InPath}/anomaly/'
    anomalyRec = f'{InPath}/anomaly/reconstruct/'
    anomalySig = f'{InPath}/anomaly/signs/'
    detection = f'{InPath}/detection/'
    non_det = f'{InPath}/non_detected/'
    recog = f'{InPath}/recog/'
    allFile = [anomaly, anomalySig, anomalyRec, detection, non_det, recog]
    for path in allFile:
        fileList = glob.glob(f'{path}/*.png')
        for f in fileList:
            os.remove(f)

    # os.remove("/tmp/<file_name>.txt")


#empty_output_folder()


def setKerasMemory(limit=0.3):
    from tensorflow import ConfigProto as tf_ConfigProto
    from tensorflow import Session as tf_Session
    from tensorflow.keras.backend.tensorflow_backend import set_session
    config = tf_ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = limit
    set_session(tf_Session(config=config))


def set_GPU_memory():
    import tensorflow as tf
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    cofig.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    from tensorflow.keras import backend as k
    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))


def convertImageToGray(InImage):
    return np.expand_dims(
        np.clip(
            convertImageToFloat(
                cv2.cvtColor(convertImageFloatToInt(InImage),
                             cv2.COLOR_RGB2GRAY)), 0, 1), 2)


def plot_3d(InX, InY, InZ):

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib as mpl
    mpl.style.use('ggplot')

    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(InX, InY)
    # R = np.sqrt(X**2 + Y**2)
    Z = np.sin(InZ)

    # Plot the surface.
    surf = ax.plot_surface(X,
                           Y,
                           Z,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_scatter_point(xs, ys, zs, InYTest):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # n = 100

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25),
    #                           ('b', '^', -30, -5)]:
    # xs = randrange(n, 23, 32)
    # ys = randrange(n, 0, 100)
    # zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=InYTest, marker='1')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def preview_image_batch(InFileName, InImageBatch, TotalImage=100):
    np.random.seed(1)
    # image_batch   = InImageBatch[:TotalImage]
    image_batch = InImageBatch[np.random.randint(0,
                                                 InImageBatch.shape[0],
                                                 size=TotalImage)]

    def plot_figure(InImages, InFileName):
        plot_fig = np.abs(np.sqrt(TotalImage))
        plt.figure(figsize=(plot_fig, plot_fig))
        for i in range(InImages.shape[0]):
            plt.subplot(plot_fig, plot_fig, i + 1)
            image = InImages[i, :, :, :]
            plt.imshow((image * 255).astype(np.uint8))
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(InFileName)
        plt.close()

    plot_figure(image_batch, InFileName)


def load_mask(index, InMaskImages):
    if index == 13:
        return (InMaskImages[0])
    elif index >= 18 and index <= 31 or index == 11:
        return (InMaskImages[1])
    elif index == 12:
        return (InMaskImages[2])
    elif index >= 0 and index <= 10 or index >= 15 and index <= 17 or index >= 32 and index <= 42:
        return (InMaskImages[3])
    elif index == 14:
        return (InMaskImages[4])


def apply_mask(image, index, InMaskList):
    return np.clip(image - load_mask(index, InMaskList), 0, 1)


def convertInRange(InImage, InMin, InMax):
    return (InImage - np.min(InImage)) * (InMax - InMin) / (
        np.max(InImage) - np.min(InImage)) + InMin


def MaskMeanSquareError(yTrue, yPred):
    return np.sqrt(np.sqrt(np.mean(np.square(yTrue - yPred))))


def get_learning_rate(epoch):
    if config.LEARNING_RATE_OPTION == 'cycle':
        return clr(epoch)
    else:
        return poly_decay(epoch)


def get_batch_size(epoch):
    if config.BATCH_CLR_METHOD == 'cycle':
        return cycle_batch_size(epoch)
    else:
        return poly_decay_batch(epoch)


def train_test_split_(InX, InY, InSplit=0.2, seed=0):
    X_train, X_test, Y_train, Y_test = train_test_split(InX,
                                                        InY,
                                                        test_size=InSplit,
                                                        random_state=seed)
    return X_train, X_test, Y_train, Y_test
